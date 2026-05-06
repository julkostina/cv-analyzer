import json
import logging
import re
import traceback
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings
from app.models import (
    CVAnalysisRequest,
    CVAnalysisResponse,
    CertificateItem,
    EducationItem,
    ExperienceItem,
    ProjectItem,
)
from app.services.semantic_matcher import SemanticMatchResult, compute_semantic_match
from app.utils.text_preprocess import normalize_text_for_pipeline

logger = logging.getLogger(__name__)

_HUMAN_PROMPT = (
    "Return the FULL JSON: skills, experience, certificates, education, projects, "
    "analysis, matched_competencies, missing_competencies, match_score (null), "
    "match_score_reasoning (null), recommendations. Use [] for empty lists.\n\nCV:\n{cv_text}\n{job_description_section}"
)

_CV_ANALYZER_SYSTEM = """You are an AI CV analyzer.

Task: extract structured data from the CV text and return a COMPLETE JSON object every time.

CRITICAL: all user-facing text (summaries, recommendations, skills lines, matched_competencies, missing_competencies, employer/title/name fields, etc.) must be in English. Technology names (React, TypeScript) stay as usual.

Required arrays: skills, experience, certificates, education, projects — use [] when empty. Do not use null for these arrays.

analysis — object with keys summary, strengths, weaknesses (English). strengths and weaknesses may be strings or short lists.

recommendations — at least one string in English.

If a job description is provided:
- matched_competencies — job requirements/skills clearly supported by the CV.
- missing_competencies — job requirements weak or missing in the CV.
- match_score and match_score_reasoning must always be null (the server fills them via semantic scoring).

If no job description:
- matched_competencies: [], missing_competencies: [].
- match_score and match_score_reasoning: null.
- recommendations — general market-oriented advice in English.

Classification:
- EXPERIENCE — jobs, internships, freelance with duties. Not courses.
- CERTIFICATES — courses, training (Meta, Coursera, Epam, AWS) without job duties.
- EDUCATION — universities, degrees.
- PROJECTS — pet projects, GitHub without formal employment.

PDF text order may be broken — classify by content.

Rules: Meta / Course / Program without work tasks → CERTIFICATE. Words "course", "training", "program" in a learning context → CERTIFICATE. Work tasks described → EXPERIENCE. GitHub link and no hiring company → PROJECT.
"""

_OLLAMA_FALLBACK_SYSTEM = (
    "Return exactly ONE valid JSON object and nothing else: no markdown, no code fences, no text before or after. "
    "Keys (English): skills, experience, certificates, education, projects, analysis, "
    "matched_competencies, missing_competencies, match_score, match_score_reasoning, recommendations. "
    "All user-facing string values in English. recommendations must have at least one item."
)

_OLLAMA_FALLBACK_HUMAN = (
    "Fill JSON using the same classification rules as the main prompt (experience / certificates / education / projects). "
    "Empty sections as []. match_score and match_score_reasoning must be null.\n\n"
    "CV:\n{cv_text}\n{job_description_section}"
)

_OLLAMA_FALLBACK_RULES = (
    "Classification: EXPERIENCE — jobs, internship, freelance; CERTIFICATES — courses without job duties; "
    "EDUCATION — school/university; PROJECTS — side projects. Meta/Course/Program without work tasks → CERTIFICATE."
)


class CVAnalysisOutput(BaseModel):
    skills: Optional[List[str]] = Field(None)
    experience: Optional[List[ExperienceItem]] = Field(None)
    certificates: Optional[List[CertificateItem]] = Field(None)
    education: Optional[List[EducationItem]] = Field(None)
    projects: Optional[List[ProjectItem]] = Field(None)
    analysis: Optional[Dict[str, Any]] = Field(None)
    match_score: Optional[float] = Field(None, ge=0, le=1)
    match_score_reasoning: Optional[str] = Field(None)
    recommendations: List[str] = Field(..., min_length=1)
    matched_competencies: Optional[List[str]] = Field(None)
    missing_competencies: Optional[List[str]] = Field(None)


def _parse_llm_json_blob(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    start = t.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(t[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _cv_output_from_parsed_dict(raw: Dict[str, Any]) -> Optional[CVAnalysisOutput]:
    data = dict(raw)
    recs = data.get("recommendations")
    if not recs or not isinstance(recs, list) or len(recs) == 0:
        data["recommendations"] = ["Review your CV and try the analysis again."]
    try:
        return CVAnalysisOutput.model_validate(data)
    except Exception:
        return None


def _normalize_llm_result(result: CVAnalysisOutput) -> tuple[CVAnalysisOutput, bool]:
    incomplete = False
    if result.skills is None:
        result = result.model_copy(update={"skills": []})
        incomplete = True
    if result.experience is None:
        result = result.model_copy(update={"experience": []})
        incomplete = True
    if result.certificates is None:
        result = result.model_copy(update={"certificates": []})
        incomplete = True
    if result.education is None:
        result = result.model_copy(update={"education": []})
        incomplete = True
    if result.projects is None:
        result = result.model_copy(update={"projects": []})
        incomplete = True
    if result.analysis is None:
        result = result.model_copy(update={"analysis": {}})
        incomplete = True
    if result.matched_competencies is None:
        result = result.model_copy(update={"matched_competencies": []})
        incomplete = True
    if result.missing_competencies is None:
        result = result.model_copy(update={"missing_competencies": []})
        incomplete = True
    return result, incomplete


def _is_extraction_empty(result: CVAnalysisOutput) -> bool:
    return (
        (not result.skills or len(result.skills) == 0)
        and (not result.experience or len(result.experience) == 0)
        and (not result.education or len(result.education) == 0)
        and (not result.certificates or len(result.certificates) == 0)
        and (not result.projects or len(result.projects) == 0)
        and (not result.analysis or result.analysis == {})
    )


def _ollama_empty_extraction_hint(model_name: str) -> str:
    """Hints that do not suggest the same model tag the user already runs."""
    m = model_name.lower()
    parts: List[str] = []
    small = any(tok in m for tok in ("3.2", ":3b", "1b", "1.5b", ":2b", "2.3b")) and "8b" not in m and "70b" not in m
    if small:
        parts.append("Small models often fail on complex JSON schemas.")
        parts.append("Try OLLAMA_MODEL=llama3:8b or llama3.1:8b (ollama pull …).")
    elif "llama3" in m and "8b" in m:
        parts.append("Even llama3:8b can return empty structured output (Ollama + LangChain JSON schema).")
        parts.append(
            "Try: mistral / qwen2.5 / llama3.1:8b; upgrade Ollama and re-pull the model; or ENVIRONMENT=production with GEMINI_API_KEY."
        )
    else:
        parts.append("Try another model (mistral, qwen2.5, llama3.1:8b) or Gemini in production.")
    parts.append(
        f"If needed raise OLLAMA_NUM_CTX (currently {settings.ollama_num_ctx}). Keep MAX_CV_CHARS_FOR_LLM=0 to avoid truncating the CV."
    )
    return " ".join(parts)


def _empty_extraction_error(
    backend: str,
    model_name: str,
    cv_chars_sent: int,
    cv_chars_total: int,
) -> str:
    if cv_chars_sent < cv_chars_total:
        cv_detail = f"CV sent to the model: {cv_chars_sent} characters (truncated from {cv_chars_total})."
    else:
        cv_detail = f"CV length: {cv_chars_total} characters."
    base = (
        "Analysis failed: the model returned no structured data. "
        "All required sections (experience, education, certificates, projects, skills) are empty. "
        f"Backend: {backend}, model: {model_name}. {cv_detail} "
    )
    if backend == "Ollama":
        return base + _ollama_empty_extraction_hint(model_name)
    return base + "Try a shorter CV or verify model configuration."


def _experience_text_from_result(result: CVAnalysisOutput) -> str:
    parts: List[str] = []
    for e in result.experience or []:
        bits = [x for x in (e.title, e.employer, e.duration) if x]
        if bits:
            parts.append(" — ".join(bits))
    return "\n".join(parts)


def _semantic_reasoning_text(sem: SemanticMatchResult) -> str:
    return (
        "Match score from cosine similarity of resume and job embeddings "
        "(Sentence Transformers multilingual model).\n\n"
        f"• Skills block similarity: {sem.skills_similarity:.1%}\n"
        f"• Experience vs requirements similarity: {sem.experience_similarity:.1%}\n"
        f"• Full-text similarity: {sem.overall_similarity:.1%}\n\n"
        f"Weighted overall score: {sem.score:.1%}."
    )


def _build_success_response(
    cv_text: str,
    result: CVAnalysisOutput,
    job_description: Optional[str],
) -> CVAnalysisResponse:
    recs = result.recommendations or ["Review your CV and try again."]
    matched = list(result.matched_competencies or [])
    missing = list(result.missing_competencies or [])

    match_score: Optional[float] = None
    match_reason: Optional[str] = None
    semantic_breakdown: Optional[Dict[str, float]] = None

    job_stripped = (job_description or "").strip()
    if job_stripped and settings.use_semantic_matching:
        try:
            cv_exp = _experience_text_from_result(result)
            sem = compute_semantic_match(
                cv_skills=result.skills or [],
                cv_experience_text=cv_exp or cv_text[:8000],
                cv_full_text=cv_text,
                job_requirements_text=job_stripped,
                job_full_text=job_stripped,
            )
            match_score = sem.score
            semantic_breakdown = {
                "skills_similarity": sem.skills_similarity,
                "experience_similarity": sem.experience_similarity,
                "overall_similarity": sem.overall_similarity,
            }
            match_reason = _semantic_reasoning_text(sem)
            logger.info(
                "Semantic match score=%.3f (skills=%.3f exp=%.3f overall=%.3f)",
                sem.score,
                sem.skills_similarity,
                sem.experience_similarity,
                sem.overall_similarity,
            )
        except Exception:
            logger.exception("Semantic matching failed; match_score left unset")
            match_score = result.match_score
            match_reason = result.match_score_reasoning

    return CVAnalysisResponse(
        success=True,
        extracted_text=cv_text,
        analysis=result.analysis,
        skills=result.skills,
        experience=result.experience,
        certificates=result.certificates,
        education=result.education,
        projects=result.projects,
        match_score=match_score,
        match_score_reasoning=match_reason,
        recommendations=recs,
        matched_competencies=matched if job_stripped else [],
        missing_competencies=missing if job_stripped else [],
        semantic_breakdown=semantic_breakdown,
        error=None,
    )


def _job_section(job_description: Optional[str]) -> str:
    if job_description and job_description.strip():
        return (
            f"\n\nJob description (requirements):\n{job_description.strip()}\n\n"
            "Compare the CV to the requirements. Fill matched_competencies and missing_competencies in English. "
            "Leave match_score and match_score_reasoning null. Tailor recommendations to this role, in English."
        )
    return (
        "\n\nNo job description provided. matched_competencies and missing_competencies must be []. "
        "match_score and match_score_reasoning must be null. Give general market-oriented recommendations in English."
    )


class CVAnalyzer:
    def __init__(self) -> None:
        self._ollama_llm = None
        self._gemini_llm = None

    async def analyze_cv(
        self,
        cv_text: str,
        request: Optional[CVAnalysisRequest] = None,
    ) -> CVAnalysisResponse:
        cv_text = normalize_text_for_pipeline(cv_text or "")
        job = request.job_description if request else None
        if job:
            job = normalize_text_for_pipeline(job)

        if settings.environment == "development":
            return await self._analyze_with_ollama(cv_text, job)
        return await self._analyze_with_gemini(cv_text, job)

    async def _ollama_raw_json_fallback(
        self,
        cv_text_for_prompt: str,
        job_description_section: str,
    ) -> Optional[CVAnalysisOutput]:
        if not self._ollama_llm:
            return None
        sys_content = f"{_OLLAMA_FALLBACK_SYSTEM}\n\n{_OLLAMA_FALLBACK_RULES}"
        human_content = _OLLAMA_FALLBACK_HUMAN.format(
            cv_text=cv_text_for_prompt,
            job_description_section=job_description_section,
        )
        messages = [
            SystemMessage(content=sys_content),
            HumanMessage(content=human_content),
        ]
        try:
            resp = await self._ollama_llm.ainvoke(messages)
        except Exception:
            logger.exception("Ollama JSON fallback: invoke failed")
            return None
        raw = resp.content
        if isinstance(raw, list):
            pieces: List[str] = []
            for block in raw:
                if isinstance(block, str):
                    pieces.append(block)
                elif isinstance(block, dict):
                    pieces.append(str(block.get("text", block)))
                else:
                    pieces.append(str(block))
            raw = "".join(pieces)
        blob = _parse_llm_json_blob(str(raw))
        if not blob:
            logger.warning(
                "Ollama JSON fallback: no JSON in model reply (preview: %.220s)",
                str(raw).replace("\n", " "),
            )
            return None
        return _cv_output_from_parsed_dict(blob)

    async def _run_structured_chain(
        self,
        cv_text: str,
        cv_text_for_prompt: str,
        job_description: Optional[str],
        structured_llm: Any,
        backend: str,
        model_name: str,
        *,
        ollama_json_fallback: bool = False,
    ) -> CVAnalysisResponse:
        job_description_section = _job_section(job_description)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", _CV_ANALYZER_SYSTEM),
                ("human", _HUMAN_PROMPT),
            ]
        )
        try:
            chain = prompt_template | structured_llm
            result: CVAnalysisOutput = await chain.ainvoke(
                {
                    "cv_text": cv_text_for_prompt,
                    "job_description_section": job_description_section,
                }
            )
            result, incomplete = _normalize_llm_result(result)
            if incomplete:
                logger.warning("LLM returned incomplete response; filled defaults.")
            if _is_extraction_empty(result):
                if ollama_json_fallback and self._ollama_llm is not None:
                    fb = await self._ollama_raw_json_fallback(
                        cv_text_for_prompt,
                        job_description_section,
                    )
                    if fb is not None:
                        fb, _fb_inc = _normalize_llm_result(fb)
                        if not _is_extraction_empty(fb):
                            logger.info("Ollama: recovered via raw JSON fallback after empty structured output")
                            return _build_success_response(cv_text, fb, job_description)
                return CVAnalysisResponse(
                    success=False,
                    extracted_text=cv_text,
                    error=_empty_extraction_error(
                        backend,
                        model_name,
                        len(cv_text_for_prompt),
                        len(cv_text),
                    ),
                )
            return _build_success_response(cv_text, result, job_description)
        except Exception as e:
            error_msg = f"Analysis error: {e!s}\n{traceback.format_exc()}"
            return CVAnalysisResponse(success=False, extracted_text=cv_text, error=error_msg)

    def _cv_text_for_prompt(self, cv_text: str) -> str:
        max_chars = settings.max_cv_chars_for_llm
        if max_chars > 0 and len(cv_text) > max_chars:
            logger.warning("CV truncated from %d to %d chars for LLM context", len(cv_text), max_chars)
            return (
                cv_text[:max_chars]
                + "\n\n[CV text truncated for model context. Extract from the text above.]"
            )
        return cv_text

    async def _analyze_with_ollama(
        self,
        cv_text: str,
        job_description: Optional[str] = None,
    ) -> CVAnalysisResponse:
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("langchain-ollama package is required for development")

        cv_text_for_prompt = self._cv_text_for_prompt(cv_text)

        if not self._ollama_llm:
            self._ollama_llm = ChatOllama(
                model=settings.ollama_model,
                temperature=0.2,
                num_ctx=settings.ollama_num_ctx,
            )
            logger.info(
                "Using Ollama model: %s (num_ctx=%s)",
                settings.ollama_model,
                settings.ollama_num_ctx,
            )

        structured_llm = self._ollama_llm.with_structured_output(CVAnalysisOutput)
        return await self._run_structured_chain(
            cv_text,
            cv_text_for_prompt,
            job_description,
            structured_llm,
            "Ollama",
            settings.ollama_model,
            ollama_json_fallback=True,
        )

    async def _analyze_with_gemini(
        self,
        cv_text: str,
        job_description: Optional[str] = None,
    ) -> CVAnalysisResponse:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("langchain-google-genai package is required for production")

        cv_text_for_prompt = self._cv_text_for_prompt(cv_text)

        if not self._gemini_llm:
            self._gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.gemini_api_key,
                temperature=0.3,
            )

        structured_llm = self._gemini_llm.with_structured_output(CVAnalysisOutput)
        return await self._run_structured_chain(
            cv_text,
            cv_text_for_prompt,
            job_description,
            structured_llm,
            "Gemini",
            "gemini-2.5-flash",
        )
