import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from app.config import settings
from app.models import (
    CVAnalysisRequest,
    CVAnalysisResponse,
    ExperienceItem,
    CertificateItem,
    EducationItem,
    ProjectItem,
)

logger = logging.getLogger(__name__)


# System prompt: CV analyzer role and classification rules (content-based, not position)
_CV_ANALYZER_SYSTEM = """You are an AI CV analyzer. Extract structured information from the CV and return a COMPLETE JSON with ALL fields every time.

CRITICAL: You MUST include every field. Use empty list [] when there are no items (do NOT use null for experience, certificates, education, projects, skills). Required: skills (array), experience (array), certificates (array), education (array), projects (array), analysis (object with summary/strengths/weaknesses or empty object), recommendations (array, at least one string). If job description is provided also: match_score (number 0-1), match_score_reasoning (see format below).

PDF text may have broken order — use content only to classify, not position.

Classification: EXPERIENCE = real jobs, internships, freelance (with duties/tech). CERTIFICATES = courses, Meta/Coursera/Epam/AWS, no job duties. EDUCATION = universities, degrees. PROJECTS = pet/GitHub projects. Use English for all values.

match_score estimation rules (use when job description is provided):
- Formula: start from base 0.5. Add points for each job requirement met; subtract for each critical requirement missing or weak. Clamp result to [0, 1].
- Add points (examples): +0.05 to +0.15 per required skill present in CV; +0.1 for relevant experience (internship/job in same domain); +0.05 for each year of relevant experience beyond 1; +0.05 for relevant education or certificates.
- Subtract points (examples): -0.1 to -0.2 per critical requirement missing; -0.05 to -0.1 per important requirement weak or not shown; -0.05 if years of experience below job ask.
- match_score_reasoning MUST contain: (1) One line stating the formula you used, e.g. "Formula: base 0.5, then add/subtract per requirement." (2) A breakdown listing each factor with points and reason, e.g. "+0.1 React and TypeScript match job; +0.1 front-end internship; -0.15 no 5+ years experience; -0.05 SEO not mentioned." (3) One line with the calculation and final score, e.g. "Total: 0.5 + 0.1 + 0.1 - 0.15 - 0.05 = 0.5." Use English. Do NOT use match_score_reasoning for general advice or recommendations; that goes in the recommendations field only.

2. CERTIFICATES
Include: online courses, training programs, Meta/Coursera/Epam/AWS training, any learning without job duties.
Even if dates are present — this is NOT work experience.

3. EDUCATION
Include: universities, degrees, bachelor/master.

4. PROJECTS
Pet projects or GitHub projects without official employment.

Additional rules:
- If organization = Meta / Course / Program → CERTIFICATE.
- If words "курс", "program", "training" appear → CERTIFICATE.
- If there is a description of work tasks → EXPERIENCE.
- If entry has a GitHub link and no company is specified → PROJECT.

Return the result strictly in the required JSON schema (experience, certificates, education, projects). Use English for field values. Also provide: skills (list), analysis (summary, strengths, weaknesses), match_score (0-1 if job description given), match_score_reasoning (formula + breakdown + total, as above), recommendations. When job description is given, match_score_reasoning must be the scoring breakdown only, not general advice."""

# Pydantic model for structured LLM output (uses same item types as API response)
class CVAnalysisOutput(BaseModel):
    """Structured output: experience, certificates, education, projects + skills, analysis, match_score, recommendations."""
    skills: Optional[List[str]] = Field(None, description="List of skills/technologies")
    experience: Optional[List[ExperienceItem]] = Field(
        None,
        description="Only real work: jobs, internships, freelance. Exclude courses and pet projects.",
    )
    certificates: Optional[List[CertificateItem]] = Field(
        None,
        description="Online courses, Meta/Epam/Coursera/AWS training, any learning without job duties.",
    )
    education: Optional[List[EducationItem]] = Field(
        None,
        description="Universities, degrees, bachelor/master.",
    )
    projects: Optional[List[ProjectItem]] = Field(
        None,
        description="Pet projects, GitHub projects without official employment.",
    )
    analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Short summary, strengths, weaknesses",
    )
    match_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="How well the CV matches the position requirements (0-1). Only set when job description/position URL is provided; otherwise null.",
    )
    match_score_reasoning: Optional[str] = Field(
        None,
        description="Scoring breakdown in English: (1) Formula used (e.g. base 0.5 + add/subtract per requirement). (2) Point-by-point: each +X or -X with reason (e.g. '+0.1 React in CV', '-0.15 missing 5y exp'). (3) Total line (e.g. 'Total: 0.5+0.1-0.15=0.45'). Not recommendations—only score reasoning.",
    )
    recommendations: List[str] = Field(
        ...,
        min_length=1,
        description="At least one recommendation. If no job description: general market requirements. If job description given: tailored to that position. Cannot be empty.",
    )


def _normalize_llm_result(result: CVAnalysisOutput) -> tuple[CVAnalysisOutput, bool]:
    """Replace null list/analysis with []/{} so API response is consistent. Returns (normalized result, True if any fallback was applied)."""
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
    return result, incomplete


def _is_extraction_empty(result: CVAnalysisOutput) -> bool:
    """True if LLM returned no meaningful extraction (all lists empty, analysis empty)."""
    return (
        (not result.skills or len(result.skills) == 0)
        and (not result.experience or len(result.experience) == 0)
        and (not result.education or len(result.education) == 0)
        and (not result.certificates or len(result.certificates) == 0)
        and (not result.projects or len(result.projects) == 0)
        and (not result.analysis or result.analysis == {})
    )


def _empty_extraction_error(
    backend: str,
    model_name: str,
    cv_chars_sent: int,
    cv_chars_total: int,
) -> str:
    """Build a specific error message when LLM returns no structured extraction."""
    if cv_chars_sent < cv_chars_total:
        cv_detail = f"CV sent to model: {cv_chars_sent} chars (truncated from {cv_chars_total} total)."
    else:
        cv_detail = f"CV length: {cv_chars_total} chars."
    if backend == "Ollama":
        return (
            f"Analysis failed: the model returned no structured data (experience, education, skills, etc. are all empty). "
            f"Backend: {backend}, model: {model_name}. {cv_detail} "
            f"Suggestions: set OLLAMA_MODEL=llama3.2:8b in .env for a stronger model; or set MAX_CV_CHARS_FOR_LLM=0 to disable truncation and rely on a larger context; or run with ENVIRONMENT=production (Gemini)."
        )
    return (
        f"Analysis failed: the model returned no structured data (experience, education, skills, etc. are all empty). "
        f"Backend: {backend}, model: {model_name}. {cv_detail} "
        f"Try a shorter CV or check the model configuration."
    )


class CVAnalyzer:
    
    def __init__(self):
        self._ollama_llm = None
        self._gemini_llm = None
    
    async def analyze_cv(
        self, 
        cv_text: str, 
        request: Optional[CVAnalysisRequest] = None
    ) -> CVAnalysisResponse:
        
        if settings.environment == "development":
            return await self._analyze_with_ollama(cv_text, request.job_description if request else None)
        else:
            return await self._analyze_with_gemini(cv_text, request.job_description if request else None)
    
    async def _analyze_with_ollama(
        self, 
        cv_text: str, 
        job_description: Optional[str] = None
    ) -> CVAnalysisResponse:
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError:
            raise ImportError("langchain-ollama package is required for development")

        # Truncate CV so prompt fits in context (full text still returned as extracted_text)
        max_chars = getattr(settings, "max_cv_chars_for_llm", 0) or 0
        if max_chars > 0 and len(cv_text) > max_chars:
            cv_text_for_prompt = cv_text[:max_chars] + "\n\n[CV text truncated for model context. Extract from the above.]"
            logger.warning("CV truncated from %d to %d chars for LLM context", len(cv_text), max_chars)
        else:
            cv_text_for_prompt = cv_text

        if not self._ollama_llm:
            self._ollama_llm = ChatOllama(
                model=settings.ollama_model,
                temperature=0.3,
                num_ctx=8192,
            )
            logger.info("Using Ollama model: %s", settings.ollama_model)

        job_description_section = ""
        if job_description:
            logger.info(
                "match_score: job description provided → asking LLM to set match_score (0-1) by how well CV matches position requirements"
            )
            job_description_section = (
                f"\n\nJob description (position requirements):\n{job_description}\n\n"
                "Set match_score (0-1) using the formula (base 0.5, add/subtract per requirement). "
                "Set match_score_reasoning with: formula line, then each +X or -X with reason, then total. Give recommendations tailored to this position."
            )
        else:
            logger.info(
                "match_score: no job description → LLM will leave match_score null; recommendations will be general market advice"
            )
            job_description_section = (
                "\n\nNo job description provided. Leave match_score null. "
                "Give recommendations based on general market requirements (at least one)."
            )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", _CV_ANALYZER_SYSTEM),
            ("human", "Extract and return the FULL JSON: skills, experience, certificates, education, projects, analysis, match_score, match_score_reasoning (if job given), recommendations. Use [] for empty lists. CV text below.\n\nCV:\n{cv_text}\n{job_description_section}"),
        ])
        prompt_len = len(_CV_ANALYZER_SYSTEM) + len(cv_text_for_prompt) + len(job_description_section) + 200
        logger.info("match_score: sending prompt to LLM (Ollama), approx %d chars; LLM will determine match_score from CV vs job requirements", prompt_len)
        if prompt_len > 3500:
            logger.warning(
                "match_score: prompt is large (~%d chars). Ollama context is often 4096 tokens; input may be truncated and match_score can be affected.",
                prompt_len,
            )

        try:
            structured_llm = self._ollama_llm.with_structured_output(CVAnalysisOutput)
            chain = prompt_template | structured_llm
            result: CVAnalysisOutput = await chain.ainvoke({
                "cv_text": cv_text_for_prompt,
                "job_description_section": job_description_section,
            })
            if job_description and result.match_score is not None:
                logger.info(
                    "match_score: LLM returned match_score=%.2f (how well CV matches position; 0=low, 1=perfect)",
                    result.match_score,
                )
                if result.match_score_reasoning:
                    logger.info("match_score: LLM reasoning: %s", result.match_score_reasoning.strip())
            elif job_description and result.match_score is None:
                logger.warning("match_score: job description was provided but LLM returned match_score=null")
            else:
                logger.info("match_score: no job description was provided → match_score=null as expected")
            result, incomplete = _normalize_llm_result(result)
            if incomplete:
                logger.warning(
                    "LLM returned incomplete response (some fields null); filled with [] or {}. Consider using a larger context or a stronger model."
                )
            if _is_extraction_empty(result):
                logger.error("LLM returned no structured extraction (all lists and analysis empty). Returning failure.")
                return CVAnalysisResponse(
                    success=False,
                    extracted_text=cv_text,
                    error=_empty_extraction_error(
                        "Ollama",
                        settings.ollama_model,
                        len(cv_text_for_prompt),
                        len(cv_text),
                    ),
                )
            recs = result.recommendations if result.recommendations else ["Align your CV with current market expectations."]
            return CVAnalysisResponse(
                success=True,
                extracted_text=cv_text,
                analysis=result.analysis,
                skills=result.skills,
                experience=result.experience,
                certificates=result.certificates,
                education=result.education,
                projects=result.projects,
                match_score=result.match_score,
                match_score_reasoning=result.match_score_reasoning,
                recommendations=recs,
                error=None,
            )
        except Exception as e:
            import traceback
            error_msg = f"Analysis failed: {str(e)}\n{traceback.format_exc()}"
            return CVAnalysisResponse(
                success=False,
                extracted_text=cv_text,
                error=error_msg
            )
    
    async def _analyze_with_gemini(
        self, 
        cv_text: str, 
        job_description: Optional[str] = None
    ) -> CVAnalysisResponse:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError:
            raise ImportError("langchain-google-genai package is required for production")

        max_chars = getattr(settings, "max_cv_chars_for_llm", 0) or 0
        if max_chars > 0 and len(cv_text) > max_chars:
            cv_text_for_prompt = cv_text[:max_chars] + "\n\n[CV text truncated for model context. Extract from the above.]"
            logger.warning("CV truncated from %d to %d chars for LLM context", len(cv_text), max_chars)
        else:
            cv_text_for_prompt = cv_text

        if not self._gemini_llm:
            self._gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=settings.gemini_api_key,
                temperature=0.3,
            )
        
        job_description_section = ""
        if job_description:
            logger.info(
                "match_score: job description provided → asking LLM to set match_score (0-1) by how well CV matches position requirements"
            )
            job_description_section = (
                f"\n\nJob description (position requirements):\n{job_description}\n\n"
                "Set match_score (0-1) using the formula (base 0.5, add/subtract per requirement). "
                "Set match_score_reasoning with: formula line, then each +X or -X with reason, then total. Give recommendations tailored to this position."
            )
        else:
            logger.info(
                "match_score: no job description → LLM will leave match_score null; recommendations will be general market advice"
            )
            job_description_section = (
                "\n\nNo job description provided. Leave match_score null. "
                "Give recommendations based on general market requirements (at least one)."
            )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", _CV_ANALYZER_SYSTEM),
            ("human", "Extract and return the FULL JSON: skills, experience, certificates, education, projects, analysis, match_score, match_score_reasoning (if job given), recommendations. Use [] for empty lists. CV text below.\n\nCV:\n{cv_text}\n{job_description_section}"),
        ])
        prompt_len = len(_CV_ANALYZER_SYSTEM) + len(cv_text_for_prompt) + len(job_description_section) + 200
        logger.info("match_score: sending prompt to LLM (Gemini), approx %d chars; LLM will determine match_score from CV vs job requirements", prompt_len)

        try:
            structured_llm = self._gemini_llm.with_structured_output(CVAnalysisOutput)
            chain = prompt_template | structured_llm
            result: CVAnalysisOutput = await chain.ainvoke({
                "cv_text": cv_text_for_prompt,
                "job_description_section": job_description_section,
            })
            if job_description and result.match_score is not None:
                logger.info(
                    "match_score: LLM returned match_score=%.2f (how well CV matches position; 0=low, 1=perfect)",
                    result.match_score,
                )
                if result.match_score_reasoning:
                    logger.info("match_score: LLM reasoning: %s", result.match_score_reasoning.strip())
            elif job_description and result.match_score is None:
                logger.warning("match_score: job description was provided but LLM returned match_score=null")
            else:
                logger.info("match_score: no job description was provided → match_score=null as expected")
            result, incomplete = _normalize_llm_result(result)
            if incomplete:
                logger.warning(
                    "LLM returned incomplete response (some fields null); filled with [] or {}. Consider using a larger context or a stronger model."
                )
            if _is_extraction_empty(result):
                logger.error("LLM returned no structured extraction (all lists and analysis empty). Returning failure.")
                return CVAnalysisResponse(
                    success=False,
                    extracted_text=cv_text,
                    error=_empty_extraction_error(
                        "Gemini",
                        "gemini-1.5-flash",
                        len(cv_text_for_prompt),
                        len(cv_text),
                    ),
                )
            recs = result.recommendations if result.recommendations else ["Align your CV with current market expectations."]
            return CVAnalysisResponse(
                success=True,
                extracted_text=cv_text,
                analysis=result.analysis,
                skills=result.skills,
                experience=result.experience,
                certificates=result.certificates,
                education=result.education,
                projects=result.projects,
                match_score=result.match_score,
                match_score_reasoning=result.match_score_reasoning,
                recommendations=recs,
                error=None,
            )
        except Exception as e:
            import traceback
            error_msg = f"Analysis failed: {str(e)}\n{traceback.format_exc()}"
            return CVAnalysisResponse(
                success=False,
                extracted_text=cv_text,
                error=error_msg
            )
