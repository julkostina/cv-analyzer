import json
import logging
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

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
from app.services.semantic_matcher import (
    SEMANTIC_METRIC_GUIDES,
    SemanticMatchResult,
    compute_semantic_match,
    normalized_semantic_weights,
)
from app.services.match_explainer import explain_match_score
from app.utils.text_preprocess import normalize_text_for_pipeline

logger = logging.getLogger(__name__)

_HUMAN_PROMPT = (
    "Поверни ПОВНИЙ JSON з ключами (англійською, як у схемі): skills, experience, certificates, education, projects, "
    "analysis, matched_competencies, missing_competencies, match_score (null), "
    "match_score_reasoning (null), recommendations. Для порожніх списків використовуй [].\n\nРезюме:\n{cv_text}\n{job_description_section}"
)

_CV_ANALYZER_SYSTEM = """Ти — ШІ-аналітик резюме.

Завдання: витягни структуровані дані з тексту резюме й щоразу поверни ПОВНИЙ JSON-об’єкт.

КРИТИЧНО: увесь текст для користувача (підсумки, рекомендації, рядки навичок, matched_competencies, missing_competencies, поля роботодавця/посади/назв тощо) — українською мовою. Назви технологій (React, TypeScript) залишай латиницею.

Обов’язкові масиви: skills, experience, certificates, education, projects — якщо порожньо, використовуй [], не null.

analysis — об’єкт із ключами summary, strengths, weaknesses (українською). strengths і weaknesses можуть бути рядком або коротким списком.

recommendations — щонайменше один рядок українською. Кожен пункт — КОНКРЕТНА порада щодо покращення резюме/кандидатури (редагування CV, навички, які варто підкреслити, проєкти, прогалини). Пиши наказовим або на «ви» («Додайте…», «Сформулюйте…», «Розгляньте…»). Не рекламуй вакансії, не описуй «переваги ролі», не використовуй штампи на кшталт «ви матимете можливість», «ви працюватимете над…», «як [посада] ви будете…».

Якщо є опис вакансії:
- matched_competencies — вимоги/навички з оголошення, які чітко підтверджує резюме.
- missing_competencies — вимоги, які у резюме слабкі або відсутні.
- match_score і match_score_reasoning завжди null (сервер заповнить їх семантичною оцінкою).

Якщо опису вакансії немає:
- matched_competencies: [], missing_competencies: [].
- match_score і match_score_reasoning: null.
- recommendations — загальні ринкові поради щодо покращення резюме (ті самі правила: лише дії для кандидата).

Класифікація:
- EXPERIENCE — роботи, стажування, фриланс з обов’язками. Не курси.
- CERTIFICATES — курси, тренінги (Meta, Coursera, Epam, AWS) без робочих обов’язків.
- EDUCATION — університети, ступені.
- PROJECTS — пет-проєкти, GitHub без формального найму.

Порядок тексту в PDF може бути зламаним — класифікуй за змістом.

Правила: Meta / Course / Program без робочих задач → CERTIFICATE. Слова «курс», «навчання», «програма» в навчальному контексті → CERTIFICATE. Описані робочі задачі → EXPERIENCE. Лише GitHub без компанії-наймача → PROJECT.
"""

_OLLAMA_FALLBACK_SYSTEM = (
    "Поверни рівно ОДИН валідний JSON-об’єкт і нічого більше: без markdown, без огорож коду, без тексту до чи після. "
    "Ключі (англійською): skills, experience, certificates, education, projects, analysis, "
    "matched_competencies, missing_competencies, match_score, match_score_reasoning, recommendations. "
    "Усі рядки для користувача українською. recommendations — щонайменше один пункт; кожен — конкретна порада щодо резюме/кандидатури, "
    "не маркетинг вакансії."
)

_OLLAMA_FALLBACK_HUMAN = (
    "Заповни JSON за тими самими правилами класифікації (досвід / сертифікати / освіта / проєкти). "
    "Порожні розділи як []. match_score і match_score_reasoning мають бути null.\n\n"
    "Резюме:\n{cv_text}\n{job_description_section}"
)

_OLLAMA_FALLBACK_RULES = (
    "Класифікація: EXPERIENCE — роботи, стаж, фриланс; CERTIFICATES — курси без робочих обов’язків; "
    "EDUCATION — школа/університет; PROJECTS — бічні проєкти. Meta/Course/Program без робочих задач → CERTIFICATE."
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
    recommendations: List[str] = Field(
        ...,
        min_length=1,
        description="Concrete improvements to the CV or candidacy; never job-pitch or 'what you will gain' role copy.",
    )
    matched_competencies: Optional[List[str]] = Field(None)
    missing_competencies: Optional[List[str]] = Field(None)


class MatchScoreFallbackOutput(BaseModel):
    """LLM-only estimate when embedding-based semantic scoring is unavailable."""

    match_score: float = Field(..., ge=0, le=1, description="Оцінка відповідності 0..1")
    match_score_reasoning: str = Field(
        ...,
        min_length=10,
        description="Обґрунтування українською; згадай, що це резервна оцінка без деталізації за векторами.",
    )


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
        data["recommendations"] = ["Перегляньте резюме та спробуйте аналіз ще раз."]
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
        parts.append("Малі моделі часто не справляються зі складними JSON-схемами.")
        parts.append("Спробуйте OLLAMA_MODEL=llama3:8b або llama3.1:8b (ollama pull …).")
    elif "llama3" in m and "8b" in m:
        parts.append("Навіть llama3:8b іноді повертає порожній структурований вивід (Ollama + LangChain JSON schema).")
        parts.append(
            "Спробуйте: mistral / qwen2.5 / llama3.1:8b; оновіть Ollama та перезавантажте модель; або ENVIRONMENT=production з GEMINI_API_KEY."
        )
    else:
        parts.append("Спробуйте іншу модель (mistral, qwen2.5, llama3.1:8b) або Gemini у production.")
    parts.append(
        f"За потреби збільште OLLAMA_NUM_CTX (зараз {settings.ollama_num_ctx}). Тримайте MAX_CV_CHARS_FOR_LLM=0, щоб не обрізати резюме."
    )
    return " ".join(parts)


def _empty_extraction_error(
    backend: str,
    model_name: str,
    cv_chars_sent: int,
    cv_chars_total: int,
) -> str:
    if cv_chars_sent < cv_chars_total:
        cv_detail = f"До моделі надіслано {cv_chars_sent} символів резюме (обрізано з {cv_chars_total})."
    else:
        cv_detail = f"Довжина резюме: {cv_chars_total} символів."
    base = (
        "Аналіз не вдався: модель не повернула структуровані дані. "
        "Усі обов’язкові розділи (досвід, освіта, сертифікати, проєкти, навички) порожні. "
        f"Бекенд: {backend}, модель: {model_name}. {cv_detail} "
    )
    if backend == "Ollama":
        return base + _ollama_empty_extraction_hint(model_name)
    return base + "Спробуйте коротше резюме або перевірте налаштування моделі."


def _experience_text_from_result(result: CVAnalysisOutput) -> str:
    parts: List[str] = []
    for e in result.experience or []:
        bits = [x for x in (e.title, e.employer, e.duration) if x]
        if bits:
            parts.append(" — ".join(bits))
    return "\n".join(parts)


def _semantic_reasoning_text(sem: SemanticMatchResult) -> str:
    return (
        "Бал відповідності з косинусної схожості векторних подань резюме та вакансії "
        "(багатомовна модель Sentence Transformers).\n\n"
        f"• Схожість блоку навичок: {sem.skills_similarity:.1%}\n"
        f"• Схожість досвіду до вимог: {sem.experience_similarity:.1%}\n"
        f"• Схожість повного резюме до повного тексту вакансії: {sem.overall_similarity:.1%}\n\n"
        f"Зважений загальний бал: {sem.score:.1%}."
    )


_SEMANTIC_NARRATIVE_SYSTEM = """Ти — кар’єрний коуч, який допомагає кандидату зрозуміти, як його резюме лягає на текст вакансії.

Правила:
- Числові бали, які ти отримуєш, ОСТАТОЧНІ (вже пораховані з ембеддингів). Не змінюй їх, не перераховуй і не вигадуй інші числа.
- Поясни простою українською, що ці бали ймовірно означають саме для ЦІЄЇ людини, спираючись на наведені уривки.
- Будь стислим, підтримливим і конкретним до уривків (список навичок, фрагмент резюме, фрагмент вакансії). Якщо список навичок порожній або крихітний, скажи, що це може занизити бал навичок, не означаючи відсутності навичок узагалі.
- Звертайся на «ви». Лише українська мова. Без markdown-заголовків; короткі абзаци або марковані рядки — гаразд.
- Не більше 280 слів."""

_FALLBACK_MATCH_SCORE_SYSTEM = """Ти оцінюєш відповідність резюме вакансії, коли основний серверний конвейєр векторних подань (ембеддинги) тимчасово недоступний або дав збій.

Поверни:
- match_score — дійсне число від 0 до 1 (чим ближче до 1, тим краща відповідність за твоїм судженням).
- match_score_reasoning — 2–6 речень українською; обов’язково згадай, що це РЕЗЕРВНА оцінка без розбиття на підпоказники, бо вбудований семантичний підрахунок не спрацював.

Оцінюй лише за тим, що видно в наданих уривках резюме та вакансії; не вигадуй фактів, яких немає в тексті. Будь обережним і чесним щодо невизначеності."""


def _llm_message_text(msg: Any) -> str:
    raw = getattr(msg, "content", msg)
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        pieces: List[str] = []
        for block in raw:
            if isinstance(block, str):
                pieces.append(block)
            elif isinstance(block, dict):
                pieces.append(str(block.get("text", block)))
            else:
                pieces.append(str(block))
        return "".join(pieces)
    return str(raw or "")


def _build_success_response(
    cv_text: str,
    result: CVAnalysisOutput,
    job_description: Optional[str],
) -> Tuple[CVAnalysisResponse, bool]:
    recs = result.recommendations or ["Перегляньте резюме та спробуйте ще раз."]
    matched = list(result.matched_competencies or [])
    missing = list(result.missing_competencies or [])

    match_score: Optional[float] = None
    match_reason: Optional[str] = None
    semantic_breakdown: Optional[Dict[str, float]] = None
    semantic_weights: Optional[Dict[str, float]] = None
    semantic_metric_guides: Optional[Dict[str, str]] = None
    semantic_pipeline_failed = False
    match_explainability = None

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
            ws, we, wo = normalized_semantic_weights()
            semantic_weights = {"skills": ws, "experience": we, "overall": wo}
            semantic_metric_guides = dict(SEMANTIC_METRIC_GUIDES)
            match_reason = _semantic_reasoning_text(sem)
            try:
                match_explainability = explain_match_score(
                    sem=sem,
                    skills=result.skills or [],
                    experience=result.experience,
                    cv_experience_text=cv_exp or cv_text[:8000],
                    cv_full_text=cv_text,
                    job_requirements_text=job_stripped,
                    job_full_text=job_stripped,
                )
            except Exception:
                logger.warning("Match explainability failed; continuing without SHAP/LIME", exc_info=True)
            logger.info(
                "Semantic match score=%.3f (skills=%.3f exp=%.3f overall=%.3f)",
                sem.score,
                sem.skills_similarity,
                sem.experience_similarity,
                sem.overall_similarity,
            )
        except Exception:
            logger.exception("Semantic matching failed; will try LLM fallback for match_score")
            semantic_pipeline_failed = True
            match_score = result.match_score
            match_reason = result.match_score_reasoning

    resp = CVAnalysisResponse(
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
        semantic_weights=semantic_weights,
        semantic_metric_guides=semantic_metric_guides,
        semantic_score_narrative=None,
        match_explainability=match_explainability,
        error=None,
    )
    return resp, semantic_pipeline_failed


def _job_section(job_description: Optional[str]) -> str:
    if job_description and job_description.strip():
        return (
            f"\n\nОпис вакансії (вимоги):\n{job_description.strip()}\n\n"
            "Порівняй резюме з вимогами. Заповни matched_competencies та missing_competencies українською. "
            "Залиш match_score і match_score_reasoning null. "
            "Рекомендації українською — що покращити в резюме чи профілі під цю роль "
            "(лише конкретні кроки; без реклами вакансії та «переваг ролі»)."
        )
    return (
        "\n\nОпис вакансії не надано. matched_competencies і missing_competencies мають бути []. "
        "match_score і match_score_reasoning — null. "
        "Дай загальні ринкові рекомендації українською щодо покращення резюме: лише дії для кандидата, без реклами вакансій."
    )


class CVAnalyzer:
    def __init__(self) -> None:
        self._ollama_llm = None
        self._gemini_llm = None

    async def _enrich_semantic_score_narrative(
        self,
        resp: CVAnalysisResponse,
        raw_llm: Any,
        job_description: Optional[str],
        cv_text_for_prompt: str,
    ) -> CVAnalysisResponse:
        if not settings.use_llm_semantic_narrative:
            return resp
        if not resp.semantic_breakdown or resp.match_score is None:
            return resp
        jd = (job_description or "").strip()
        if not jd:
            return resp
        sb = resp.semantic_breakdown
        sw = resp.semantic_weights or {}
        ws, we, wo = float(sw.get("skills", 0.5)), float(sw.get("experience", 0.3)), float(sw.get("overall", 0.2))
        skills_line = ", ".join(s for s in (resp.skills or [])[:120] if s and str(s).strip()) or "(не витягнуто)"
        cv_snip = (cv_text_for_prompt or "").strip()[:2800]
        job_snip = jd[:2800]
        human = (
            "Нижче подібності кожна в діапазоні від 0 до 1 (більше — ближче за змістом у просторі ембеддингів).\n\n"
            f"Загальний match_score (зважена суміш): {float(resp.match_score):.4f}\n"
            f"Ваги: навички {ws:.4f}, досвід {we:.4f}, усе резюме vs вакансія {wo:.4f}\n\n"
            f"skills_similarity (навички vs вакансія): {float(sb.get('skills_similarity', 0)):.4f}\n"
            f"experience_similarity (досвід vs вакансія): {float(sb.get('experience_similarity', 0)):.4f}\n"
            f"overall_similarity (повне резюме vs повний текст вакансії): {float(sb.get('overall_similarity', 0)):.4f}\n\n"
            f"Витягнуті навички (через кому): {skills_line}\n\n"
            f"Уривок вакансії:\n{job_snip}\n\n"
            f"Уривок резюме:\n{cv_snip}\n\n"
            "Напиши пояснення для кандидата згідно з інструкціями."
        )
        try:
            out = await raw_llm.ainvoke(
                [SystemMessage(content=_SEMANTIC_NARRATIVE_SYSTEM), HumanMessage(content=human)]
            )
            text = _llm_message_text(out).strip()
            if not text:
                return resp
            if len(text) > 6000:
                text = text[:6000].rsplit(" ", 1)[0] + "…"
            return resp.model_copy(update={"semantic_score_narrative": text})
        except Exception:
            logger.warning("LLM semantic score narrative failed; response left without narrative", exc_info=True)
            return resp

    async def _llm_fallback_match_score(
        self,
        resp: CVAnalysisResponse,
        raw_llm: Any,
        cv_text_for_prompt: str,
        job_description: Optional[str],
    ) -> CVAnalysisResponse:
        jd = (job_description or "").strip()
        if not jd or resp.match_score is not None:
            return resp
        cv_snip = (cv_text_for_prompt or "").strip()[:6000]
        job_snip = jd[:6000]
        human = (
            "Оціни відповідність за цими уривками.\n\n"
            f"Резюме:\n{cv_snip}\n\n"
            f"Вакансія:\n{job_snip}"
        )
        structured = raw_llm.with_structured_output(MatchScoreFallbackOutput)
        try:
            out = await structured.ainvoke(
                [
                    SystemMessage(content=_FALLBACK_MATCH_SCORE_SYSTEM),
                    HumanMessage(content=human),
                ]
            )
            reason = (out.match_score_reasoning or "").strip()
            if not reason:
                return resp
            return resp.model_copy(
                update={
                    "match_score": float(out.match_score),
                    "match_score_reasoning": reason,
                }
            )
        except Exception:
            logger.warning("LLM fallback match score failed; leaving match_score unset", exc_info=True)
            return resp

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
        raw_llm: Any,
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
                            built, sem_failed = _build_success_response(cv_text, fb, job_description)
                            if sem_failed:
                                built = await self._llm_fallback_match_score(
                                    built, raw_llm, cv_text_for_prompt, job_description
                                )
                            return await self._enrich_semantic_score_narrative(
                                built, raw_llm, job_description, cv_text_for_prompt
                            )
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
            built, sem_failed = _build_success_response(cv_text, result, job_description)
            if sem_failed:
                built = await self._llm_fallback_match_score(
                    built, raw_llm, cv_text_for_prompt, job_description
                )
            return await self._enrich_semantic_score_narrative(
                built, raw_llm, job_description, cv_text_for_prompt
            )
        except Exception as e:
            error_msg = f"Помилка аналізу: {e!s}\n{traceback.format_exc()}"
            return CVAnalysisResponse(success=False, extracted_text=cv_text, error=error_msg)

    def _cv_text_for_prompt(self, cv_text: str) -> str:
        max_chars = settings.max_cv_chars_for_llm
        if max_chars > 0 and len(cv_text) > max_chars:
            logger.warning("CV truncated from %d to %d chars for LLM context", len(cv_text), max_chars)
            return (
                cv_text[:max_chars]
                + "\n\n[Текст резюме обрізано для контексту моделі. Витягни дані з наведеного вище.]"
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
            self._ollama_llm,
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
            self._gemini_llm,
            "Gemini",
            "gemini-2.5-flash",
        )
