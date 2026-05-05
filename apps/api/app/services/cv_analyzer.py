import logging
import traceback
from typing import Any, Dict, List, Optional

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
    "Поверни ПОВНИЙ JSON: skills, experience, certificates, education, projects, "
    "analysis, matched_competencies, missing_competencies, match_score (null), "
    "match_score_reasoning (null), recommendations. Порожні списки як [].\n\nРезюме:\n{cv_text}\n{job_description_section}"
)

_CV_ANALYZER_SYSTEM = """Ти — асистент для аналізу резюме (AI CV analyzer).

Завдання: з тексту резюме витягни структуровані дані та поверни ПОВНИЙ JSON з усіма полями щоразу.

КРИТИЧНО: усі текстові значення для користувача (підсумки, рекомендації, рядки в skills, matched_competencies, missing_competencies, поля employer/title/name тощо) — українською мовою. Оригінальні назви технологій (React, TypeScript) можна лишати латиницею.

Обовʼязкові масиви: skills, experience, certificates, education, projects — якщо немає даних, поверни []. Не використовуй null для цих масивів.

analysis — об'єкт з ключами summary, strengths, weaknesses (українською). strengths і weaknesses — рядки або короткі переліки в тексті.

recommendations — масив з щонайменше одного рядка українською.

Якщо надано опис вакансії:
- matched_competencies — пункти вимог/навичок з вакансії, які підтверджуються резюме.
- missing_competencies — вимоги вакансії, яких бракує або мало видно в резюме.
- match_score та match_score_reasoning завжди null (їх обчислює сервер окремо за семантичною моделлю).

Якщо опису вакансії немає:
- matched_competencies: [], missing_competencies: [].
- match_score та match_score_reasoning: null.
- recommendations — загальні поради для ринку праці (українською).

Класифікація:
- EXPERIENCE — робота, стажування, фриланс з обовʼязками. Не курси.
- CERTIFICATES — курси, тренінги (Meta, Coursera, Epam, AWS) без робочих обовʼязків.
- EDUCATION — виші, ступінь.
- PROJECTS — пет-проєкти, GitHub без офіційного найму.

Текст із PDF може мати «зламаний» порядок рядків — класифікуй за змістом.

Додатково: Meta / Course / Program без робочих задач → CERTIFICATE. Слова «курс», «training», «program» у контексті навчання → CERTIFICATE. Опис робочих задач → EXPERIENCE. GitHub без компанії найму → PROJECT.
"""


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
        "Усі обовʼязкові блоки (досвід, освіта, сертифікати, проєкти, навички) порожні. "
        f"Бекенд: {backend}, модель: {model_name}. {cv_detail} "
    )
    if backend == "Ollama":
        return (
            base
            + "Спробуйте: змінити OLLAMA_MODEL на потужнішу модель (наприклад llama3.2:3b); "
            "або MAX_CV_CHARS_FOR_LLM=0; або ENVIRONMENT=production (Gemini)."
        )
    return base + "Спробуйте коротше резюме або перевірте налаштування моделі."


def _experience_text_from_result(result: CVAnalysisOutput) -> str:
    parts: List[str] = []
    for e in result.experience or []:
        bits = [x for x in (e.title, e.employer, e.duration) if x]
        if bits:
            parts.append(" — ".join(bits))
    return "\n".join(parts)


def _semantic_reasoning_ua(sem: SemanticMatchResult) -> str:
    return (
        "Оцінка відповідності обчислена автоматично за косинусною подібністю "
        "векторних представлень резюме та опису вакансії (Sentence Transformers, багатомовна модель).\n\n"
        f"• Подібність за блоком навичок: {sem.skills_similarity:.1%}\n"
        f"• Подібність досвіду до вимог: {sem.experience_similarity:.1%}\n"
        f"• Загальна семантична близькість текстів: {sem.overall_similarity:.1%}\n\n"
        f"Зважена підсумкова оцінка: {sem.score:.1%}."
    )


def _build_success_response(
    cv_text: str,
    result: CVAnalysisOutput,
    job_description: Optional[str],
) -> CVAnalysisResponse:
    recs = result.recommendations or ["Уточніть резюме та повторіть аналіз."]
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
            match_reason = _semantic_reasoning_ua(sem)
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
            f"\n\nОпис вакансії (вимоги):\n{job_description.strip()}\n\n"
            "Порівняй резюме з вимогами. Заповни matched_competencies та missing_competencies українською. "
            "match_score і match_score_reasoning залиш null. Рекомендації — конкретно під цю вакансію, українською."
        )
    return (
        "\n\nОпис вакансії не надано. matched_competencies і missing_competencies — порожні списки []. "
        "match_score і match_score_reasoning — null. Дай загальні рекомендації для ринку праці українською."
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

    async def _run_structured_chain(
        self,
        cv_text: str,
        cv_text_for_prompt: str,
        job_description: Optional[str],
        structured_llm: Any,
        backend: str,
        model_name: str,
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
            error_msg = f"Помилка аналізу: {e!s}\n{traceback.format_exc()}"
            return CVAnalysisResponse(success=False, extracted_text=cv_text, error=error_msg)

    def _cv_text_for_prompt(self, cv_text: str) -> str:
        max_chars = settings.max_cv_chars_for_llm
        if max_chars > 0 and len(cv_text) > max_chars:
            logger.warning("CV truncated from %d to %d chars for LLM context", len(cv_text), max_chars)
            return (
                cv_text[:max_chars]
                + "\n\n[Текст резюме обрізано для контексту моделі. Витягни дані з наведеного.]"
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
                temperature=0.3,
                num_ctx=8192,
            )
            logger.info("Using Ollama model: %s", settings.ollama_model)

        structured_llm = self._ollama_llm.with_structured_output(CVAnalysisOutput)
        return await self._run_structured_chain(
            cv_text,
            cv_text_for_prompt,
            job_description,
            structured_llm,
            "Ollama",
            settings.ollama_model,
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
