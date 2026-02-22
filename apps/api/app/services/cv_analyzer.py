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


# System prompt: CV analyzer role and classification rules (content-based, not position)
_CV_ANALYZER_SYSTEM = """You are an AI CV analyzer. Your task is to extract structured information from the CV.

IMPORTANT: PDF text may have broken order, so do NOT rely on text position. Use only the content to classify.

Classification rules:

1. EXPERIENCE (work experience / ДОСВІД)
Include only: real jobs, internships, commercial or freelance positions, entries with job duties or work results.
If there are: task descriptions, technologies used, contribution to a project → this is EXPERIENCE.

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

Return the result strictly in the required JSON schema (experience, certificates, education, projects). Use English for field values. Also provide: skills (list), analysis (summary, strengths, weaknesses), match_score (0-1 if job description given), recommendations."""

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
    match_score: Optional[float] = Field(None, ge=0, le=1, description="0-1 if job description provided")
    recommendations: Optional[List[str]] = Field(None, description="Improvement tips")


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
        
        if not self._ollama_llm:
            self._ollama_llm = ChatOllama(
                model="llama3.2:3b",
                temperature=0.3,
            )
        
        job_description_section = ""
        if job_description:
            job_description_section = f"\n\nJob description:\n{job_description}\n\nProvide match_score (0-1)."

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", _CV_ANALYZER_SYSTEM),
            ("human", "Extract experience, certificates, education, projects, skills, analysis, match_score, recommendations.\n\nCV:\n{cv_text}\n{job_description_section}"),
        ])
        
        try:
            structured_llm = self._ollama_llm.with_structured_output(CVAnalysisOutput)
            chain = prompt_template | structured_llm
            result: CVAnalysisOutput = await chain.ainvoke({
                "cv_text": cv_text,
                "job_description_section": job_description_section,
            })
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
                recommendations=result.recommendations,
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
        
        if not self._gemini_llm:
            self._gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=settings.gemini_api_key,
                temperature=0.3,
            )
        
        job_description_section = ""
        if job_description:
            job_description_section = f"\n\nJob description:\n{job_description}\n\nProvide match_score (0-1)."

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", _CV_ANALYZER_SYSTEM),
            ("human", "Extract experience, certificates, education, projects, skills, analysis, match_score, recommendations.\n\nCV:\n{cv_text}\n{job_description_section}"),
        ])
        
        try:
            structured_llm = self._gemini_llm.with_structured_output(CVAnalysisOutput)
            chain = prompt_template | structured_llm
            result: CVAnalysisOutput = await chain.ainvoke({
                "cv_text": cv_text,
                "job_description_section": job_description_section,
            })
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
                recommendations=result.recommendations,
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
