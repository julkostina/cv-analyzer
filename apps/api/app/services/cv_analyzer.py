from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from app.config import settings
from app.models import CVAnalysisRequest, CVAnalysisResponse


# Pydantic model for structured LLM output
class CVAnalysisOutput(BaseModel):
    """Structured output schema for CV analysis"""
    analysis: Optional[Dict[str, Any]] = Field(
        None, 
        description="Analysis summary with strengths, weaknesses, and overall assessment"
    )
    skills: Optional[List[str]] = Field(
        None, 
        description="List of skills extracted from the CV"
    )
    experience: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="List of work experience entries with title, company, duration"
    )
    education: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="List of education entries with degree, institution, year"
    )
    match_score: Optional[float] = Field(
        None, 
        ge=0, 
        le=1, 
        description="Match score between CV and job description (0-1)"
    )
    recommendations: Optional[List[str]] = Field(
        None, 
        description="List of recommendations for improving the CV"
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
        
        if not self._ollama_llm:
            self._ollama_llm = ChatOllama(
                model="llama3.2:3b",
                temperature=0.3,
            )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert CV analyzer. Analyze the provided CV and extract structured information.
Respond in English only, regardless of the CV language.
Return your response as valid JSON matching the required schema."""),
            ("human", """Analyze this CV and extract the following information:
- Skills and competencies
- Work experience (title, company, duration)
- Education (degree, institution, year)
- Overall analysis (summary, strengths, weaknesses)
- Match score (0-1) if job description is provided
- Recommendations for improvement

CV Content:
{cv_text}
{job_description_section}""")
        ])
        
        job_description_section = ""
        if job_description:
            job_description_section = f"\n\nJob Description:\n{job_description}\n\nPlease provide a match score (0-1) based on how well the CV matches this job description."
        
        try:
            # Use with_structured_output for better compatibility with Ollama
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
                education=result.education,
                match_score=result.match_score,
                recommendations=result.recommendations,
                error=None
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
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert CV analyzer. Analyze the provided CV and extract structured information.
Respond in English only, regardless of the CV language.
Return your response as valid JSON matching the required schema."""),
            ("human", """Analyze this CV and extract the following information:
- Skills and competencies
- Work experience (title, company, duration)
- Education (degree, institution, year)
- Overall analysis (summary, strengths, weaknesses)
- Match score (0-1) if job description is provided
- Recommendations for improvement

CV Content:
{cv_text}
{job_description_section}""")
        ])
        
        job_description_section = ""
        if job_description:
            job_description_section = f"\n\nJob Description:\n{job_description}\n\nPlease provide a match score (0-1) based on how well the CV matches this job description."
        
        try:
            # Use with_structured_output for better compatibility
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
                education=result.education,
                match_score=result.match_score,
                recommendations=result.recommendations,
                error=None
            )
        except Exception as e:
            import traceback
            error_msg = f"Analysis failed: {str(e)}\n{traceback.format_exc()}"
            return CVAnalysisResponse(
                success=False,
                extracted_text=cv_text,
                error=error_msg
            )
