from typing import Optional, Dict, Any
from app.config import settings
from app.models import CVAnalysisRequest, CVAnalysisResponse

class CVAnalyzer:
    
    def __init__(self):
        self._ollama = None
        self._gemini_model = None
    
    async def analyze_cv(
        self, 
        cv_text: str, 
        request: Optional[CVAnalysisRequest] = None
    ) -> CVAnalysisResponse:
        
        if settings.environment == "development":
            result =  await self._analyze_with_ollama(cv_text, request.job_description if request else None)
        else:
            result =  await self._analyze_with_gemini(cv_text, request.job_description if request else None)
        return CVAnalysisResponse(
            success=True,
            extracted_text=result['extracted_text'],
            analysis=result['analysis'],
            skills=result['skills'],
            experience=result['experience'],
            education=result['education'],
            match_score=result['match_score'],
            recommendations=result['recommendations'],
            
        )
    async def _analyze_with_ollama(
        self, 
        cv_text: str, 
        job_description: Optional[str] = None
    ) -> CVAnalysisResponse:
        try:
            import ollama 
        except ImportError:
            raise ImportError("ollama package is required for development")
        
        prompt = f"Analyze this CV:\n{cv_text}"
        if job_description:
            prompt += f"\n\nJob description:\n{job_description}"
        
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[{
                'role': 'user',
                'content': prompt
            }]
        )
        
        return CVAnalysisResponse(
            success=True,
            extracted_text=response.messages,
            analysis=None,
            skills=None,
            experience=None,
            education=None,
            match_score=None,
            recommendations=None,
            error=None
        )
    
    async def _analyze_with_gemini(
        self, 
        cv_text: str, 
        job_description: Optional[str] = None
    ) -> CVAnalysisResponse:
        try:
            import google.generativeai as genai  
        except ImportError:
            raise ImportError("google-generativeai package is required for production")
        
        if not self._gemini_model:
            genai.configure(api_key=settings.gemini_api_key)
            self._gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"Analyze this CV:\n{cv_text}"
        if job_description:
            prompt += f"\n\nJob description:\n{job_description}"
        
        response = self._gemini_model.generate_content(prompt)
        
        return CVAnalysisResponse(
            success=True,
            extracted_text=response.text,
            analysis=None,
            skills=None,
            experience=None,
            education=None,
            match_score=None,
            recommendations=None,
            error=None
        )
