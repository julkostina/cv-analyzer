from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

class CVUpload(BaseModel):
    filename: str = Field(...,min_length=1, pattern=r"^[\w]+\.(pdf|docx?)$")
    file_type: Literal["application/pdf", "application/msword"]
    file_size: int = Field(..., gt=0)
    file_content: bytes = Field(...)

class CVAnalysisRequest(BaseModel):
    job_description: Optional[str] = None
    analysis_type: str = Field(default="full")
    extract_keywords: bool = True

class CVAnalysisResponse(BaseModel):
    success: bool
    extracted_text: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    skills: Optional[List[str]] = None
    experience: Optional[List[Dict[str, Any]]] = None
    education: Optional[List[Dict[str, Any]]] = None
    match_score: Optional[float] = Field(None, ge=0, le=1)
    recommendations: Optional[List[str]] = None
    error: Optional[str] = None
