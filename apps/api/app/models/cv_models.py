from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal


class CVUpload(BaseModel):
    filename: str = Field(..., min_length=1, pattern=r"^[\w]+\.(pdf|docx?)$")
    file_type: Literal["application/pdf", "application/msword"]
    file_size: int = Field(..., gt=0)
    file_content: bytes = Field(...)


class CVAnalysisRequest(BaseModel):
    job_description: Optional[str] = None
    analysis_type: str = Field(default="full")
    extract_keywords: bool = True


# Explicit schemas for OpenAPI docs and validation
class ExperienceItem(BaseModel):
    """Single work experience entry."""
    employer: Optional[str] = Field(None, description="Company or organization name")
    title: Optional[str] = Field(None, description="Job title or role")
    duration: Optional[str] = Field(None, description="Employment period (e.g. 2022-2024)")


class CertificateItem(BaseModel):
    """Single certificate or course entry."""
    name: Optional[str] = Field(None, description="Certificate or course name")
    institution: Optional[str] = Field(None, description="Issuer (e.g. Meta, Epam, Coursera)")
    year: Optional[str] = Field(None, description="Completion date or year")


class EducationItem(BaseModel):
    """Single education entry."""
    degree: Optional[str] = Field(None, description="Degree or qualification")
    institution: Optional[str] = Field(None, description="School or university")
    year: Optional[str] = Field(None, description="Graduation year or period")


class ProjectItem(BaseModel):
    """Single project entry (pet / GitHub projects)."""
    name: Optional[str] = Field(None, description="Project name")
    description: Optional[str] = Field(None, description="Short description")
    link: Optional[str] = Field(None, description="URL (e.g. GitHub)")


class CVAnalysisResponse(BaseModel):
    success: bool
    extracted_text: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    skills: Optional[List[str]] = None
    experience: Optional[List[ExperienceItem]] = None
    certificates: Optional[List[CertificateItem]] = None
    education: Optional[List[EducationItem]] = None
    projects: Optional[List[ProjectItem]] = None
    match_score: Optional[float] = Field(None, ge=0, le=1)
    recommendations: Optional[List[str]] = None
    error: Optional[str] = None
