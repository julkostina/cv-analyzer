import logging
import os
import re
import traceback
from pathlib import Path
from typing import Annotated, Optional

import aiofiles
import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import settings
from app.models import CVAnalysisRequest, CVAnalysisResponse
from app.services.analysis_pdf import render_analysis_pdf
from app.services.cv_analyzer import CVAnalyzer
from app.services.cv_parser import CVParser
from app.utils.file_validator import validate_file
from app.utils.text_preprocess import normalize_text_for_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

JOB_URL_CONTENT_LIMIT = 50_000


async def _fetch_job_description_from_url(url: str) -> str:
    """Fetch URL and return body as plain text. Strips HTML tags."""
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        text = resp.text
    if len(text) > JOB_URL_CONTENT_LIMIT:
        text = text[:JOB_URL_CONTENT_LIMIT] + "\n[... truncated]"
    # Strip HTML tags for a rough plain-text version
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return normalize_text_for_pipeline(text)


@router.post("/analyze", response_model=None)
async def analyze_cv(
    file: UploadFile = File(..., description="CV file (PDF or DOCX)"),
    job_description: Annotated[
        Optional[str], Form(description="Position requirements as plain text (optional)")
    ] = None,
    job_description_url: Annotated[
        Optional[str],
        Form(description="URL of job posting to fetch requirements from (optional). Used with or instead of job_description."),
    ] = None,
    return_pdf: Annotated[
        bool,
        Form(description="If true and analysis succeeds, response is application/pdf."),
    ] = False,
):
    file_path = None
    try:
        job_text = (job_description or "").strip()
        if job_description_url and job_description_url.strip():
            try:
                url_content = await _fetch_job_description_from_url(job_description_url.strip())
                job_text = f"{job_text}\n\n{url_content}".strip() if job_text else url_content
            except Exception as e:
                logger.warning("Failed to fetch job_description_url %s: %s", job_description_url, e)
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not fetch job description from URL: {e!s}",
                ) from e

        job_text = normalize_text_for_pipeline(job_text) if job_text else ""

        content = await file.read()
        file_type, safe_filename = validate_file(
            content,
            file.filename,
            file.content_type,
            settings.max_upload_size_bytes,
        )
        file_path = UPLOAD_DIR / safe_filename
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        parser = CVParser()
        cv_text = await parser.parse_file(str(file_path), file_type)
        cv_text = normalize_text_for_pipeline(cv_text or "")

        if not cv_text:
            raise HTTPException(status_code=400, detail="Failed to parse CV")

        analyzer = CVAnalyzer()
        request = (
            CVAnalysisRequest(job_description=job_text or None)
            if job_text
            else None
        )
        result = await analyzer.analyze_cv(cv_text, request)

        if return_pdf:
            if not result.success:
                raise HTTPException(
                    status_code=400,
                    detail=result.error or "Analysis failed; PDF was not generated.",
                )
            pdf_bytes = render_analysis_pdf(result)
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": 'attachment; filename="cv-analysis-report.pdf"',
                },
            )

        return result

    except ValueError as e:
        logger.warning("CV analyze validation error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CV analyze failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=traceback.format_exc() if settings.environment == "development" else str(e),
        )
    finally:
        if file_path is not None and file_path.exists():
            try:
                os.remove(file_path)
            except OSError:
                pass