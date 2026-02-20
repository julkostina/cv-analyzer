from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models import CVAnalysisResponse
from app.services.cv_parser import CVParser
from app.models import CVAnalysisRequest
from app.services.cv_analyzer import CVAnalyzer
from app.utils.file_validator import validate_file
from app.config import settings
import aiofiles
import os
from pathlib import Path

router = APIRouter()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/analyze", response_model=CVAnalysisResponse)
async def analyze_cv(
    file: UploadFile = File(...),
    job_description: str = None,
):
    file_path = None
    try:
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

        if not cv_text:
            raise HTTPException(status_code=400, detail="Failed to parse CV")

        analyzer = CVAnalyzer()
        request = (
            CVAnalysisRequest(job_description=job_description)
            if job_description
            else None
        )
        result = await analyzer.analyze_cv(cv_text, request)

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        raise HTTPException(
            status_code=500,
            detail=str(e) if settings.environment != "development" else traceback.format_exc(),
        )
    finally:
        if file_path is not None and file_path.exists():
            try:
                os.remove(file_path)
            except OSError:
                pass