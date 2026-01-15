from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models import CVAnalysisResponse
from app.services.cv_parser import CVParser
from app.models import CVAnalysisRequest
from app.services.cv_analyzer import CVAnalyzer
import aiofiles
import os
from pathlib import Path

router = APIRouter()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/analyze", response_model=CVAnalysisResponse)
async def analyze_cv(
    file: UploadFile = File(...),
    job_description: str = None
):
    try:
        file_path = UPLOAD_DIR / file.filename
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        parser = CVParser()
        file_type = file.content_type.split("/")[1].lower() if file.content_type else file.filename.split(".")[-1].lower()
        cv_text = await parser.parse_file(str(file_path), file_type)

        if not cv_text:
            raise HTTPException(status_code=400, detail="Failed to parse CV")
        
        analyzer = CVAnalyzer()
        request = CVAnalysisRequest(job_description=job_description) if job_description else None
        result = await analyzer.analyze_cv(cv_text, request)
        
        os.remove(file_path)
        return result

    except Exception as e:
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))