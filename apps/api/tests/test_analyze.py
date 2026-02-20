"""Tests for POST /api/v1/analyze endpoint."""
import pytest
from unittest.mock import AsyncMock, patch

from app.models import CVAnalysisResponse


# Minimal PDF magic bytes (passes file_validator)
MINIMAL_PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 2\ntrailer\n<<>>\nstartxref\n10\n%%EOF"


@pytest.mark.asyncio
async def test_analyze_rejects_invalid_file_type(client):
    """Upload with wrong magic bytes returns 400."""
    response = await client.post(
        "/api/v1/analyze",
        files={"file": ("fake.pdf", b"not a pdf at all", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Invalid or unsupported file" in response.json()["detail"]


@pytest.mark.asyncio
async def test_analyze_success_with_mocked_parser_and_analyzer(client):
    """Valid PDF with mocked parser and analyzer returns 200 and response body."""
    mock_response = CVAnalysisResponse(
        success=True,
        extracted_text="Sample CV text",
        analysis={"summary": "Good candidate"},
        skills=["Python"],
        experience=[],
        education=[],
        match_score=0.85,
        recommendations=[],
        error=None,
    )

    with (
        patch("app.routers.cv_router.CVParser") as MockParser,
        patch("app.routers.cv_router.CVAnalyzer") as MockAnalyzer,
    ):
        mock_parser_instance = MockParser.return_value
        mock_parser_instance.parse_file = AsyncMock(return_value="Sample CV text")

        mock_analyzer_instance = MockAnalyzer.return_value
        mock_analyzer_instance.analyze_cv = AsyncMock(return_value=mock_response)

        response = await client.post(
            "/api/v1/analyze",
            files={"file": ("cv.pdf", MINIMAL_PDF, "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["skills"] == ["Python"]
    assert data.get("match_score") == 0.85
