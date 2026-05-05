"""PDF-звіт з результату аналізу резюме."""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

from app.config import settings
from app.models import CVAnalysisResponse

logger = logging.getLogger(__name__)

_DEJAVU_CDN_TTF = (
    "https://cdn.jsdelivr.net/npm/dejavu-fonts-ttf@2.37.3/ttf/DejaVuSans.ttf"
)


def _font_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".cache"
    d = root / "cv-analyzer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_dejavu_font_path() -> Path:
    explicit = (settings.pdf_font_path or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        raise RuntimeError(f"PDF_FONT_PATH не знайдено або не файл: {p}")

    cached = _font_cache_dir() / "DejaVuSans.ttf"
    if cached.is_file() and cached.stat().st_size > 10_000:
        return cached

    logger.info("Downloading DejaVu Sans for PDF (one-time cache): %s", cached)
    try:
        urllib.request.urlretrieve(_DEJAVU_CDN_TTF, cached)  # noqa: S310
    except Exception as e:
        raise RuntimeError(
            "Не вдалося завантажити шрифт DejaVu для PDF. "
            "Перевірте мережу або встановіть PDF_FONT_PATH на локальний .ttf."
        ) from e
    return cached


def render_analysis_pdf(resp: CVAnalysisResponse) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError as e:
        raise RuntimeError("Для PDF потрібен пакет fpdf2: pip install fpdf2") from e

    font_path = _resolve_dejavu_font_path()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("DejaVu", "", str(font_path))
    pdf.set_font("DejaVu", "", 11)

    def _line(text: str) -> None:
        for paragraph in (text or "").split("\n"):
            p = (paragraph or " ").strip() or " "
            pdf.multi_cell(0, 6, p)
        pdf.ln(2)

    pdf.set_font("DejaVu", "", 14)
    _line("Звіт аналізу резюме")
    pdf.set_font("DejaVu", "", 11)

    if not resp.success:
        _line("Помилка обробки")
        _line(resp.error or "Невідома помилка")
        return _pdf_bytes(pdf)

    if resp.match_score is not None:
        pct = round(float(resp.match_score) * 100, 1)
        _line(f"Оцінка відповідності вакансії: {pct}%")

    if resp.semantic_breakdown:
        sb = resp.semantic_breakdown
        _line("Семантичний розклад (подібність 0–1):")
        _line(
            f"  Навички: {sb.get('skills_similarity', 0):.3f}\n"
            f"  Досвід: {sb.get('experience_similarity', 0):.3f}\n"
            f"  Загальна: {sb.get('overall_similarity', 0):.3f}"
        )

    if resp.match_score_reasoning:
        _line("Обґрунтування оцінки:")
        _line(resp.match_score_reasoning)

    if resp.matched_competencies:
        _line("Відповідні компетенції:")
        _line("\n".join(f"• {x}" for x in resp.matched_competencies))

    if resp.missing_competencies:
        _line("Відсутні або слабкі компетенції:")
        _line("\n".join(f"• {x}" for x in resp.missing_competencies))

    if resp.analysis:
        summ = resp.analysis.get("summary")
        if summ:
            _line("Підсумок:")
            _line(str(summ))
        strengths = resp.analysis.get("strengths")
        if strengths:
            _line("Сильні сторони:")
            if isinstance(strengths, list):
                _line("\n".join(f"• {s}" for s in strengths))
            else:
                _line(str(strengths))
        weaknesses = resp.analysis.get("weaknesses")
        if weaknesses:
            _line("Слабкі сторони:")
            if isinstance(weaknesses, list):
                _line("\n".join(f"• {w}" for w in weaknesses))
            else:
                _line(str(weaknesses))

    if resp.recommendations:
        _line("Рекомендації:")
        _line("\n".join(f"• {r}" for r in resp.recommendations))

    if resp.skills:
        _line("Витягнуті навички:")
        _line(", ".join(resp.skills))

    return _pdf_bytes(pdf)


def _pdf_bytes(pdf) -> bytes:
    raw = pdf.output(dest="S")
    if isinstance(raw, str):
        return raw.encode("latin-1")
    return bytes(raw)
