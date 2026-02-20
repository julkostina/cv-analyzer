"""
File validation: allowlist extension/MIME, magic-byte detection, size limit, safe filenames.
"""
from uuid import uuid4

# Allowlist: only these types are accepted
ALLOWED_EXTENSIONS = ("pdf", "docx")
ALLOWED_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

# Magic bytes
PDF_SIGNATURE = b"%PDF-"
DOCX_ZIP_SIGNATURE = b"PK\x03\x04"
DOCX_REQUIRED_MEMBER = b"word/document.xml"


def detect_type_from_magic_bytes(content: bytes) -> str | None:
    """
    Detect file type from content (magic bytes).
    Returns 'pdf', 'docx', or None if not recognized.
    """
    if len(content) < 5:
        return None
    if content.startswith(PDF_SIGNATURE):
        return "pdf"
    if content.startswith(DOCX_ZIP_SIGNATURE) and DOCX_REQUIRED_MEMBER in content:
        return "docx"
    return None


def get_extension_from_filename(filename: str | None) -> str | None:
    """
    Extract extension from filename only if it is in the allowlist.
    Returns None if missing or not allowed.
    """
    if not filename or "." not in filename:
        return None
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext if ext in ALLOWED_EXTENSIONS else None


def validate_file(
    content: bytes,
    filename: str | None,
    content_type: str | None,
    max_size_bytes: int,
) -> tuple[str, str]:
    """
    Validate upload: size, magic bytes, and extension/MIME consistency.
    Returns (file_type, safe_filename) for the validated file.
    Raises ValueError with a clear message for invalid files.
    """
    if len(content) > max_size_bytes:
        raise ValueError(
            f"File too large. Maximum size: {max_size_bytes // (1024 * 1024)} MB"
        )

    detected = detect_type_from_magic_bytes(content)
    if detected is None:
        raise ValueError(
            "Invalid or unsupported file: content does not match PDF or DOCX."
        )

    ext = get_extension_from_filename(filename)
    if ext is not None and ext != detected:
        raise ValueError(
            f"File extension (.{ext}) does not match file content (detected: {detected})."
        )

    if content_type and content_type in ALLOWED_MIME_TYPES:
        mime_type = ALLOWED_MIME_TYPES[content_type]
        if mime_type != detected:
            raise ValueError(
                f"Content-Type ({content_type}) does not match file content (detected: {detected})."
            )

    safe_filename = f"{uuid4().hex}.{detected}"
    return (detected, safe_filename)
