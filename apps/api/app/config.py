from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
_API_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _API_DIR / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), case_sensitive=False, env_file_encoding="utf-8")
    
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    openai_api_key: str = ""
    gemini_api_key: str = ""
    environment: str = "development"

    # Ollama model for development (ollama pull …). 8B+ models follow JSON schema more reliably than 3B.
    ollama_model: str = "llama3:8b"
    # Context window (tokens). Long Ukrainian prompt + CV + job needs headroom for structured output.
    ollama_num_ctx: int = 24576

    # Max CV characters sent to the LLM (rest is truncated). 0 = no truncation (full CV sent; needs larger context).
    max_cv_chars_for_llm: int = 0

    max_upload_size_mb: int = 10

    use_semantic_matching: bool = True
    embedding_provider: str = "sentence_transformers"
    pdf_font_path: str = ""

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

settings = Settings()
