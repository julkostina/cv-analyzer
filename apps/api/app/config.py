from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, env_file_encoding="utf-8")
    
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    openai_api_key: str = ""
    gemini_api_key: str = ""
    environment: str = "development"

    # Ollama model for development (e.g. llama3.2:3b, llama3.2:8b). Larger models give better extraction on long CVs.
    ollama_model: str = "llama3.2:3b"

    # Max CV characters sent to the LLM (rest is truncated so the prompt fits in context). 0 = no limit.
    max_cv_chars_for_llm: int = 12_000

    max_upload_size_mb: int = 10

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

settings = Settings()
