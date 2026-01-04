from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, env_file_encoding="utf-8")
    
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    openai_api_key: str = ""

    environment: str = "development"

settings = Settings()
