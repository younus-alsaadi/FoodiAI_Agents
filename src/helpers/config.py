from pydantic_settings import BaseSettings
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent          # …/src/helpers
ENV_PATH = THIS_DIR.parent / ".env"

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    OPENAI_API_KEY:str
    TAVILY_API_KEY:str
    YOUTUBE_API_KEY: str

    LANGCHAIN_TRACING_V2:str
    LANGCHAIN_ENDPOINT:str
    LANGCHAIN_API_KEY:str
    LANGCHAIN_PROJECT:str

    class Config:
        env_file = ENV_PATH

def get_settings() -> Settings:

    return Settings()