from src.helpers.config import get_settings
from fastapi import FastAPI
from src.routes import base

app = FastAPI()

async def startup_span():
    settings = get_settings()

app.include_router(base.base_router)