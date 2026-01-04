from typing import Union
from app.config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.cv_router import router 

app = FastAPI()

if settings.environment == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(router, prefix="/api/v1", tags="cv")

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "environment": settings.environment}