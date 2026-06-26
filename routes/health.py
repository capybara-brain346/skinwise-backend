from fastapi import APIRouter
from config.cnn import load_model
from config.llm import load_openrouter
from s3 import get_s3_service

router = APIRouter()

session = load_model()
openrouter_client = load_openrouter()
s3_service = get_s3_service()

@router.get("/health", tags=["Health"], summary="Health check", description="Reports model, OpenRouter, and S3 availability.")
async def health_check():
    model_loaded = session is not None
    openrouter_loaded = openrouter_client is not None
    s3_configured = s3_service is not None
    return {
        "status": "healthy" if model_loaded else "model not loaded",
        "model_loaded": model_loaded,
        "openrouter_loaded": openrouter_loaded,
        "s3_configured": s3_configured,
    }
