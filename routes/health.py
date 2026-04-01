from fastapi import APIRouter
from config.cnn import load_model
from config.llm import load_gemini
from s3 import get_s3_service

router = APIRouter()

session = load_model()
gemini_client = load_gemini()
s3_service = get_s3_service()

@router.get("/health", tags=["Health"], summary="Health check", description="Reports model, Gemini, and S3 availability.")
async def health_check():
    model_loaded = session is not None
    gemini_loaded = gemini_client is not None
    s3_configured = s3_service is not None
    return {
        "status": "healthy" if model_loaded else "model not loaded",
        "model_loaded": model_loaded,
        "gemini_loaded": gemini_loaded,
        "s3_configured": s3_configured,
    }