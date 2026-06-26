from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from config.llm import load_openrouter
import io
from PIL import Image
from pydantic import BaseModel
from utils.analyzer import image_analyzer

router = APIRouter()

openrouter_client = load_openrouter()

class DetailedAnalysis(BaseModel):
    request_id: str
    timestamp: str
    condition_name: str
    severity_level: str
    confidence_level: str
    visible_symptoms: list[str]
    affected_area_description: str
    possible_causes: list[str]
    recommended_actions: list[str]
    when_to_see_doctor: str
    additional_notes: str
    disclaimer: str

@router.post("/analyze", response_model=DetailedAnalysis, tags=["Analysis"], summary="Detailed AI analysis", description="Performs detailed analysis using Gemini and returns comprehensive insights.")
async def analyze_with_gemini(file: UploadFile = File(...), language: str = Form("English")):
    if openrouter_client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenRouter not configured. Please set OPENROUTER_API_KEY environment variable.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    try:
        return await image_analyzer(file, image, contents, language)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing image with Gemini: {str(e)}"
        )
