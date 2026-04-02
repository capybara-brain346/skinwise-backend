from fastapi import HTTPException
from config.llm import load_gemini
from s3 import get_s3_service
import uuid
from datetime import datetime
import json
from PIL import Image
from pydantic import BaseModel

gemini_client = load_gemini()
s3_service = get_s3_service()

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

request_id = str(uuid.uuid4())
timestamp = datetime.utcnow().isoformat()

async def image_analyzer(file, image):
    try:
        prompt = """You are an expert dermatologist AI assistant. Analyze this skin condition image and provide a detailed, structured assessment.

Please provide your analysis in the following JSON format (respond ONLY with valid JSON, no additional text):

{
  "condition_name": "The most likely skin condition name",
  "severity_level": "Mild/Moderate/Severe/Critical",
  "confidence_level": "Low/Medium/High",
  "visible_symptoms": ["symptom1", "symptom2", "symptom3"],
  "affected_area_description": "Detailed description of what you see in the image",
  "possible_causes": ["cause1", "cause2", "cause3"],
  "recommended_actions": ["action1", "action2", "action3"],
  "when_to_see_doctor": "Specific guidance on when immediate medical attention is needed",
  "additional_notes": "Any other relevant observations or information",
  "disclaimer": "Important disclaimer about AI limitations and need for professional medical advice"
}

Be thorough, professional, and ensure all fields are populated with relevant information. If you're uncertain about something, mention it in the confidence_level and additional_notes."""

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=[prompt, image]
        )

        response_text = response.text.strip()

        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        analysis_data = json.loads(response_text)

        analysis_data["request_id"] = request_id
        analysis_data["timestamp"] = timestamp

        detailed_analysis = DetailedAnalysis(**analysis_data)

        if s3_service is not None:
            try:
                image_filename = (
                    f"input_image_{file.filename}"
                    if file.filename
                    else "input_image.jpg"
                )
                s3_service.upload_image(contents, request_id, image_filename)

                s3_service.upload_analysis(analysis_data, request_id)

                print(
                    f"Successfully uploaded image and analysis data for request {request_id}"
                )
            except Exception as s3_error:
                print(f"Warning: Could not upload to S3: {s3_error}")

        return detailed_analysis

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing Gemini response: {str(e)}. Response: {response_text[:200]}",
        )