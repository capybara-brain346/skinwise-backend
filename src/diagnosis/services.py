import os
import io
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

import onnxruntime as ort
from google import genai
from dotenv import load_dotenv

from s3 import S3Service


load_dotenv()


CLASSES: List[str] = [
    "Acne",
    "Eczema",
    "Psoriasis",
    "SkinCancer",
    "Tinea",
    "Unknown_Normal",
    "Vitiligo",
    "Warts",
]

MODEL_PATH = Path("artifacts/resnet50_best_20251014_074535.onnx")


class RuntimeState:
    session: Optional[ort.InferenceSession] = None
    gemini_client: Optional[genai.Client] = None
    s3_service: Optional[S3Service] = None


def init_s3_service() -> Optional[S3Service]:
    try:
        RuntimeState.s3_service = S3Service()
        return RuntimeState.s3_service
    except Exception:
        return None


def ensure_model_loaded() -> Optional[ort.InferenceSession]:
    if RuntimeState.session is not None:
        return RuntimeState.session

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    if not MODEL_PATH.exists():
        if RuntimeState.s3_service is not None:
            RuntimeState.s3_service.download_latest_model(
                model_prefix="models/", local_path=MODEL_PATH
            )
        else:
            return None

    RuntimeState.session = ort.InferenceSession(str(MODEL_PATH))
    return RuntimeState.session


def ensure_gemini_loaded() -> Optional[genai.Client]:
    if RuntimeState.gemini_client is not None:
        return RuntimeState.gemini_client
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    RuntimeState.gemini_client = genai.Client(api_key=api_key)
    return RuntimeState.gemini_client


def get_health_status() -> Dict[str, Any]:
    return {
        "status": "healthy" if RuntimeState.session is not None else "model not loaded",
        "model_loaded": RuntimeState.session is not None,
        "gemini_loaded": RuntimeState.gemini_client is not None,
        "s3_configured": RuntimeState.s3_service is not None,
    }


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32)
    img_array = img_array / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)


def run_prediction(image_bytes: bytes) -> Dict[str, Any]:
    if RuntimeState.s3_service is None:
        init_s3_service()
    session = ensure_model_loaded()
    if session is None:
        raise RuntimeError("Model not loaded")

    image = Image.open(io.BytesIO(image_bytes))
    input_data = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_data})
    predictions = outputs[0][0]
    probabilities = softmax(predictions)

    results = [
        {"class": CLASSES[i], "confidence": float(probabilities[i])}
        for i in range(len(CLASSES))
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)

    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    top_prediction = results[0]

    response_data = {
        "success": True,
        "request_id": request_id,
        "timestamp": timestamp,
        "prediction": top_prediction["class"],
        "confidence": top_prediction["confidence"],
        "top_5": results[:5],
        "all_predictions": results,
    }

    if RuntimeState.s3_service is not None:
        image_filename = "input_image.jpg"
        try:
            RuntimeState.s3_service.upload_image(
                image_bytes, request_id, image_filename
            )
            RuntimeState.s3_service.upload_prediction(response_data, request_id)
        except Exception:
            pass

    return response_data


def run_gemini_analysis(image_bytes: bytes) -> Dict[str, Any]:
    if RuntimeState.s3_service is None:
        init_s3_service()
    client = ensure_gemini_loaded()
    if client is None:
        raise RuntimeError("Gemini not configured")

    image = Image.open(io.BytesIO(image_bytes))
    prompt = (
        "You are an expert dermatologist AI assistant. Analyze this skin condition image and provide a detailed, structured assessment.\n\n"
        "Please provide your analysis in the following JSON format (respond ONLY with valid JSON, no additional text):\n\n"
        "{\n"
        '  "condition_name": "The most likely skin condition name",\n'
        '  "severity_level": "Mild/Moderate/Severe/Critical",\n'
        '  "confidence_level": "Low/Medium/High",\n'
        '  "visible_symptoms": ["symptom1", "symptom2", "symptom3"],\n'
        '  "affected_area_description": "Detailed description of what you see in the image",\n'
        '  "possible_causes": ["cause1", "cause2", "cause3"],\n'
        '  "recommended_actions": ["action1", "action2", "action3"],\n'
        '  "when_to_see_doctor": "Specific guidance on when immediate medical attention is needed",\n'
        '  "additional_notes": "Any other relevant observations or information",\n'
        '  "disclaimer": "Important disclaimer about AI limitations and need for professional medical advice"\n'
        "}\n\n"
        "Be thorough, professional, and ensure all fields are populated with relevant information. If you're uncertain about something, mention it in the confidence_level and additional_notes."
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp", contents=[prompt, image]
    )

    response_text = response.text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    data = json.loads(response_text)

    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    data["request_id"] = request_id
    data["timestamp"] = timestamp

    if RuntimeState.s3_service is not None:
        try:
            RuntimeState.s3_service.upload_image(
                image_bytes, request_id, "input_image.jpg"
            )
            RuntimeState.s3_service.upload_analysis(data, request_id)
        except Exception:
            pass

    return data
