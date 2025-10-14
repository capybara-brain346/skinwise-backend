from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import onnxruntime as ort
import io
import os
from pathlib import Path
from google import genai
from pydantic import BaseModel
import json
import uuid
from datetime import datetime
from s3 import S3Service
from contextlib import asynccontextmanager

load_dotenv()

tags_metadata = [
    {"name": "Info", "description": "General API information."},
    {"name": "Health", "description": "Service health and readiness."},
    {"name": "Metadata", "description": "Model and classes metadata."},
    {"name": "Inference", "description": "Image classification endpoints."},
    {"name": "Analysis", "description": "Detailed AI analysis endpoints."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_s3_service()
    except Exception as e:
        print(f"Warning: Could not initialize S3 service: {e}")

    try:
        load_model()
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")

    try:
        load_gemini()
        print("Gemini model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load Gemini model: {e}")

    yield


app = FastAPI(
    title="SkinWise API",
    description="Skin disease classification API using ResNet50",
    version="1.0.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = [
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
session = None
gemini_client = None
s3_service = None


def init_s3_service():
    global s3_service
    try:
        s3_service = S3Service()
        print("S3 service initialized successfully")
        return s3_service
    except ValueError as e:
        print(f"S3 service not configured: {e}")
        return None
    except Exception as e:
        print(f"Warning: Could not initialize S3 service: {e}")
        return None


def load_model():
    global session, s3_service

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")

        if s3_service is not None:
            try:
                print("Attempting to download latest model from S3...")
                downloaded_path = s3_service.download_latest_model(
                    model_prefix="models/", local_path=MODEL_PATH
                )
                print(f"Model downloaded successfully to {downloaded_path}")
            except Exception as e:
                raise FileNotFoundError(
                    f"Model file not found at {MODEL_PATH} and could not download from S3: {e}"
                )
        else:
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH} and S3 service not configured"
            )
    else:
        print(f"Model already exists at {MODEL_PATH}, skipping download")

    session = ort.InferenceSession(str(MODEL_PATH))
    return session


def load_gemini():
    global gemini_client
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found in environment variables")
        return None
    gemini_client = genai.Client(api_key=api_key)
    return gemini_client


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


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)


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


@app.get(
    "/",
    tags=["Info"],
    summary="API root",
    description="Service index and available endpoints.",
)
async def root():
    return {
        "message": "SkinWise API is running",
        "endpoints": {
            "predict": "/predict",
            "analyze": "/analyze",
            "health": "/health",
            "classes": "/classes",
        },
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
    description="Reports model, Gemini, and S3 availability.",
)
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


@app.get(
    "/classes",
    tags=["Metadata"],
    summary="List supported classes",
    description="Returns the list of condition classes the model can predict.",
)
async def get_classes():
    return {"classes": CLASSES, "count": len(CLASSES)}


@app.post(
    "/predict",
    tags=["Inference"],
    summary="Predict skin condition",
    description="Upload an image file using multipart/form-data with field name 'file'. Returns top prediction and confidences.",
)
async def predict(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model.onnx exists in the project root.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

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

        if s3_service is not None:
            try:
                image_filename = (
                    f"input_image_{file.filename}"
                    if file.filename
                    else "input_image.jpg"
                )
                s3_service.upload_image(contents, request_id, image_filename)

                s3_service.upload_prediction(response_data, request_id)

                print(
                    f"Successfully uploaded image and prediction data for request {request_id}"
                )
            except Exception as s3_error:
                print(f"Warning: Could not upload to S3: {s3_error}")

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post(
    "/analyze",
    response_model=DetailedAnalysis,
    tags=["Analysis"],
    summary="Detailed AI analysis",
    description="Generates a structured dermatology analysis using Gemini. Requires GOOGLE_API_KEY.",
)
async def analyze_with_gemini(file: UploadFile = File(...)):
    if gemini_client is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini AI not configured. Please set GOOGLE_API_KEY environment variable.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

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
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing image with Gemini: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
