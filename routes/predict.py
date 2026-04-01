from fastapi import APIRouter
from fastapi import File, UploadFile, HTTPException
from PIL import Image
import io
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from config.cnn import load_model
from utils.preprocess import preprocess_image
from config.cnn import load_model
import numpy as np
from s3 import get_s3_service

router = APIRouter()

session = load_model()

s3_service = get_s3_service()

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

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

@router.post("/predict", tags=["Inference"], summary="Predict skin condition", description="Upload an image file using multipart/form-data with field name 'file'. Returns top prediction and confidences.")
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
            "all_confidences": results,
        }

        if s3_service:
            try:
                image_bytes = await file.read()
                file_key = f"uploads/{request_id}_{file.filename}"
                s3_service.upload_bytes(file_key, image_bytes, "image/jpeg")
                response_data["s3_key"] = file_key
            except Exception as e:
                print(f"Warning: Could not upload to S3: {e}")

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")