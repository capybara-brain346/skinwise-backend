# SkinWise Backend

A FastAPI backend for skin disease classification using a ResNet50 ONNX model.

## Features

- Image upload endpoint for skin disease classification
- AI-powered detailed analysis using Gemini 2.5 Flash
- Support for 22 different skin conditions
- Returns predictions with confidence scores
- Fast inference using ONNX Runtime

## Supported Classes

1. Acne
2. Actinic Keratosis
3. Benign Tumors
4. Bullous
5. Candidiasis
6. Drug Eruption
7. Eczema
8. Infestations & Bites
9. Lichen
10. Lupus
11. Moles
12. Psoriasis
13. Rosacea
14. Seborrheic Keratoses
15. Skin Cancer
16. Sun/Sunlight Damage
17. Tinea
18. Unknown/Normal
19. Vascular Tumors
20. Vasculitis
21. Vitiligo
22. Warts

## Setup

1. Install dependencies using uv:

```bash
uv sync
```

2. Place your ONNX model file as `model.onnx` in the project root directory.

3. Set up your Gemini API key (required for the `/analyze` endpoint):

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

4. (Optional) Set up AWS S3 credentials for model storage and prediction data persistence:

```bash
export S3_BUCKET_NAME="your-s3-bucket-name"
export AWS_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="your_aws_access_key_id"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_access_key"
```

Or create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_api_key_here
S3_BUCKET_NAME=your-s3-bucket-name
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
```

5. Run the server:

```bash
uv run python main.py
```

Or using uvicorn directly:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### GET /

Root endpoint with API information.

### GET /health

Check if the API and model are loaded properly.

### GET /classes

Get list of all supported classification classes.

### POST /predict

Upload an image for classification using the ONNX ResNet50 model.

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file)

**Response:**

```json
{
  "success": true,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-01T12:00:00.000000",
  "prediction": "Acne",
  "confidence": 0.95,
  "top_5": [
    {"class": "Acne", "confidence": 0.95},
    {"class": "Rosacea", "confidence": 0.03},
    {"class": "Eczema", "confidence": 0.01},
    {"class": "Psoriasis", "confidence": 0.005},
    {"class": "Unknown_Normal", "confidence": 0.003}
  ],
  "all_predictions": [...]
}
```

**Note:** If S3 is configured, the image and prediction data are automatically stored in S3 under `predictions/{request_id}/`.

### POST /analyze

Upload an image for detailed AI-powered analysis using Gemini 2.0 Flash. Requires `GOOGLE_API_KEY` environment variable.

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file)

**Response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-01T12:00:00.000000",
  "condition_name": "Acne Vulgaris",
  "severity_level": "Moderate",
  "confidence_level": "High",
  "visible_symptoms": [
    "Multiple inflammatory papules",
    "Comedones present",
    "Mild erythema"
  ],
  "affected_area_description": "The image shows facial skin with multiple small to medium-sized red bumps concentrated around the cheek area...",
  "possible_causes": [
    "Hormonal changes",
    "Excess sebum production",
    "Bacterial colonization (P. acnes)"
  ],
  "recommended_actions": [
    "Maintain consistent cleansing routine",
    "Consider over-the-counter benzoyl peroxide treatment",
    "Avoid picking or squeezing lesions"
  ],
  "when_to_see_doctor": "Consult a dermatologist if condition persists for more than 3 months, worsens significantly, or causes scarring",
  "additional_notes": "No signs of cystic acne or severe inflammation. Good candidate for topical treatments.",
  "disclaimer": "This is an AI-powered analysis and should not replace professional medical advice. Always consult with a qualified healthcare provider for accurate diagnosis and treatment."
}
```

**Note:** If S3 is configured, the image and analysis data are automatically stored in S3 under `predictions/{request_id}/`.

## Testing with curl

**Test classification endpoint:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

**Test analysis endpoint:**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

## Testing with Python

**Test classification:**

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("path/to/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Test detailed analysis:**

```python
import requests

url = "http://localhost:8000/analyze"
files = {"file": open("path/to/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## API Documentation

Once the server is running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## S3 Integration

The application includes an S3 service (`s3.py`) that provides automatic cloud storage capabilities.

### Features

1. **Automatic Model Download**: At startup, if the model doesn't exist locally in the `artifacts/` directory, the latest ONNX model is automatically downloaded from S3 (sorted by timestamp)
2. **Automatic Image Storage**: Every prediction and analysis request automatically uploads the user's image to S3
3. **Automatic Prediction Storage**: All prediction results are automatically stored as JSON files in S3
4. **Automatic Analysis Storage**: All detailed AI analysis results are automatically stored in S3

All data is organized by unique request ID for easy retrieval and management.

### Usage Example

```python
from s3 import S3Service
import uuid

s3_service = S3Service()

request_id = str(uuid.uuid4())

s3_service.download_latest_model(model_prefix="models/")

with open("image.jpg", "rb") as f:
    image_data = f.read()
    s3_service.upload_image(image_data, request_id)

prediction_data = {
    "prediction": "Acne",
    "confidence": 0.95,
    "timestamp": "2024-01-01T12:00:00Z"
}
s3_service.upload_prediction(prediction_data, request_id)

retrieved_prediction = s3_service.get_prediction(request_id)

request_ids = s3_service.list_predictions()

s3_service.delete_prediction(request_id)
```

### S3 Bucket Structure

```
your-bucket/
├── models/
│   ├── resnet50_v1.onnx
│   └── resnet50_v2.onnx
└── predictions/
    └── {request_id}/
        ├── input_image.jpg
        ├── prediction.json
        └── analysis.json
```

## Model Requirements

The ONNX model should:

- Accept input shape: [1, 3, 224, 224] (batch, channels, height, width)
- Use RGB color format
- Expect normalized images with ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
- Output logits for 22 classes
