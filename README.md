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

Or create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_api_key_here
```

4. Run the server:

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

### POST /analyze

Upload an image for detailed AI-powered analysis using Gemini 2.0 Flash. Requires `GOOGLE_API_KEY` environment variable.

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file)

**Response:**

```json
{
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

## Model Requirements

The ONNX model should:

- Accept input shape: [1, 3, 224, 224] (batch, channels, height, width)
- Use RGB color format
- Expect normalized images with ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
- Output logits for 22 classes
