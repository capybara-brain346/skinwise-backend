### SkinWise API

Base URL: `http://localhost:8000`

OpenAPI/Swagger UI: `http://localhost:8000/docs`

### Endpoints

- **GET /** (Info)

  - Summary: API root
  - Response example:
    ```json
    {
      "message": "SkinWise API is running",
      "endpoints": {
        "predict": "/predict",
        "analyze": "/analyze",
        "health": "/health",
        "classes": "/classes"
      }
    }
    ```

- **GET /health** (Health)

  - Summary: Health check
  - Example:
    ```bash
    curl -s http://localhost:8000/health | jq
    ```

- **GET /classes** (Metadata)

  - Summary: List supported classes
  - Example:
    ```bash
    curl -s http://localhost:8000/classes | jq
    ```

- **POST /predict** (Inference)

  - Summary: Predict skin condition
  - Description: multipart/form-data with field name `file`
  - Example:
    ```bash
    curl -s -X POST http://localhost:8000/predict \
      -F file=@/path/to/image.jpg | jq
    ```

- **POST /analyze** (Analysis)
  - Summary: Detailed AI analysis
  - Requirements: env `GOOGLE_API_KEY` and image `file`
  - Example:
    ```bash
    curl -s -X POST http://localhost:8000/analyze \
      -F file=@/path/to/image.jpg | jq
    ```

### Running locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Notes

- Image files must use content type starting with `image/`.
- `predict` uses an ONNX ResNet50 model. Ensure model file exists at `artifacts/resnet50_best_20251011_163346.onnx`.
- `analyze` uses Gemini via `GOOGLE_API_KEY`.
