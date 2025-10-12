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
- `predict` uses an ONNX ResNet50 model. If the model doesn't exist locally, it will be automatically downloaded from S3 if configured.
- `analyze` uses Gemini via `GOOGLE_API_KEY`.
- All responses include a unique `request_id` and `timestamp`.
- If S3 is configured (via `S3_BUCKET_NAME` env var), all images and predictions are automatically stored in S3 under `predictions/{request_id}/`.
- S3 environment variables: `S3_BUCKET_NAME`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.
