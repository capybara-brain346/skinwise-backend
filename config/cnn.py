from s3 import S3Service
from pathlib import Path
import onnxruntime as ort

MODEL_PATH = Path("artifacts/resnet50_best_20251014_074535.onnx")

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