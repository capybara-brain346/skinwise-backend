from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from s3 import S3Service
from contextlib import asynccontextmanager
from config.cnn import load_model
from config.llm import load_gemini
from routes import predict, classes, health, root, analyze

load_dotenv()

tags_metadata = [
    {"name": "Info", "description": "General API information."},
    {"name": "Health", "description": "Service health and readiness."},
    {"name": "Metadata", "description": "Model and classes metadata."},
    {"name": "Inference", "description": "Image classification endpoints."},
    {"name": "Analysis", "description": "Detailed AI analysis endpoints."},
]

app = FastAPI(
    title="SkinWise API",
    description="Skin disease classification API using ResNet50",
    version="1.0.0",
    openapi_tags=tags_metadata,
)

app.include_router(root.router)
app.include_router(health.router)
app.include_router(classes.router)
app.include_router(predict.router)
app.include_router(analyze.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)