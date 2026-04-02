from fastapi import APIRouter

router = APIRouter()

@router.get("/", tags=["Info"], summary="API root", description="Service index and available endpoints.")
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