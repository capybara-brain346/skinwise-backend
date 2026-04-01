from fastapi import APIRouter

router = APIRouter()

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

@router.get("/classes", tags=["Metadata"], summary="List supported classes", description="Returns the list of condition classes the model can predict.")
async def get_classes():
    return {"classes": CLASSES, "count": len(CLASSES)}