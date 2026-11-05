from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.config import MAX_UPLOAD_BYTES
from app.services.brain_tumor import service

router=APIRouter(prefix="/api/v1/brain-tumor",tags=["Brain tumor segmentation"])
VARIANTS=["UNet","ResUNet","AttentionUNet","UNet++"]

@router.get("/models")
def list_models():
    return {"available_models":VARIANTS,"default_model":service.cfg["model"],
            "default_threshold":service.cfg["threshold"],"default_min_area":service.cfg["min_area"]}

@router.post("/segment")
async def segment_mri(file: UploadFile=File(...),
                      model: Optional[str]=Form(None),
                      threshold: Optional[float]=Form(None)):
    if file.content_type not in {"image/jpeg","image/png"}:
        raise HTTPException(415,"Upload a JPEG or PNG MRI image.")
    if model is not None and model not in VARIANTS:
        raise HTTPException(422,f"model must be one of: {', '.join(VARIANTS)}")
    if threshold is not None and not 0.0 < threshold < 1.0:
        raise HTTPException(422,"threshold must be between 0 and 1.")
    raw=await file.read()
    if not raw: raise HTTPException(400,"Uploaded file is empty.")
    if len(raw)>MAX_UPLOAD_BYTES: raise HTTPException(413,"Image exceeds 10 MB.")
    try: return service.segment(raw,model,threshold)
    except Exception as exc: raise HTTPException(503,str(exc)) from exc
