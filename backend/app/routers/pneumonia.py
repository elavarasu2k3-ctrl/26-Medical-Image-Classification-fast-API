from fastapi import APIRouter, File, HTTPException, UploadFile
from app.config import MAX_UPLOAD_BYTES
from app.services.pneumonia import service

router=APIRouter(prefix="/api/v1/pneumonia",tags=["Pneumonia classification"])

@router.post("/predict")
async def predict_xray(file: UploadFile=File(...)):
    if file.content_type not in {"image/jpeg","image/png"}:
        raise HTTPException(415,"Upload a JPEG or PNG image.")
    raw=await file.read()
    if not raw: raise HTTPException(400,"Uploaded file is empty.")
    if len(raw)>MAX_UPLOAD_BYTES: raise HTTPException(413,"Image exceeds 10 MB.")
    try: return service.predict(raw)
    except Exception as exc: raise HTTPException(503,str(exc)) from exc
