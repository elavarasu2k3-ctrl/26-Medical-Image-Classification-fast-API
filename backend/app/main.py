from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import pneumonia, brain_tumor

app=FastAPI(
    title="Medical Image AI API",
    description="Research API for chest X-ray classification and brain MRI tumor segmentation.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=False,
                   allow_methods=["*"],allow_headers=["*"])
app.include_router(pneumonia.router)
app.include_router(brain_tumor.router)

@app.get("/",tags=["System"])
def root():
    return {"message":"Medical Image AI API is running","docs":"/docs",
            "endpoints":["/api/v1/pneumonia/predict","/api/v1/brain-tumor/segment"]}

@app.get("/health",tags=["System"])
def health():
    return {"status":"ok","service":"medical-image-ai-api"}
