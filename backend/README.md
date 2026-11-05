# Medical Image AI — FastAPI Backend

This backend combines the two supplied research projects:
1. Chest X-ray binary classification using the supplied DenseNet121 Keras model.
2. Brain MRI tumor segmentation using the supplied U-Net variants (U-Net, ResUNet, Attention U-Net, U-Net++).

## Project structure

```text
backend/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── routers/
│   │   ├── pneumonia.py
│   │   └── brain_tumor.py
│   └── services/
│       ├── pneumonia.py
│       └── brain_tumor.py
├── models/
│   ├── pneumonia/
│   │   ├── pneumonia_densenet121.keras
│   │   └── model_card.json
│   └── brain_tumor/
│       ├── UNet.pt
│       ├── ResUNet.pt
│       ├── AttentionUNet.pt
│       ├── UNet++.pt
│       └── deploy_config.json
├── requirements.txt
└── README.md
```

## Setup (Windows PowerShell)

Use Python 3.11 or 3.12 for the smoothest compatibility with TensorFlow and PyTorch.

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installing TensorFlow and PyTorch together causes dependency or disk-space issues, install the CPU builds separately according to their official installation instructions.

## Run

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open Swagger UI: http://127.0.0.1:8000/docs

## API endpoints

### 1. Pneumonia classification

`POST /api/v1/pneumonia/predict`

Multipart form-data:
- `file`: JPEG or PNG chest X-ray (max 10 MB)

Example with curl:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/pneumonia/predict" \
  -F "file=@chest_xray.png"
```

Example response:

```json
{
  "model": "Pneumonia classifier (DenseNet121 transfer learning)",
  "prediction": "PNEUMONIA",
  "probabilities": {"NORMAL": 0.12, "PNEUMONIA": 0.88},
  "pneumonia_probability": 0.88,
  "decision_threshold": 0.0981,
  "notice": "Research prototype only; not a medical diagnosis."
}
```

The X-ray is converted to grayscale, resized to 160x160, and replicated into 3 channels. The exported Keras model contains its own model preprocessing. The classification threshold is read from the supplied model card.

### 2. Brain tumor segmentation

`GET /api/v1/brain-tumor/models`

`POST /api/v1/brain-tumor/segment`

Multipart form-data:
- `file`: JPEG or PNG brain MRI slice (max 10 MB)
- `model` (optional): `UNet`, `ResUNet`, `AttentionUNet`, or `UNet++`
- `threshold` (optional): probability threshold between 0 and 1

Example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/brain-tumor/segment" \
  -F "file=@brain_mri.png" \
  -F "model=ResUNet"
```

Response contains:
- selected model and threshold
- predicted tumor pixel count at model resolution
- `mask_png_base64`: binary segmentation mask encoded as PNG
- `probability_map_png_base64`: probability heatmap encoded as PNG
- research-use notice

The default segmentation model is ResUNet, with threshold and minimum-area filtering derived from the supplied project validation/test artifacts. Models load lazily on first request.

## Important model details

- Pneumonia model: DenseNet121, 160x160 input, threshold 0.0981.
- Brain MRI models: the supplied notebook architecture, 128x128 input and base channel width 16. Normalization values are taken from the notebook output. Default model is ResUNet; threshold 0.85 and minimum area 37 pixels.
- The supplied model checkpoints are used; this backend does not train or fine-tune models.

## Validation and limitations

The checkpoints are wired to inference endpoints, but run a local smoke test with representative images before presenting the API. In particular, confirm checkpoint loading on the target Python/PyTorch version. Uploaded images must be the same modality/domain as the training data.

**Research/educational prototype only.** These models are not medical devices and must not be used to diagnose, rule out, or guide treatment. Predictions can be wrong; clinical decisions require qualified medical professionals and validated systems.
