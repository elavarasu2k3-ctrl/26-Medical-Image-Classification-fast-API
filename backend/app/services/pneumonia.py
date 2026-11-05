import io, json
from pathlib import Path
import numpy as np
from PIL import Image
from app.config import MODEL_DIR

class PneumoniaService:
    def __init__(self):
        self.model = None
        self.card = None
        self.error = None

    def load(self):
        if self.model is not None:
            return
        model_path = MODEL_DIR / "pneumonia" / "pneumonia_densenet121.keras"
        card_path = MODEL_DIR / "pneumonia" / "model_card.json"
        if not model_path.exists():
            raise RuntimeError(f"Missing model weights: {model_path}")
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.card = json.loads(card_path.read_text())
        except Exception as exc:
            raise RuntimeError(f"Could not load TensorFlow pneumonia model: {exc}") from exc

    def predict(self, raw: bytes):
        self.load()
        with Image.open(io.BytesIO(raw)) as im:
            gray = im.convert("L").resize((160, 160), Image.Resampling.BILINEAR)
            arr = np.asarray(gray, dtype=np.uint8)
        rgb = np.stack([arr] * 3, axis=-1).astype("float32")
        prob = float(np.asarray(self.model.predict(rgb[None, ...], verbose=0)).reshape(-1)[0])
        threshold = float(self.card.get("decision_threshold", 0.5))
        label = "PNEUMONIA" if prob >= threshold else "NORMAL"
        return {
            "model": self.card.get("model_name", "DenseNet121"),
            "prediction": label,
            "probabilities": {"NORMAL": round(1.0-prob, 6), "PNEUMONIA": round(prob, 6)},
            "pneumonia_probability": round(prob, 6),
            "decision_threshold": threshold,
            "notice": "Research prototype only; not a medical diagnosis."
        }

service = PneumoniaService()
