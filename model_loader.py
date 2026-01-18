from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 128

# NOTE: Verify this ordering matches how your model was trained.
# Commonly it comes from folder order used during training (often alphabetical).
CLASS_NAMES = [
    "pituitary",
    "meningioma",
    "notumor",
    "glioma",
]

_ROOT = Path(__file__).resolve().parent
MODEL_PATH = _ROOT / "models" / "mri_vgg16_model.keras"
SAVED_MODEL_DIR = (
    _ROOT
    / "models"
    / "mri_vgg16_model_tf-20260113T101522Z-1-001"
    / "mri_vgg16_model_tf"
)


class _SavedModelPredictor:
    def __init__(self, infer_fn):
        self._infer_fn = infer_fn

    def predict(self, x, verbose: int = 0):
        _ = verbose
        tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        output = self._infer_fn(tensor)
        value = list(output.values())[0]
        return value.numpy()


def load_trained_model() -> tf.keras.Model:
    # Prefer native Keras model when possible.
    try:
        print("✅ Loading .keras model...")
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        # Some older Keras configs serialize InputLayer with `batch_shape`,
        # which newer Keras may reject. Fall back to the exported SavedModel.
        print(f"⚠️ Failed to load .keras model ({e}).")
        print("➡️ Falling back to SavedModel...")
        sm = tf.saved_model.load(str(SAVED_MODEL_DIR))
        infer = sm.signatures.get("serve")
        if infer is None:
            infer = next(iter(sm.signatures.values()))
        print("✅ SavedModel loaded successfully")
        return _SavedModelPredictor(infer)  # type: ignore[return-value]


def preprocess_bgr_image(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty/invalid image")

    if img_bgr.ndim == 2:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    elif img_bgr.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)
