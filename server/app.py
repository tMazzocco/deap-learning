"""Small dumb Flask server for the photo-classification PoC.

Two routes, matching the Expo app:

  POST /sample   multipart: image (jpeg) + optional label
                 -> saves the photo under samples/<label>/ for dataset building.

  POST /analyse  multipart: image (jpeg)
                 -> runs the model and returns {label, confidence}.

Plug your model manually. Two backends, auto-detected by file extension:

  * model/model.h5  -> TensorFlow/Keras (train.py). Needs model/labels.txt.
  * model/model.pt  -> PyTorch/timm      (train_torch.py). Labels are baked in.

Set MODEL_PATH to force one, otherwise .pt is preferred if present, else .h5.
No model? /sample still works; /analyse returns 503.
"""

import io
import os
from datetime import datetime

from flask import Flask, jsonify, request
from PIL import Image

# --- Config (override via env) ---------------------------------------------
LABELS_PATH = os.environ.get("LABELS_PATH", os.path.join("model", "labels.txt"))
SAMPLES_DIR = os.environ.get("SAMPLES_DIR", "samples")
# Railguard: cap the longest edge of stored training samples (px). The phone
# already shrinks, this protects the dataset if a client sends something big.
MAX_SAMPLE_SIZE = int(os.environ.get("MAX_SAMPLE_SIZE", "1024"))
# TF backend only: model input size (H, W). Torch reads its own from the ckpt.
IMG_SIZE = (
    int(os.environ.get("IMG_HEIGHT", "224")),
    int(os.environ.get("IMG_WIDTH", "224")),
)

app = Flask(__name__)


def resolve_model_path():
    """Explicit MODEL_PATH wins; else prefer a .pt, then a .h5."""
    env = os.environ.get("MODEL_PATH")
    if env:
        return env
    for candidate in (os.path.join("model", "model.pt"), os.path.join("model", "model.h5")):
        if os.path.exists(candidate):
            return candidate
    return os.path.join("model", "model.h5")


MODEL_PATH = resolve_model_path()

# --- Lazy predictor loading -------------------------------------------------
# _predict is a cached callable: (PIL.Image) -> (label:str, confidence:float)
_predict = None
_file_labels = None


def get_file_labels():
    """labels.txt (used by the TF backend)."""
    global _file_labels
    if _file_labels is None:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                _file_labels = [line.strip() for line in f if line.strip()]
        else:
            _file_labels = []
    return _file_labels


def _load_tf(path):
    import numpy as np
    import tensorflow as tf

    model = tf.keras.models.load_model(path)
    labels = get_file_labels()

    def predict(img):
        img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))  # PIL is (W, H)
        # Raw [0, 255]: EfficientNetV2 normalizes inside the backbone, and
        # train.py feeds it the same way. Do NOT divide by 255 here.
        arr = np.asarray(img, dtype="float32")
        preds = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
        idx = int(np.argmax(preds))
        label = labels[idx] if idx < len(labels) else str(idx)
        return label, float(preds[idx])

    return predict


def _load_torch(path):
    import timm
    import torch
    from timm.data import create_transform, resolve_model_data_config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device)
    labels = ckpt["classes"]  # authoritative, in output order
    model = timm.create_model(ckpt["model_name"], num_classes=len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    transform = create_transform(**{**resolve_model_data_config(model), "is_training": False})

    def predict(img):
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x)[0], dim=0)
        conf, idx = torch.max(probs, dim=0)
        return labels[int(idx)], float(conf)

    return predict


def get_predictor():
    """Load the model once, on first /analyse call. None if no model file."""
    global _predict
    if _predict is None:
        if not os.path.exists(MODEL_PATH):
            return None
        if MODEL_PATH.endswith(".pt"):
            _predict = _load_torch(MODEL_PATH)
        else:
            _predict = _load_tf(MODEL_PATH)
        print(f"[model] loaded {MODEL_PATH}")
    return _predict


# --- Routes -----------------------------------------------------------------
@app.get("/")
def health():
    backend = "torch" if MODEL_PATH.endswith(".pt") else "tensorflow"
    return jsonify(
        status="ok",
        model_path=MODEL_PATH,
        model_loaded=os.path.exists(MODEL_PATH),
        backend=backend,
    )


@app.post("/sample")
def sample():
    if "image" not in request.files:
        return jsonify(error="missing 'image' file"), 400

    label = (request.form.get("label") or "unlabeled").strip() or "unlabeled"
    # Keep it filesystem-safe and dumb.
    label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)

    dest_dir = os.path.join(SAMPLES_DIR, label)
    os.makedirs(dest_dir, exist_ok=True)

    fname = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".jpg"
    path = os.path.join(dest_dir, fname)

    # Railguard: downscale if the longest edge exceeds MAX_SAMPLE_SIZE.
    img = Image.open(request.files["image"].stream).convert("RGB")
    long_edge = max(img.size)
    resized = False
    if long_edge > MAX_SAMPLE_SIZE:
        scale = MAX_SAMPLE_SIZE / long_edge
        img = img.resize((round(img.width * scale), round(img.height * scale)))
        resized = True
    img.save(path, format="JPEG", quality=90)

    print(f"[sample] saved {path} {img.size}{' (resized)' if resized else ''}")
    return jsonify(saved=path, label=label, size=list(img.size), resized=resized)


@app.post("/analyse")
def analyse():
    if "image" not in request.files:
        return jsonify(error="missing 'image' file"), 400

    predict = get_predictor()
    if predict is None:
        return jsonify(error=f"no model at {MODEL_PATH}"), 503

    img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    label, confidence = predict(img)

    print(f"[analyse] {label} ({confidence:.2f})")
    return jsonify(label=label, confidence=confidence)


if __name__ == "__main__":
    # 0.0.0.0 so the phone on the same LAN can reach it.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=True)
