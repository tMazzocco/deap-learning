"""Small dumb Flask server for the photo-classification PoC.

Two routes, matching the Expo app:

  POST /sample   multipart: image (jpeg) + optional label
                 -> saves the photo under samples/<label>/ for dataset building.

  POST /analyse  multipart: image (jpeg)
                 -> runs the .h5 Keras model and returns {label, confidence}.

Plug your model manually: drop a Keras .h5 file at model/model.h5 (or set
MODEL_PATH) and list your class names in model/labels.txt (one per line, in the
model's output order). No model? /sample still works; /analyse returns 503.
"""

import io
import os
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

# --- Config (override via env) ---------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("model", "model.h5"))
LABELS_PATH = os.environ.get("LABELS_PATH", os.path.join("model", "labels.txt"))
SAMPLES_DIR = os.environ.get("SAMPLES_DIR", "samples")
# Model input size (H, W). Change to match your model.
IMG_SIZE = (
    int(os.environ.get("IMG_HEIGHT", "224")),
    int(os.environ.get("IMG_WIDTH", "224")),
)

app = Flask(__name__)

# --- Lazy model loading -----------------------------------------------------
_model = None
_labels = None


def get_labels():
    global _labels
    if _labels is None:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                _labels = [line.strip() for line in f if line.strip()]
        else:
            _labels = []
    return _labels


def get_model():
    """Load the .h5 model once, on first /analyse call."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            return None
        # Import here so the server boots even without tensorflow installed yet.
        import tensorflow as tf

        _model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[model] loaded {MODEL_PATH}")
    return _model


def read_image(file_storage):
    img = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))  # PIL is (W, H)
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


# --- Routes -----------------------------------------------------------------
@app.get("/")
def health():
    return jsonify(
        status="ok",
        model_loaded=os.path.exists(MODEL_PATH),
        labels=get_labels(),
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
    request.files["image"].save(path)

    print(f"[sample] saved {path}")
    return jsonify(saved=path, label=label)


@app.post("/analyse")
def analyse():
    if "image" not in request.files:
        return jsonify(error="missing 'image' file"), 400

    model = get_model()
    if model is None:
        return jsonify(error=f"no model at {MODEL_PATH}"), 503

    x = read_image(request.files["image"])
    preds = model.predict(x, verbose=0)[0]

    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    labels = get_labels()
    label = labels[idx] if idx < len(labels) else str(idx)

    print(f"[analyse] {label} ({confidence:.2f})")
    return jsonify(label=label, confidence=confidence)


if __name__ == "__main__":
    # 0.0.0.0 so the phone on the same LAN can reach it.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=True)
