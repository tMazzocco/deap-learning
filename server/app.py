"""Small dumb Flask server for the photo-classification PoC.

Two routes, matching the Expo app:

  POST /sample   multipart: image (jpeg) + optional label
                 -> saves the photo under samples/<label>/ for dataset building.

  POST /analyse  multipart: image (jpeg)
                 -> runs the model and returns {label, confidence}. A top class
                    below CONF_THRESHOLD is reported as OTHER_LABEL.

Plug your model manually. TensorFlow/Keras only:

  * model/model.keras -> native Keras 3 format, written by train.py.
  * model/model.h5    -> same, legacy format.

Both need model/labels.txt, one class per line in output order. Set MODEL_PATH
to force a file, otherwise .keras wins over .h5.
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
# Model input size (H, W). Must match what the model was trained at.
IMG_SIZE = (
    int(os.environ.get("IMG_HEIGHT", "224")),
    int(os.environ.get("IMG_WIDTH", "224")),
)
# Below this top-class probability the answer is not trusted and gets reported
# as OTHER_LABEL instead. Softmax always sums to 1, so a model shown something
# it never trained on still returns a winner — this is what stops that winner
# from being announced as a real object. 0 disables the check.
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.6"))
OTHER_LABEL = os.environ.get("OTHER_LABEL", "other")

app = Flask(__name__)


# Auto-detection order, highest priority first. .keras is the native Keras 3
# format train.py writes; .h5 is the legacy fallback.
MODEL_CANDIDATES = (
    os.path.join("model", "model.keras"),
    os.path.join("model", "model.h5"),
)


def resolve_model_path():
    """Explicit MODEL_PATH wins; else first existing MODEL_CANDIDATES entry."""
    env = os.environ.get("MODEL_PATH")
    if env:
        return env
    for candidate in MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return MODEL_CANDIDATES[0]


MODEL_PATH = resolve_model_path()

# --- Lazy predictor loading -------------------------------------------------
# _predict is a cached callable: (PIL.Image) -> (label:str, confidence:float)
_predict = None
_file_labels = None


def get_file_labels():
    """labels.txt, one class per line in the model's output order."""
    global _file_labels
    if _file_labels is None:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                _file_labels = [line.strip() for line in f if line.strip()]
        else:
            _file_labels = []
    return _file_labels


def _load_model(path):
    import numpy as np
    import tensorflow as tf

    model = tf.keras.models.load_model(path)
    labels = get_file_labels()
    if CONF_THRESHOLD > 0 and OTHER_LABEL not in labels:
        print(
            f"[warn] '{OTHER_LABEL}' is not in {LABELS_PATH}; low-confidence "
            f"answers will still be reported as '{OTHER_LABEL}'"
        )

    def predict(img):
        img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))  # PIL is (W, H)
        # Raw [0, 255]: EfficientNetV2 normalizes inside the backbone, and
        # train.py feeds it the same way. Do NOT divide by 255 here.
        arr = np.asarray(img, dtype="float32")
        preds = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        if conf < CONF_THRESHOLD:
            # Keep conf as the top probability that failed, not the "other"
            # class's own: the client shows how close the call was.
            return OTHER_LABEL, conf
        label = labels[idx] if idx < len(labels) else str(idx)
        return label, conf

    return predict


def get_predictor():
    """Load the model once, on first /analyse call. None if no model file."""
    global _predict
    if _predict is None:
        if not os.path.exists(MODEL_PATH):
            return None
        _predict = _load_model(MODEL_PATH)
        print(f"[model] loaded {MODEL_PATH}")
    return _predict


# --- Routes -----------------------------------------------------------------
@app.get("/")
def health():
    return jsonify(
        status="ok",
        model_path=MODEL_PATH,
        model_loaded=os.path.exists(MODEL_PATH),
        backend="tensorflow",
        conf_threshold=CONF_THRESHOLD,
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
