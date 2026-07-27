# PoC classification server

Small, dumb Flask server for the Expo photo-classification app. Loads a Keras
`.h5` model manually and exposes the two routes the app calls.

## Routes

| method | route      | body (multipart)          | returns                          |
|--------|------------|---------------------------|----------------------------------|
| GET    | `/`        | —                         | health + whether model is loaded |
| POST   | `/sample`  | `image` + optional `label`| saves photo under `samples/<label>/` |
| POST   | `/analyse` | `image`                   | `{ "label": "cat", "confidence": 0.93 }` |

## Setup (dev)

Python **3.11** (TensorFlow has no 3.14 wheels yet).

```bash
cd server
py -3.11 -m venv venv
source venv/Scripts/activate   # Windows Git Bash;  venv\Scripts\activate in cmd/PowerShell
pip install -r requirements.txt
```

## Plug your AI

1. Drop your trained Keras model at `server/model/model.h5`.
2. Put the class names in `server/model/labels.txt`, one per line, **in the
   model's output order**.
3. If your model's input isn't 224×224, set `IMG_HEIGHT` / `IMG_WIDTH`.

## Run

```bash
python app.py
```

Serves on `http://0.0.0.0:8000`. In the Expo app **Settings**, set the backend
URL to your machine's LAN IP (e.g. `http://192.168.1.10:8000`).

No model yet? `/sample` still saves photos; `/analyse` returns `503` until a
`.h5` is present.

## Config (env vars)

| var          | default            | meaning                     |
|--------------|--------------------|-----------------------------|
| `MODEL_PATH` | `model/model.h5`   | path to the `.h5` model     |
| `LABELS_PATH`| `model/labels.txt` | class-name list             |
| `SAMPLES_DIR`| `samples`          | where `/sample` writes      |
| `IMG_HEIGHT` / `IMG_WIDTH` | `224` / `224` | model input size |
| `PORT`       | `8000`             | listen port                 |
