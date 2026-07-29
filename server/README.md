# PoC classification server

Small, dumb Flask server for the Expo photo-classification app. Loads a model
manually and exposes the two routes the app calls. **TensorFlow/Keras only.**

- `model/model.keras` — native Keras 3 format, written by `train.py`.
- `model/model.h5` — legacy format, still loads.

Auto-detection: `model.keras` first, then `model.h5` — the first that exists
wins. Force any file with `MODEL_PATH`.

## Routes

A top class scoring below `CONF_THRESHOLD` (default **0.6**) is reported as
`other` rather than guessed. Softmax always sums to 1, so a photo of something
the model never trained on still produces a winner — the threshold is what keeps
that winner from being announced as a real object. The returned `confidence`
stays the top probability that failed the check, so the client can see how close
it was. Set `CONF_THRESHOLD=0` to disable.

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
pip install -r requirements.txt          # TensorFlow (CPU on Windows)
```

## Train

```bash
python train.py            # -> model/model.keras + labels.txt
```

It scans `samples/<label>/` and builds `(objects + 1)` output neurons with the
trailing one reserved for **"other"** (fed by `other/` or `unlabeled/`
folders).

Training is split in two files, no CLI flags — edit the constants:

| file       | what it owns                                              |
|------------|-----------------------------------------------------------|
| `data.py`  | photo ingestion: folder scan, `IMG_SIZE`, `BATCH`, `VAL_SPLIT`, train/val split, `AUGMENT` |
| `train.py` | `BACKBONE`, epochs, learning rates, fine-tuning, saving   |

`train.py` runs two phases: **warmup** (`EPOCHS` at `LR`, backbone frozen, only
the new head learns), then **finetune** (`FINETUNE_EPOCHS` at the lower `FT_LR`
with the top `UNFREEZE_LAYERS` of the backbone unfrozen). Set `FINETUNE = False`
to stop after the warmup. Each phase early-stops after `PATIENCE` epochs with no
val gain and keeps its best weights.

Underfitting (low train+val acc)? raise `UNFREEZE_LAYERS` and/or add data.
Overfitting (train ≫ val)? lower it, add data or stronger `AUGMENT`.

Swapping `BACKBONE` for another `tf.keras.applications` model: EfficientNet
takes raw `[0, 255]` pixels, so families expecting `[-1, 1]` need a `Rescaling`
layer in `build_model()`, and `data.IMG_SIZE` must match the backbone.
`OUT_MODEL` is `model/model.keras` (native Keras 3 format) — MobileNetV3 and
ConvNeXt can't be reloaded from legacy `.h5` under Keras 3, so leave it alone
unless you have a reason.

## Plug your AI

Training writes the model + `labels.txt` for you. To plug a pre-made model
manually, drop `model/model.keras` (or `model/model.h5`) plus
`model/labels.txt`, one class per line **in output order** — a mismatch there
shows up as confidently wrong labels, not as an error. Input not 224×224? set
`IMG_HEIGHT`/`IMG_WIDTH`. The server feeds raw `[0, 255]` pixels, so a model
without internal normalization needs its own `Rescaling` layer.

### labels.txt for a model trained elsewhere

If the model came from `image_dataset_from_directory(..., labels="inferred")`,
Keras assigned class indices as `sorted(os.listdir(dataset_dir))` — plain
alphabetical, **not** creation order and **not** "other last". Regenerate the
file from the dataset folder instead of writing it by hand:

```bash
python make_labels.py --dataset path/to/datasets/objects100 \
                      --model model/model.keras
```

`--model` compares the class count against the model's output units and refuses
to write on a mismatch. `--dry-run` prints the order without touching anything.

## Run

```bash
python app.py
```

Serves on `http://0.0.0.0:8000`. In the Expo app **Settings**, set the backend
URL to your machine's LAN IP (e.g. `http://192.168.1.10:8000`).

No model yet? `/sample` still saves photos; `/analyse` returns `503` until a
model is present.

## Config (env vars)

| var          | default              | meaning                          |
|--------------|----------------------|----------------------------------|
| `MODEL_PATH` | `model.keras`, else `model.h5` | force a specific model file |
| `LABELS_PATH`| `model/labels.txt`   | class-name list, in output order |
| `CONF_THRESHOLD` | `0.6`            | below this top probability, answer `other` (0 = off) |
| `OTHER_LABEL`| `other`              | label used when the threshold rejects |
| `SAMPLES_DIR`| `samples`            | where `/sample` writes           |
| `IMG_HEIGHT` / `IMG_WIDTH` | `224` / `224` | model input size        |
| `PORT`       | `8000`               | listen port                      |
