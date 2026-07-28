# PoC classification server

Small, dumb Flask server for the Expo photo-classification app. Loads a model
manually and exposes the two routes the app calls. Two interchangeable
backends, auto-detected by file extension:

- **TensorFlow** — `model/model.h5`, trained by `train.py`.
- **PyTorch** — `model/model.pt`, trained by `train_torch.py` (GPU on native
  Windows, e.g. RTX 4080).

`.pt` is preferred if present, else `.h5`; force one with `MODEL_PATH`.

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
pip install -r requirements.txt          # TensorFlow (CPU on Windows)
```

**PyTorch + GPU (native Windows):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-torch.txt
```

## Train

```bash
python train.py            # TensorFlow -> model/model.h5  + labels.txt
python train_torch.py      # PyTorch    -> model/model.pt  + labels.txt
```

Both scan `samples/<label>/` and build `(objects + 1)` output neurons with the
trailing one reserved for **"other"** (fed by `other/` or `unlabeled/`
folders).

The TensorFlow side is split in two files, no CLI flags — edit the constants:

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
MobileNetV3 and ConvNeXt can't be reloaded from legacy `.h5` under Keras 3 —
point `OUT_MODEL` at `model/model.keras` if you pick one.

### Fine-tuning (PyTorch, CLI flags)

Frozen-backbone training plateaus. To unfreeze and fine-tune the top backbone
stages after a short head warmup:

```bash
python train_torch.py --finetune --warmup-epochs 3 --epochs 40 \
                      --unfreeze 2 --lr 1e-3 --ft-lr 1e-4
```

- `--warmup-epochs` head-only epochs before unfreezing (`--lr`).
- `--finetune` then unfreezes the last `--unfreeze` backbone stages + head and
  trains `--epochs` more at the lower `--ft-lr`.
- Early stopping on `--patience` epochs without val gain (0 = off); the
  **best-val** weights are what gets saved. Frozen BatchNorm layers stay in eval
  mode to keep their stats stable on small data.

Underfitting (low train+val acc)? unfreeze more (`--unfreeze 3`) and/or add data.
Overfitting (train ≫ val)? fewer unfrozen stages, more data/augmentation.

## Plug your AI

Training writes the model + `labels.txt` for you. To plug a pre-made model
manually, drop `model/model.h5` (Keras) or `model/model.pt` (the dict saved by
`train_torch.py`). For `.h5`, also provide `model/labels.txt` in output order;
`.pt` carries its own labels. TF input not 224×224? set `IMG_HEIGHT`/`IMG_WIDTH`.

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
| `MODEL_PATH` | `.pt` if present else `.h5` | force a specific model file |
| `LABELS_PATH`| `model/labels.txt`   | class-name list (TF backend)     |
| `SAMPLES_DIR`| `samples`            | where `/sample` writes           |
| `IMG_HEIGHT` / `IMG_WIDTH` | `224` / `224` | TF backend input size   |
| `PORT`       | `8000`               | listen port                      |
