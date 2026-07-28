"""Transfer-learn + fine-tune a backbone on the captured samples.

Photo ingestion lives in data.py; this file is the model and the schedule.
Two phases:

    1. warmup    backbone frozen, only the fresh head learns (LR)
    2. finetune  top UNFREEZE_LAYERS of the backbone unfrozen, lower FT_LR

Classes come from data.py: every object folder found, PLUS a trailing "other"
neuron fed by `other/` or `unlabeled/` folders.

    output neurons = (number of object labels) + 1        # last one = "other"

Outputs (consumed directly by app.py):
    model/model.h5     the trained Keras model
    model/labels.txt   class names, one per line, in output order

Run it:
    python train.py

No CLI flags: edit the constants below (and data.py for anything photo-side).
"""

import os

import numpy as np
import tensorflow as tf

import data

# --- knobs ------------------------------------------------------------------
SAMPLES_DIR = "samples"
OUT_MODEL = os.path.join("model", "model.h5")
OUT_LABELS = os.path.join("model", "labels.txt")

# The backbone. Its ImageNet weights are downloaded on first run. EfficientNet
# takes raw [0, 255] pixels (it normalizes internally) — swapping in a family
# that expects [-1, 1] means adding a Rescaling layer in build_model().
# Note: MobileNetV3 / ConvNeXt cannot be reloaded from legacy .h5 under Keras 3
# — if you pick one, change OUT_MODEL to model/model.keras.
BACKBONE = tf.keras.applications.EfficientNetV2B0

DROPOUT = 0.4
EPOCHS = 50
LR = 1e-3

FINETUNE = True  # phase 2 on/off
FINETUNE_EPOCHS = 15
FT_LR = 1e-4  # must stay well below LR or the pretrained weights get wrecked
# How many of the backbone's top layers to unfreeze. Keep it low: a few hundred
# near-duplicate photos cannot support a million trainable params — the phase
# just memorizes them and early stopping throws the whole thing away.
UNFREEZE_LAYERS = 15
PATIENCE = 5  # stop a phase after N epochs without val gain (0 = off)

print(f"[train] using TensorFlow {tf.__version__} with GPU: {tf.config.list_physical_devices('GPU')}")


def build_model(num_classes):
    """Frozen backbone + fresh (N+1)-way head. Returns (model, backbone)."""
    base = BACKBONE(
        include_top=False,
        weights="imagenet",
        input_shape=(data.IMG_SIZE, data.IMG_SIZE, 3),
        pooling="avg",
    )
    base.trainable = False  # phase 1: feature extraction

    inputs = tf.keras.Input(shape=(data.IMG_SIZE, data.IMG_SIZE, 3))
    # training=False pins BatchNorm to inference mode for good — that is what
    # keeps its running stats stable once we unfreeze for fine-tuning.
    x = base(inputs, training=False)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def unfreeze_top(base):
    """Unfreeze the top UNFREEZE_LAYERS of the backbone, BatchNorm excepted
    (its running stats would drift on a few hundred photos)."""
    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
    for layer in base.layers[-UNFREEZE_LAYERS:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def fit(tag, model, training_ds, test_ds, epochs, lr, class_weight=None):
    """Compile at the given LR and train. Recompiling is mandatory after any
    trainable-flag change, so each phase gets a fresh optimizer."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = []
    if test_ds is not None and PATIENCE:
        callbacks.append(
            # Keras calls the held-out metrics val_loss / val_accuracy — that
            # is our X_test / y_test. Watch the loss, not the accuracy: on a
            # small test set accuracy moves in coarse steps and ties resolve to
            # the earliest epoch, throwing away the epochs that actually
            # improved the model. Loss is continuous and sees that.
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=1,
            )
        )
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    print(f"[{tag}] {epochs} epochs, lr {lr:g}, trainable params {trainable:,}")
    model.fit(
        training_ds,
        validation_data=test_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        shuffle=False,
    )


def report(model, test_ds, class_names):
    """Confusion matrix + per-class accuracy, so a bad class is visible
    instead of hidden inside one average number."""
    y_test = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    n = len(class_names)
    matrix = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_test, y_pred):
        matrix[true, pred] += 1

    width = max(len(c) for c in class_names) + 2
    print("\n[eval] confusion matrix (rows = true, columns = predicted)")
    print(" " * width + "".join(f"{c:>14}" for c in class_names))
    for i, name in enumerate(class_names):
        print(f"{name:<{width}}" + "".join(f"{v:>14}" for v in matrix[i]))

    print("[eval] per-class accuracy")
    for i, name in enumerate(class_names):
        total = matrix[i].sum()
        acc = f"{matrix[i, i] / total:.3f}" if total else "  n/a"
        print(f"[eval]   {name:<{width}} {acc}  ({matrix[i, i]}/{total})")
    print(f"[eval] overall test accuracy {np.trace(matrix) / matrix.sum():.3f}")


def main():
    training_ds, test_ds, class_names, class_weight = data.load(SAMPLES_DIR)
    model, base = build_model(len(class_names))

    if EPOCHS > 0:
        fit("warmup", model, training_ds, test_ds, EPOCHS, LR, class_weight)

    if FINETUNE and FINETUNE_EPOCHS > 0:
        unfreeze_top(base)
        fit("finetune", model, training_ds, test_ds, FINETUNE_EPOCHS, FT_LR, class_weight)

    if test_ds is not None:
        report(model, test_ds, class_names)

    os.makedirs(os.path.dirname(OUT_MODEL) or ".", exist_ok=True)
    model.save(OUT_MODEL)
    with open(OUT_LABELS, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")

    print(f"[done] saved {OUT_MODEL} and {OUT_LABELS}")


if __name__ == "__main__":
    main()
