"""Transfer-learn EfficientNetV2-B0 on the captured samples.

Reads samples/<label>/*.jpg (the folders written by /sample), and fine-tunes a
head on top of a frozen ImageNet EfficientNetV2-B0 backbone.

Classes = every object folder found, PLUS a trailing "other" neuron:

    output neurons = (number of object labels) + 1        # last one = "other"

Any image sitting in an `other/` or `unlabeled/` folder feeds that last neuron.

Outputs (consumed directly by app.py):
    model/model.h5     the trained Keras model
    model/labels.txt   class names, one per line, in output order

Quick local smoke test (default 1 epoch):
    python train.py

Real training happens elsewhere — bump epochs / unfreeze there:
    python train.py --epochs 30
"""

import argparse
import os

import tensorflow as tf

# EfficientNetV2 models expect raw pixels in [0, 255] (normalization is baked
# into the backbone), so we do NOT rescale here.
IMAGENET_INPUT = 224

# Folder names that are not real objects -> folded into the "other" class.
OTHER_ALIASES = {"other", "unlabeled"}


def collect(samples_dir):
    """Return (files, labels, class_names).

    class_names ends with "other"; label ints index into class_names.
    """
    if not os.path.isdir(samples_dir):
        raise SystemExit(f"no samples dir: {samples_dir}")

    subdirs = sorted(
        d for d in os.listdir(samples_dir) if os.path.isdir(os.path.join(samples_dir, d))
    )
    object_labels = [d for d in subdirs if d.lower() not in OTHER_ALIASES]
    class_names = object_labels + ["other"]  # <-- the +1 neuron, always last
    other_idx = len(class_names) - 1
    index_of = {name: i for i, name in enumerate(object_labels)}

    files, labels = [], []
    for d in subdirs:
        idx = index_of.get(d, other_idx)  # unknown/alias folders -> other
        folder = os.path.join(samples_dir, d)
        for fn in os.listdir(folder):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append(os.path.join(folder, fn))
                labels.append(idx)

    if not files:
        raise SystemExit(f"no images found under {samples_dir}")
    return files, labels, class_names


def make_ds(files, labels, img_size, batch, training):
    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32)  # keep [0, 255] for EfficientNetV2
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if training:
        ds = ds.shuffle(min(len(files), 1000), reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes, img_size):
    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
        pooling="avg",
    )
    base.trainable = False  # feature-extraction for the quick test

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    # Light augmentation, only active during training.
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="samples")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=IMAGENET_INPUT)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--out", default=os.path.join("model", "model.h5"))
    ap.add_argument("--labels-out", default=os.path.join("model", "labels.txt"))
    args = ap.parse_args()

    files, labels, class_names = collect(args.samples)
    print(f"[data] {len(files)} images, {len(class_names)} classes: {class_names}")

    # Simple deterministic split.
    idx = tf.range(len(files))
    idx = tf.random.shuffle(idx, seed=42).numpy()
    files = [files[i] for i in idx]
    labels = [labels[i] for i in idx]

    n_val = int(len(files) * args.val_split)
    val_ds = None
    if n_val >= len(class_names) and len(files) - n_val > 0:
        train_ds = make_ds(files[n_val:], labels[n_val:], args.img_size, args.batch, True)
        val_ds = make_ds(files[:n_val], labels[:n_val], args.img_size, args.batch, False)
    else:
        print("[data] too few images for a validation split, training on all")
        train_ds = make_ds(files, labels, args.img_size, args.batch, True)

    model = build_model(len(class_names), args.img_size)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    with open(args.labels_out, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")

    print(f"[done] saved {args.out} and {args.labels_out}")


if __name__ == "__main__":
    main()
