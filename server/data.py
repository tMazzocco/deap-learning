"""Photo ingestion for train.py: samples/ folders -> tf.data datasets.

Everything about reading photos lives here. Change how images are found,
resized, split or augmented in this file; change the model and the training
schedule in train.py.

Folder layout (what POST /sample writes):

    samples/
      ecouteurs/*.jpg        -> class 0
      keycap-puller/*.jpg    -> class 1
      other/*.jpg            -> the trailing "other" neuron
      unlabeled/*.jpg        -> same, alias of other

    class_names = sorted object folders + ["other"]     # "other" always last
"""

import os
import random
from datetime import datetime, timedelta

import tensorflow as tf

# --- knobs ------------------------------------------------------------------
IMG_SIZE = 224  # must match the backbone in train.py
BATCH = 16
TEST_SPLIT = 0.2  # share of each class held out as X_test / y_test
SEED = 42

# Photos taken seconds apart are near-duplicates: same lighting, same
# background, object barely moved. Splitting them frame-by-frame leaks the
# test set (the same shot ends up on both sides) and the score lies. So group
# photos into capture sessions and send WHOLE sessions to the test set.
GROUP_BY_SESSION = True
SESSION_GAP = timedelta(minutes=10)  # a gap this long starts a new session

# Folder names that are not real objects -> folded into the "other" class.
OTHER_ALIASES = {"other", "unlabeled"}
IMG_EXT = (".jpg", ".jpeg", ".png")

# Training-only augmentation. Add/remove layers here. Brightness and contrast
# matter most: every class was shot under one lighting, so this is the only
# thing teaching the model that lighting is not the label.
AUGMENT = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.3, value_range=(0, 255)),
        tf.keras.layers.RandomContrast(0.3),
    ],
    name="augment",
)

# "other" is the broadest class (everything that is not an object) but usually
# has the fewest photos. Weight the loss by class frequency so it is not
# drowned out. Set False to train on raw counts.
BALANCE_CLASSES = True


def scan(samples_dir):
    """Walk samples_dir -> (files, labels, class_names)."""
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
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(IMG_EXT):
                files.append(os.path.join(folder, fn))
                labels.append(idx)

    if not files:
        raise SystemExit(f"no images found under {samples_dir}")
    return files, labels, class_names


def timestamp(path):
    """Capture time from the /sample filename (YYYYmmdd-HHMMSS-micro.jpg)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return datetime.strptime(stem[:15], "%Y%m%d-%H%M%S")
    except ValueError:
        return None  # hand-dropped file: no session info


def sessions(paths):
    """Split one class's photos into capture sessions (list of lists)."""
    stamped = sorted((t, p) for p in paths if (t := timestamp(p)) is not None)
    loose = [p for p in paths if timestamp(p) is None]

    groups, previous = [], None
    for when, path in stamped:
        if previous is None or when - previous > SESSION_GAP:
            groups.append([])
        groups[-1].append(path)
        previous = when
    if loose:
        groups.append(loose)  # unknown times: one bucket
    return groups


def split(files, labels, class_names):
    """Per-class split. Whole capture sessions go to the test set when a class
    has more than one; otherwise it falls back to a frame-level split and says
    so, loudly, because that score is not trustworthy.

    Returns (X_training, y_training), (X_test, y_test) — X = image paths,
    y = class indices.
    """
    rng = random.Random(SEED)
    by_class = [[] for _ in class_names]
    for path, label in zip(files, labels):
        by_class[label].append(path)

    X_training, y_training, X_test, y_test = [], [], [], []
    leaky = []
    for cls, paths in enumerate(by_class):
        groups = sessions(paths) if GROUP_BY_SESSION else []
        target = int(round(len(paths) * TEST_SPLIT))

        if len(groups) >= 2:
            # Hold out whole sessions until the target is met, always leaving
            # at least one session to train on.
            order = list(range(len(groups)))
            rng.shuffle(order)
            held, taken = set(), 0
            for i in order:
                if taken >= target or len(held) >= len(groups) - 1:
                    break
                held.add(i)
                taken += len(groups[i])
            X_test_cls = [p for i in held for p in groups[i]]
            X_training_cls = [p for i in range(len(groups)) if i not in held for p in groups[i]]
            print(f"[data] {class_names[cls]}: {len(groups)} sessions, {len(held)} held out for test")
        else:
            # One session only -> every photo is a near-duplicate of the rest.
            leaky.append(class_names[cls])
            shuffled = sorted(paths)
            rng.shuffle(shuffled)
            n_test = min(target, max(len(shuffled) - 1, 0))
            X_test_cls, X_training_cls = shuffled[:n_test], shuffled[n_test:]

        X_training += X_training_cls
        y_training += [cls] * len(X_training_cls)
        X_test += X_test_cls
        y_test += [cls] * len(X_test_cls)

    if leaky:
        print(
            f"[data] WARNING: {', '.join(leaky)} only ha{'s' if len(leaky) == 1 else 've'} one "
            "capture session, so their test photos are near-duplicates of the training ones.\n"
            "[data] WARNING: the score is optimistic. Shoot more sessions "
            "(other rooms/lighting/angles) to get an honest number."
        )
    return (X_training, y_training), (X_test, y_test)


def make_ds(X, y, training):
    """(image paths, class indices) -> batched tf.data.Dataset of float32
    [0, 255] images."""

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32)  # keep [0, 255] for EfficientNetV2
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(min(len(X), 1000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH)
    if training:  # augment on batches, cheaper than per-image
        ds = ds.map(lambda x, y: (AUGMENT(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def class_weights(y, num_classes):
    """Balanced loss weights: a class with half the photos counts twice as
    much, so the model cannot win by ignoring it."""
    counts = [0] * num_classes
    for label in y:
        counts[label] += 1
    total = sum(counts)
    return {i: (total / (num_classes * c) if c else 1.0) for i, c in enumerate(counts)}


def load(samples_dir):
    """One call: scan + split + build.

    Returns (training_ds, test_ds, class_names, class_weight); test_ds is None
    when there are too few photos to spare any, class_weight is None when
    BALANCE_CLASSES is off.
    """
    files, labels, class_names = scan(samples_dir)
    print(f"[data] {len(files)} images, {len(class_names)} classes: {class_names}")
    (X_training, y_training), (X_test, y_test) = split(files, labels, class_names)
    print(f"[data] training {len(X_training)} / test {len(X_test)}")

    weight = class_weights(y_training, len(class_names)) if BALANCE_CLASSES else None
    if weight:
        print("[data] class weights: " + "  ".join(f"{n} {weight[i]:.2f}" for i, n in enumerate(class_names)))

    training_ds = make_ds(X_training, y_training, training=True)
    if not X_test:
        print("[data] too few images for a test split, training on all")
        return training_ds, None, class_names, weight
    return training_ds, make_ds(X_test, y_test, training=False), class_names, weight
