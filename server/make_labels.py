"""Write model/labels.txt from the dataset folder the model was trained on.

The server maps output index -> class name through labels.txt, so its order
must be exactly the order Keras assigned at training time. With

    tf.keras.utils.image_dataset_from_directory(dir, labels="inferred", ...)

that order is NOT the order you created the folders in and NOT "other last":
Keras' index_directory() takes sorted(os.listdir(dir)), keeps directories,
skips dotted ones, and that list is class_names. Plain alphabetical. This
script reproduces it, so it works on any dataset folder without importing
whatever train script built the model.

Usage:
    python make_labels.py --dataset path/to/datasets/objects100
    python make_labels.py --dataset ... --model model/model.keras   # cross-check
    python make_labels.py --dataset ... --dry-run

Getting this wrong is silent: the server keeps answering 200 with confident,
wrong labels. Hence the --model check, which compares the class count against
the model's output units.
"""

import argparse
import os
import sys


def infer_class_names(dataset_dir):
    """class_names as Keras' index_directory() computes them."""
    if not os.path.isdir(dataset_dir):
        sys.exit(f"[error] not a directory: {dataset_dir}")

    names = [
        entry
        for entry in sorted(os.listdir(dataset_dir))
        if os.path.isdir(os.path.join(dataset_dir, entry)) and not entry.startswith(".")
    ]
    if not names:
        sys.exit(f"[error] no class subfolders in {dataset_dir}")
    return names


def count_images(class_dir):
    """Rough per-class image count, for the printed summary only."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    total = 0
    for root, _dirs, files in os.walk(class_dir):
        total += sum(1 for f in files if f.lower().endswith(exts))
    return total


def model_output_units(model_path):
    """Output-unit count of a .keras/.h5 model, or None if it can't be read."""
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:  # unreadable model shouldn't kill the label write
        print(f"[warn] could not load {model_path}: {exc}")
        return None
    shape = model.output_shape
    return int(shape[-1]) if isinstance(shape, tuple) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, help="dataset root, one subfolder per class")
    ap.add_argument("--out", default=os.path.join("model", "labels.txt"))
    ap.add_argument("--model", help="model file to cross-check the class count against")
    ap.add_argument("--dry-run", action="store_true", help="print, don't write")
    args = ap.parse_args()

    names = infer_class_names(args.dataset)

    print(f"[dataset] {args.dataset}")
    for idx, name in enumerate(names):
        print(f"  {idx:>3}  {name}  ({count_images(os.path.join(args.dataset, name))} images)")

    if args.model:
        units = model_output_units(args.model)
        if units is None:
            pass
        elif units != len(names):
            sys.exit(
                f"[error] {args.model} has {units} output units but the dataset has "
                f"{len(names)} classes. Wrong dataset folder for this model; "
                f"labels.txt not written."
            )
        else:
            print(f"[check] {args.model}: {units} output units == {len(names)} classes")

    if args.dry_run:
        print("[dry-run] nothing written")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # newline="\n": the server strips \r anyway, but keep the file clean.
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(names) + "\n")
    print(f"[done] wrote {len(names)} labels to {args.out}")


if __name__ == "__main__":
    main()
