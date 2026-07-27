"""PyTorch alternative to train.py — uses the GPU natively on Windows.

Same idea as the TensorFlow script, but PyTorch + timm so an NVIDIA GPU (e.g.
RTX 4080) works on native Windows with no WSL:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install timm pillow

Transfer-learns a head on a frozen ImageNet EfficientNetV2-B0 (timm:
"tf_efficientnetv2_b0") over samples/<label>/ folders.

Classes = every object folder found, PLUS a trailing "other" neuron:

    output neurons = (number of object labels) + 1        # last one = "other"

Images in an `other/` or `unlabeled/` folder feed that last neuron.

Outputs:
    model/model.pt     dict: {state_dict, classes, model_name, img_size}
    model/labels.txt   class names, one per line, in output order

Note: this saves a .pt, not a .h5. The current app.py loads .h5 (TensorFlow);
to serve a .pt you'd need a torch loader in /analyse (ask and I'll add it).

Quick smoke test (default 1 epoch):
    python train_torch.py
Real run:
    python train_torch.py --epochs 30
"""

import argparse
import os

import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset

MODEL_NAME = "tf_efficientnetv2_b0"
OTHER_ALIASES = {"other", "unlabeled"}
IMG_EXT = (".jpg", ".jpeg", ".png")


def collect(samples_dir):
    """Return (files, labels, class_names); class_names ends with "other"."""
    if not os.path.isdir(samples_dir):
        raise SystemExit(f"no samples dir: {samples_dir}")

    subdirs = sorted(
        d for d in os.listdir(samples_dir) if os.path.isdir(os.path.join(samples_dir, d))
    )
    object_labels = [d for d in subdirs if d.lower() not in OTHER_ALIASES]
    class_names = object_labels + ["other"]  # the +1 neuron, always last
    other_idx = len(class_names) - 1
    index_of = {name: i for i, name in enumerate(object_labels)}

    files, labels = [], []
    for d in subdirs:
        idx = index_of.get(d, other_idx)  # unknown/alias folders -> other
        folder = os.path.join(samples_dir, d)
        for fn in os.listdir(folder):
            if fn.lower().endswith(IMG_EXT):
                files.append(os.path.join(folder, fn))
                labels.append(idx)

    if not files:
        raise SystemExit(f"no images found under {samples_dir}")
    return files, labels, class_names


class ImageList(Dataset):
    def __init__(self, files, labels, transform):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="samples")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join("model", "model.pt"))
    ap.add_argument("--labels-out", default=os.path.join("model", "labels.txt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    files, labels, class_names = collect(args.samples)
    print(f"[data] {len(files)} images, {len(class_names)} classes: {class_names}")

    # Frozen backbone + fresh N+1 head.
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(class_names))
    for p in model.parameters():
        p.requires_grad = False
    head = model.get_classifier()
    for p in head.parameters():
        p.requires_grad = True
    model.to(device)

    # timm-correct normalization/size; strong augmentation on train for the
    # domain-shift problem (rotation, color jitter, random crop).
    cfg = resolve_model_data_config(model)
    img_size = cfg["input_size"][-1]
    train_tf = create_transform(**{**cfg, "is_training": True, "auto_augment": "rand-m7-n2"})
    eval_tf = create_transform(**{**cfg, "is_training": False})

    # Deterministic shuffle + split.
    g = torch.Generator().manual_seed(42)
    order = torch.randperm(len(files), generator=g).tolist()
    files = [files[i] for i in order]
    labels = [labels[i] for i in order]
    n_val = int(len(files) * args.val_split)

    val_loader = None
    if n_val >= len(class_names) and len(files) - n_val > 0:
        train_set = ImageList(files[n_val:], labels[n_val:], train_tf)
        val_set = ImageList(files[:n_val], labels[:n_val], eval_tf)
        val_loader = DataLoader(val_set, batch_size=args.batch)
    else:
        print("[data] too few images for a validation split, training on all")
        train_set = ImageList(files, labels, train_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for e in range(args.epochs):
        tl, ta = run_epoch(model, train_loader, device, criterion, optimizer)
        msg = f"epoch {e + 1}/{args.epochs}  train_loss {tl:.4f} acc {ta:.3f}"
        if val_loader is not None:
            vl, va = run_epoch(model, val_loader, device, criterion)
            msg += f"  val_loss {vl:.4f} acc {va:.3f}"
        print(msg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": class_names,
            "model_name": MODEL_NAME,
            "img_size": img_size,
        },
        args.out,
    )
    with open(args.labels_out, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")

    print(f"[done] saved {args.out} and {args.labels_out}")


if __name__ == "__main__":
    main()
