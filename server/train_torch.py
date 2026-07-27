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


def freeze_untrained_bn(model):
    """Put BatchNorm layers that are frozen into eval mode so their running
    stats don't drift on small datasets (standard fine-tuning practice)."""
    import torch.nn as nn

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            params = list(m.parameters())
            if params and not any(p.requires_grad for p in params):
                m.eval()


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    if train:
        freeze_untrained_bn(model)
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


def freeze_all_but_head(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def unfreeze_top(model, n_blocks):
    """Unfreeze the head + final conv/bn + the last n_blocks backbone stages."""
    freeze_all_but_head(model)
    for name in ("conv_head", "bn2"):  # timm EfficientNet head-side layers
        mod = getattr(model, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True
    blocks = getattr(model, "blocks", None)
    if blocks is not None and n_blocks > 0:
        for stage in list(blocks)[-n_blocks:]:
            for p in stage.parameters():
                p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[finetune] trainable params {n_train:,} / {n_total:,}")


def fit(model, train_loader, val_loader, device, lr, epochs, patience, tag, best):
    """Train `epochs`, track best val acc into `best` dict, early-stop on
    `patience` epochs without val improvement. Returns nothing; mutates `best`."""
    import copy

    criterion = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    stale = 0
    for e in range(epochs):
        tl, ta = run_epoch(model, train_loader, device, criterion, optimizer)
        line = f"[{tag}] epoch {e + 1}/{epochs}  train_loss {tl:.4f} acc {ta:.3f}"
        if val_loader is not None:
            vl, va = run_epoch(model, val_loader, device, criterion)
            line += f"  val_loss {vl:.4f} acc {va:.3f}"
            if va > best["val_acc"]:
                best["val_acc"] = va
                best["state"] = copy.deepcopy(model.state_dict())
                stale = 0
                line += "  *best"
            else:
                stale += 1
        else:
            best["state"] = copy.deepcopy(model.state_dict())  # no val: keep last
        print(line)
        if patience and val_loader is not None and stale >= patience:
            print(f"[{tag}] early stop (no val gain in {patience} epochs)")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="samples")
    ap.add_argument("--epochs", type=int, default=1, help="fine-tune epochs (or head epochs if --finetune off)")
    ap.add_argument("--warmup-epochs", type=int, default=3, help="head-only epochs before unfreezing")
    ap.add_argument("--finetune", action="store_true", help="unfreeze top blocks after warmup")
    ap.add_argument("--unfreeze", type=int, default=2, help="how many top backbone stages to unfreeze")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3, help="head/warmup LR")
    ap.add_argument("--ft-lr", type=float, default=1e-4, help="fine-tune LR (lower!)")
    ap.add_argument("--patience", type=int, default=6, help="early-stop patience (0=off)")
    ap.add_argument("--out", default=os.path.join("model", "model.pt"))
    ap.add_argument("--labels-out", default=os.path.join("model", "labels.txt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    files, labels, class_names = collect(args.samples)
    print(f"[data] {len(files)} images, {len(class_names)} classes: {class_names}")

    # Frozen backbone + fresh N+1 head.
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(class_names))
    freeze_all_but_head(model)
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

    best = {"val_acc": -1.0, "state": None}

    # Phase 1: warm up the fresh head on the frozen backbone.
    warmup = args.warmup_epochs if args.finetune else args.epochs
    if warmup > 0:
        fit(model, train_loader, val_loader, device, args.lr, warmup, args.patience, "warmup", best)

    # Phase 2: unfreeze top blocks and fine-tune at a lower LR.
    if args.finetune:
        unfreeze_top(model, args.unfreeze)
        fit(model, train_loader, val_loader, device, args.ft_lr, args.epochs, args.patience, "finetune", best)

    # Restore the best-val weights before saving.
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    if val_loader is not None:
        print(f"[best] val acc {best['val_acc']:.3f}")

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
