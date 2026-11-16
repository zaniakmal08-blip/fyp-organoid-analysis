"""Fit TinyUNet on every image that has a mask (boundary-weighted BCE + Dice loss).

Reads   ``<data>/images/<name>.tif`` and ``<data>/masks/<name>.png`` (0/255).
Writes  ``results/local/model_candidate/`` : tiny_unet.pt, training_metrics.json,
        training_history.csv. The selected model in ``models/`` is never replaced.

There is NO held-out split: the reported Dice is measured on the training images
and must not be described as test accuracy. By default training warm-starts from
the selected checkpoint; pass a non-existing ``--warm-start`` path to start fresh.

Example::

    python src/train_prototype.py --epochs 300 --learning-rate 0.001 --dice-weight 3
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import TinyUNet
from preprocessing import letterbox, load_tiff, normalize
from project_paths import DATA as ROOT
from project_paths import MODEL, RESULTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=0.0004)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--warm-start", type=Path, default=MODEL)
    p.add_argument("--output", type=Path, default=RESULTS / "model_candidate")
    args = p.parse_args()
    if args.epochs < 1 or args.size < 32 or args.size % 4:
        p.error("Positive epochs; size >=32 and divisible by four.")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)
    names = sorted(p.stem for p in (ROOT / "masks").glob("*.png"))
    if not names:
        raise ValueError("No annotation masks found")
    xs = []
    ys = []
    sources = []
    for name in names:
        ip = ROOT / "images" / (name + ".tif")
        mp = ROOT / "masks" / (name + ".png")
        a, _ = load_tiff(ip)
        mask = np.asarray(Image.open(mp))
        if mask.shape != a.shape or set(np.unique(mask)) != {0, 255}:
            raise ValueError("Invalid mask: " + name)
        xs.append(letterbox(normalize(a), args.size)[0])
        ys.append(letterbox(mask.astype(np.float32) / 255, args.size, True)[0])
        sources.append(
            {
                "name": name,
                "image_sha256": hashlib.sha256(ip.read_bytes()).hexdigest(),
                "mask_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
            }
        )
    x = torch.tensor(np.array(xs)[:, None])
    y = torch.tensor(np.array(ys)[:, None])
    model = TinyUNet()
    if args.warm_start.is_file():
        previous = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        if previous["preprocessing"] != "letterbox_percentile_v1":
            raise ValueError("Incompatible preprocessing")
        model.load_state_dict(previous["model_state"])
        print("Warm start:", args.warm_start, flush=True)
    # Emphasise the narrow boundary band, without using measurement CSV targets.
    dilated = torch.nn.functional.max_pool2d(y, 7, stride=1, padding=3)
    eroded = -torch.nn.functional.max_pool2d(-y, 7, stride=1, padding=3)
    weights = 1 + 7 * (dilated - eroded)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=0.00002)
    history = []

    def training_score():
        model.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(names), 2):
                prediction = model(x[start : start + 2]).sigmoid() >= 0.5
                truth = y[start : start + 2] > 0
                values.extend(
                    (
                        2
                        * (prediction & truth).sum((1, 2, 3))
                        / (prediction.sum((1, 2, 3)) + truth.sum((1, 2, 3)))
                    ).tolist()
                )
        return float(np.mean(values))

    best_score = training_score()
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    print(
        f"{len(names)}-image fitting only; no independent validation. Higher resolution + boundary-weighted loss.",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        epoch_loss = 0.0
        batches = list(range(0, len(names), 2))
        for start in batches:
            xb = x[start : start + 2]
            yb = y[start : start + 2]
            wb = weights[start : start + 2]
            logits = model(xb)
            prob = logits.sigmoid()
            dims = (1, 2, 3)
            dice = (2 * (prob * yb).sum(dims) + 1) / (prob.sum(dims) + yb.sum(dims) + 1)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = (bce * wb).sum() / wb.sum() + args.dice_weight * (1 - dice).mean()
            (loss / len(batches)).backward()
            epoch_loss += float(loss.detach()) / len(batches)
        opt.step()
        scheduler.step()
        history.append([epoch, epoch_loss])
        if epoch == 1 or epoch % 20 == 0:
            print(f"{epoch}/{args.epochs}: loss {epoch_loss:.5f}", flush=True)
        if epoch % 10 == 0 or epoch == args.epochs:
            score = training_score()
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch % 20 == 0:
                print(
                    f"Mean training Dice {score:.5f}; best {best_score:.5f} at epoch {best_epoch}",
                    flush=True,
                )
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = np.concatenate(
            [model(x[i : i + 2]).sigmoid().numpy()[:, 0] >= 0.5 for i in range(0, len(names), 2)]
        )
    scores = []
    for i, name in enumerate(names):
        truth = np.array(ys[i]) > 0
        inter = np.logical_and(pred[i], truth).sum()
        scores.append(
            {"image": name, "training_Dice": float(2 * inter / (pred[i].sum() + truth.sum()))}
        )
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "tiny_unet.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "size": args.size,
            "names": names,
            "preprocessing": "letterbox_percentile_v1",
            "sources": sources,
            "seed": 42,
            "experiment": "all_annotated_images_fitting_no_holdout",
            "boundary_loss": "7x additional weight in 7-pixel boundary band; BCE + Dice",
            "epochs": args.epochs,
            "warm_start": str(args.warm_start),
            "learning_rate": args.learning_rate,
            "dice_weight": args.dice_weight,
            "selected_epoch": best_epoch,
            "selection": "highest mean Dice on training images, NOT validation",
            "roi_representation": "polygon_1px",
        },
        destination,
    )
    (args.output / "training_metrics.json").write_text(json.dumps(scores, indent=2))
    with (args.output / "training_history.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "training_loss"])
        writer.writerows(history)
    print("Saved:", destination)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
