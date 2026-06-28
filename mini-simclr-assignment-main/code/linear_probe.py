import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_utils import get_cifar10_linear_dataloader
from model import SimCLR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LinearClassifier(nn.Module):
    """Frozen encoder plus one trainable linear classifier."""

    def __init__(self, encoder, feat_dim, num_classes=10):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.classifier = nn.Linear(feat_dim, num_classes)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x)
        return self.classifier(h)


def write_linear_log(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "loss", "train_acc", "test_acc"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1) * 100


def save_prediction_figure(samples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = len(samples)
    fig, axes = plt.subplots(1, cols, figsize=(2.2 * cols, 2.6))
    if cols == 1:
        axes = [axes]

    for ax, sample in zip(axes, samples):
        image = sample["image"].permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.set_title(
            f"T: {sample['true_label']}\nP: {sample['pred_label']}",
            fontsize=9,
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def collect_prediction_samples(model, loader, device, max_samples=5):
    model.eval()
    samples = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            pred = logits.argmax(dim=1).cpu()
            for i in range(x.size(0)):
                true_idx = int(y[i].item())
                pred_idx = int(pred[i].item())
                samples.append({
                    "index": len(samples),
                    "image": x[i].cpu(),
                    "true_label": CIFAR10_CLASSES[true_idx],
                    "pred_label": CIFAR10_CLASSES[pred_idx],
                    "true_class": true_idx,
                    "pred_class": pred_idx,
                })
                if len(samples) >= max_samples:
                    return samples
    return samples


def linear_probe(config):
    """Train a linear classifier on top of a frozen pretrained encoder."""
    set_seed(config.get("seed", 42))
    device_name = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Device: {device}")

    checkpoint_path = Path(config["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {})

    encoder_name = config.get("encoder", "auto")
    if encoder_name in (None, "auto"):
        encoder_name = checkpoint_config.get("encoder", "cnn")
    proj_hidden_dim = config.get("proj_hidden_dim") or checkpoint_config.get("proj_hidden_dim", 512)
    proj_out_dim = config.get("proj_out_dim") or checkpoint_config.get("proj_out_dim", 128)

    simclr = SimCLR(
        encoder_name=encoder_name,
        proj_hidden_dim=proj_hidden_dim,
        proj_out_dim=proj_out_dim,
    ).to(device)
    simclr.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Loaded checkpoint: {checkpoint_path} "
        f"(epoch={checkpoint.get('epoch')}, loss={checkpoint.get('loss', 0.0):.4f}, "
        f"encoder={encoder_name})"
    )

    model = LinearClassifier(simclr.encoder, simclr.encoder.feat_dim, num_classes=10).to(device)

    data_dir = Path(config["data_dir"])
    train_loader = get_cifar10_linear_dataloader(
        data_dir=str(data_dir),
        batch_size=config["batch_size"],
        train=True,
        num_workers=config.get("num_workers", 0),
        subset_size=config.get("train_size"),
        download=config.get("download", True),
    )
    test_loader = get_cifar10_linear_dataloader(
        data_dir=str(data_dir),
        batch_size=config["batch_size"],
        train=False,
        num_workers=config.get("num_workers", 0),
        subset_size=config.get("test_size"),
        download=config.get("download", True),
    )
    print(
        f"Linear probe data: train_images={len(train_loader.dataset)}, "
        f"test_images={len(test_loader.dataset)}, batch_size={config['batch_size']}"
    )

    optimizer = optim.SGD(
        model.classifier.parameters(),
        lr=config.get("lr", 0.1),
        momentum=0.9,
        weight_decay=config.get("weight_decay", 1e-4),
    )
    criterion = nn.CrossEntropyLoss()

    log_dir = Path(config["log_dir"])
    results_dir = Path(config["results_dir"])
    figures_dir = Path(config["figures_dir"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Linear probe epoch {epoch}/{config['epochs']}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        })
        write_linear_log(log_dir / "linear_probe_log.csv", history)

        print(
            f"Epoch {epoch:3d}/{config['epochs']}  loss={avg_loss:.4f}  "
            f"train_acc={train_acc:.2f}%  test_acc={test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "classifier_state_dict": model.classifier.state_dict(),
                "encoder": encoder_name,
                "feat_dim": simclr.encoder.feat_dim,
                "test_acc": test_acc,
                "config": config,
            }, checkpoint_dir / "linear_probe_best.pth")

    samples = collect_prediction_samples(
        model, test_loader, device, max_samples=config.get("num_prediction_samples", 5)
    )
    prediction_figure = figures_dir / "prediction_samples.png"
    save_prediction_figure(samples, prediction_figure)

    json_samples = [
        {k: v for k, v in sample.items() if k != "image"}
        for sample in samples
    ]
    result = {
        "train_images": len(train_loader.dataset),
        "pretrain_epochs": checkpoint.get("epoch"),
        "linear_probe_epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "encoder": encoder_name,
        "test_images": len(test_loader.dataset),
        "best_epoch": best_epoch,
        "best_test_accuracy": best_acc,
        "prediction_figure": str(prediction_figure),
        "predictions": json_samples,
    }
    with (results_dir / "linear_probe_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Results saved to {results_dir / 'linear_probe_results.json'}")
    print(f"Prediction samples saved to {prediction_figure}")
    return best_acc


def default_config():
    return {
        "data_dir": str(PROJECT_ROOT / "data"),
        "checkpoint_path": str(PROJECT_ROOT / "checkpoints" / "simclr_best.pth"),
        "checkpoint_dir": str(PROJECT_ROOT / "checkpoints"),
        "log_dir": str(PROJECT_ROOT / "logs"),
        "results_dir": str(PROJECT_ROOT / "results"),
        "figures_dir": str(PROJECT_ROOT / "report" / "figures"),
        "batch_size": 128,
        "epochs": 3,
        "encoder": "auto",
        "proj_hidden_dim": None,
        "proj_out_dim": None,
        "lr": 0.1,
        "weight_decay": 1e-4,
        "num_workers": 0,
        "train_size": 50000,
        "test_size": 1000,
        "seed": 42,
        "download": True,
        "device": None,
        "num_prediction_samples": 5,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Mini SimCLR linear probe")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--encoder", choices=["auto", "cnn", "resnet18"], default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-prediction-samples", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    config = default_config()
    for key in [
        "data_dir", "checkpoint_path", "checkpoint_dir", "log_dir", "results_dir",
        "figures_dir", "batch_size", "epochs", "encoder", "train_size",
        "test_size", "lr", "num_workers", "seed", "device",
        "num_prediction_samples",
    ]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.no_download:
        config["download"] = False
    return config


if __name__ == "__main__":
    linear_probe(parse_args())
