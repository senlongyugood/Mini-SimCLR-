import argparse
import csv
import random
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data_utils import get_cifar10_dataloader
from loss import NTXentLoss
from model import SimCLR


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_pretrain_log(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "lr"])
        writer.writeheader()
        writer.writerows(rows)


def pretrain(config):
    """Run SimCLR pretraining and save checkpoints plus epoch loss logs."""
    set_seed(config.get("seed", 42))
    device_name = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Device: {device}")

    data_dir = Path(config["data_dir"])
    save_dir = Path(config["save_dir"])
    log_dir = Path(config["log_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader = get_cifar10_dataloader(
        data_dir=str(data_dir),
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 0),
        train_size=config.get("train_size"),
        download=config.get("download", True),
    )
    if len(train_loader) == 0:
        raise ValueError("No pretraining batches. Reduce batch_size or increase train_size.")

    train_images = len(train_loader.dataset)
    print(
        f"Dataset: CIFAR-10 train, images: {train_images}, "
        f"batches: {len(train_loader)}, batch_size: {config['batch_size']}"
    )

    model = SimCLR(
        encoder_name=config.get("encoder", "cnn"),
        proj_hidden_dim=config.get("proj_hidden_dim", 512),
        proj_out_dim=config.get("proj_out_dim", 128),
    ).to(device)
    print(
        f"Model: {config.get('encoder', 'cnn')}, "
        f"params: {sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config.get("lr_min", 1e-5),
    )
    criterion = NTXentLoss(temperature=config.get("temperature", 0.5))

    best_loss = float("inf")
    history = []
    log_interval = config.get("log_interval", 50)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Pretrain epoch {epoch}/{config['epochs']}")
        for step, ((x1, x2), _) in enumerate(pbar):
            x1 = x1.to(device)
            x2 = x2.to(device)
            batch_size = x1.size(0)

            z1 = model(x1)
            z2 = model(x2)
            z = torch.empty(2 * batch_size, z1.size(1), device=device)
            z[0::2] = z1
            z[1::2] = z2

            loss = criterion(z)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % log_interval == 0 and step > 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 avg=f"{total_loss / (step + 1):.4f}")
            else:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        history.append({"epoch": epoch, "loss": avg_loss, "lr": current_lr})
        write_pretrain_log(log_dir / "pretrain_log.csv", history)
        print(f"Epoch {epoch:3d}/{config['epochs']}  loss={avg_loss:.4f}  lr={current_lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = save_dir / "simclr_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config,
            }, best_path)
            print(f"Best model saved to {best_path}")

    final_path = save_dir / "simclr_final.pth"
    torch.save({
        "epoch": config["epochs"],
        "model_state_dict": model.state_dict(),
        "config": config,
    }, final_path)
    print(f"Final model saved to {final_path}")

    return model


def default_config():
    return {
        "data_dir": str(PROJECT_ROOT / "data"),
        "save_dir": str(PROJECT_ROOT / "checkpoints"),
        "log_dir": str(PROJECT_ROOT / "logs"),
        "batch_size": 128,
        "epochs": 3,
        "encoder": "cnn",
        "proj_hidden_dim": 512,
        "proj_out_dim": 128,
        "temperature": 0.5,
        "lr": 1e-3,
        "lr_min": 1e-5,
        "weight_decay": 1e-4,
        "num_workers": 0,
        "train_size": 50000,
        "log_interval": 50,
        "seed": 42,
        "download": True,
        "device": None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Mini SimCLR pretraining")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--encoder", choices=["cnn", "resnet18"], default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    config = default_config()
    for key in [
        "data_dir", "save_dir", "log_dir", "batch_size", "epochs", "encoder",
        "train_size", "lr", "temperature", "num_workers", "seed", "device",
    ]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.no_download:
        config["download"] = False
    return config


if __name__ == "__main__":
    pretrain(parse_args())
