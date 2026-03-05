import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LFWDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], image_size: int = 160, train: bool = True):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5) if train else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


class IdentityEncoder(nn.Module):
    """Lightweight identity encoder for LFW.

    Output:
      - normalized embedding
      - classification logits
    """
    def __init__(self, num_classes: int, embedding_dim: int = 256):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        emb = self.embedding(feat)
        emb = F.normalize(emb, dim=1)
        logits = self.classifier(emb)
        return emb, logits


@dataclass
class Metrics:
    train_loss: List[float]
    val_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]


def collect_samples(data_dir: str, min_images_per_identity: int = 5) -> Tuple[List[Tuple[str, int]], Dict[int, str]]:
    root = Path(data_dir)
    identities = []
    for person_dir in sorted(root.iterdir()):
        if person_dir.is_dir():
            images = sorted([p for p in person_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
            if len(images) >= min_images_per_identity:
                identities.append((person_dir.name, images))

    if not identities:
        raise ValueError('No valid identity folders found. Check data_dir and min_images_per_identity.')

    label_to_name = {}
    samples = []
    for label, (name, images) in enumerate(identities):
        label_to_name[label] = name
        for img in images:
            samples.append((str(img), label))
    return samples, label_to_name


def split_samples(samples: List[Tuple[str, int]], train_ratio: float = 0.8, val_ratio: float = 0.1):
    by_label: Dict[int, List[Tuple[str, int]]] = {}
    for s in samples:
        by_label.setdefault(s[1], []).append(s)

    train, val, test = [], [], []
    for label, group in by_label.items():
        random.shuffle(group)
        n = len(group)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = min(n_train, n - 2) if n >= 3 else max(1, n - 1)
        n_val = min(n_val, n - n_train - 1) if n - n_train >= 2 else 0

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    return train, val, test


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    losses = []
    preds, targets = [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        _, logits = model(images)
        loss = F.cross_entropy(logits, labels)
        losses.append(loss.item())
        preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        targets.extend(labels.cpu().numpy().tolist())
    acc = accuracy_score(targets, preds) if targets else 0.0
    return float(np.mean(losses)) if losses else 0.0, acc


def save_curves(metrics: Metrics, output_dir: Path) -> None:
    epochs = range(1, len(metrics.train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metrics.train_loss, label='train_loss')
    plt.plot(epochs, metrics.val_loss, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training / Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metrics.train_acc, label='train_acc')
    plt.plot(epochs, metrics.val_acc, label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training / Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_curve.png', dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=160)
    parser.add_argument('--min_images_per_identity', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    samples, label_to_name = collect_samples(args.data_dir, args.min_images_per_identity)
    train_samples, val_samples, test_samples = split_samples(samples)

    with open(output_dir / 'label_map.json', 'w') as f:
        json.dump(label_to_name, f, indent=2)

    train_ds = LFWDataset(train_samples, image_size=args.image_size, train=True)
    val_ds = LFWDataset(val_samples, image_size=args.image_size, train=False)
    test_ds = LFWDataset(test_samples, image_size=args.image_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = IdentityEncoder(num_classes=len(label_to_name)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    metrics = Metrics([], [], [], [])
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        train_preds, train_targets = [], []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            train_preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            train_targets.extend(labels.detach().cpu().numpy().tolist())
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_acc = accuracy_score(train_targets, train_preds) if train_targets else 0.0
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        metrics.train_loss.append(train_loss)
        metrics.val_loss.append(val_loss)
        metrics.train_acc.append(train_acc)
        metrics.val_acc.append(val_acc)

        print(f'Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

        ckpt = {
            'model_state_dict': model.state_dict(),
            'num_classes': len(label_to_name),
            'embedding_dim': 256,
            'image_size': args.image_size,
            'label_map': label_to_name,
        }
        torch.save(ckpt, output_dir / 'last_encoder.pt')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, output_dir / 'best_encoder.pt')

    test_loss, test_acc = evaluate(model, test_loader, device)
    save_curves(metrics, output_dir)

    summary = {
        'num_classes': len(label_to_name),
        'num_train_samples': len(train_samples),
        'num_val_samples': len(val_samples),
        'num_test_samples': len(test_samples),
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'metrics': asdict(metrics),
    }
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
