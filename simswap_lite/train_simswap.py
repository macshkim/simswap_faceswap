import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from simswap_style_models import (
    IdentityEncoder,
    PatchDiscriminator,
    SimSwapLiteGenerator,
    VGGPerceptual,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LFWPairDataset(Dataset):
    """Returns same-identity pairs for reconstruction-style training.

    source_img: identity provider
    target_img: pose/expression provider
    label: shared identity id
    """
    def __init__(self, groups: Dict[int, List[str]], image_size: int = 224, train: bool = True):
        self.groups = {k: v for k, v in groups.items() if len(v) >= 2}
        self.labels = sorted(self.groups.keys())
        self.train = train
        aug = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5) if train else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
        self.transform = transforms.Compose(aug)

        self.index = []
        for label, paths in self.groups.items():
            for p in paths:
                self.index.append((p, label))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        anchor_path, label = self.index[idx]
        candidates = self.groups[label]
        other_path = anchor_path
        while other_path == anchor_path:
            other_path = random.choice(candidates)
        src = Image.open(anchor_path).convert('RGB')
        tgt = Image.open(other_path).convert('RGB')
        return self.transform(src), self.transform(tgt), label


def collect_groups(data_dir: str, min_images_per_identity: int = 5) -> Tuple[Dict[int, List[str]], Dict[int, str]]:
    root = Path(data_dir)
    groups: Dict[int, List[str]] = {}
    label_to_name: Dict[int, str] = {}
    label = 0
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        images = sorted([p for p in person_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        if len(images) < min_images_per_identity:
            continue
        groups[label] = [str(p) for p in images]
        label_to_name[label] = person_dir.name
        label += 1
    if not groups:
        raise ValueError('No valid identity folders found.')
    return groups, label_to_name


def split_groups(groups: Dict[int, List[str]], train_ratio: float = 0.8, val_ratio: float = 0.1):
    train, val, test = {}, {}, {}
    for label, paths in groups.items():
        paths = paths.copy()
        random.shuffle(paths)
        n = len(paths)
        n_train = max(2, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
        train[label] = paths[:n_train]
        if n_val > 0:
            val[label] = paths[n_train:n_train + n_val]
        remainder = paths[n_train + n_val:]
        if len(remainder) >= 2:
            test[label] = remainder
        elif len(remainder) == 1:
            train[label].append(remainder[0])
    train = {k: v for k, v in train.items() if len(v) >= 2}
    val = {k: v for k, v in val.items() if len(v) >= 2}
    test = {k: v for k, v in test.items() if len(v) >= 2}
    return train, val, test


@torch.no_grad()
def load_identity_model(id_ckpt: str, device: torch.device):
    ckpt = torch.load(id_ckpt, map_location=device)
    model = IdentityEncoder(num_classes=ckpt['num_classes'], embedding_dim=ckpt.get('embedding_dim', 256))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    image_size = ckpt.get('image_size', 160)
    return model, image_size, ckpt.get('embedding_dim', 256)


def resize_for_id(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)


@dataclass
class EpochLog:
    epoch: int
    g_total: float
    g_adv: float
    g_recon: float
    g_id: float
    g_perc: float
    d_total: float


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        groups, label_to_name = collect_groups(args.data_dir, args.min_images_per_identity)
        self.train_groups, self.val_groups, self.test_groups = split_groups(groups)

        with open(self.output_dir / 'label_map.json', 'w') as f:
            json.dump(label_to_name, f, indent=2)

        self.train_loader = DataLoader(
            LFWPairDataset(self.train_groups, image_size=args.image_size, train=True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            LFWPairDataset(self.val_groups if self.val_groups else self.train_groups, image_size=args.image_size, train=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True,
            drop_last=False,
        )

        self.id_net, self.id_input_size, self.style_dim = load_identity_model(args.id_ckpt, self.device)
        self.G = SimSwapLiteGenerator(style_dim=self.style_dim, base_channels=args.base_channels).to(self.device)
        self.D = PatchDiscriminator(base_channels=args.base_channels).to(self.device)
        self.perceptual = VGGPerceptual().to(self.device)
        self.perceptual.eval()

        self.g_opt = torch.optim.Adam(self.G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
        self.d_opt = torch.optim.Adam(self.D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

        self.logs: List[EpochLog] = []

    def _identity_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x_id = resize_for_id(x, self.id_input_size)
        emb, _ = self.id_net(x_id)
        return emb

    def _d_hinge(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()

    def _g_hinge(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return -fake_logits.mean()

    def _train_one_epoch(self, epoch: int):
        self.G.train()
        self.D.train()
        g_totals, g_advs, g_recons, g_ids, g_percs, d_totals = [], [], [], [], [], []
        last_src = last_tgt = last_fake = None

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs}')
        for src, tgt, _ in pbar:
            src = src.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)

            with torch.no_grad():
                src_id = self._identity_embedding(src)

            fake = self.G(tgt, src_id)

            # D step
            self.d_opt.zero_grad(set_to_none=True)
            real_logits = self.D(tgt)
            fake_logits = self.D(fake.detach())
            d_loss = self._d_hinge(real_logits, fake_logits)
            d_loss.backward()
            self.d_opt.step()

            # G step
            self.g_opt.zero_grad(set_to_none=True)
            fake_logits = self.D(fake)
            adv_loss = self._g_hinge(fake_logits)
            recon_loss = F.l1_loss(fake, tgt)

            fake_id = self._identity_embedding(fake)
            id_loss = 1.0 - F.cosine_similarity(fake_id, src_id, dim=1).mean()

            perc_loss = F.l1_loss(self.perceptual(fake), self.perceptual(tgt))

            g_loss = (
                self.args.lambda_adv * adv_loss +
                self.args.lambda_recon * recon_loss +
                self.args.lambda_id * id_loss +
                self.args.lambda_perc * perc_loss
            )
            g_loss.backward()
            self.g_opt.step()

            g_totals.append(float(g_loss.item()))
            g_advs.append(float(adv_loss.item()))
            g_recons.append(float(recon_loss.item()))
            g_ids.append(float(id_loss.item()))
            g_percs.append(float(perc_loss.item()))
            d_totals.append(float(d_loss.item()))
            last_src, last_tgt, last_fake = src.detach(), tgt.detach(), fake.detach()

            pbar.set_postfix(
                g=f'{g_loss.item():.3f}', d=f'{d_loss.item():.3f}',
                rec=f'{recon_loss.item():.3f}', id=f'{id_loss.item():.3f}'
            )

        return (
            float(np.mean(g_totals)),
            float(np.mean(g_advs)),
            float(np.mean(g_recons)),
            float(np.mean(g_ids)),
            float(np.mean(g_percs)),
            float(np.mean(d_totals)),
            last_src, last_tgt, last_fake,
        )

    @torch.no_grad()
    def evaluate(self):
        self.G.eval()
        recon_losses, id_losses = [], []
        for src, tgt, _ in self.val_loader:
            src = src.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)
            src_id = self._identity_embedding(src)
            fake = self.G(tgt, src_id)
            recon_losses.append(float(F.l1_loss(fake, tgt).item()))
            fake_id = self._identity_embedding(fake)
            id_losses.append(float((1.0 - F.cosine_similarity(fake_id, src_id, dim=1).mean()).item()))
        return {
            'val_recon_l1': float(np.mean(recon_losses)) if recon_losses else 0.0,
            'val_id_loss': float(np.mean(id_losses)) if id_losses else 0.0,
        }

    def _save_preview(self, src: torch.Tensor, tgt: torch.Tensor, fake: torch.Tensor, epoch: int) -> None:
        if src is None or tgt is None or fake is None:
            return

        grid = torch.cat([src[:4], tgt[:4], fake[:4]], dim=0)
        grid = (grid.clamp(-1, 1) + 1.0) / 2.0
        vutils.save_image(grid, self.output_dir / f'preview_epoch_{epoch:03d}.png', nrow=4)

    def _save_plots(self) -> None:
        epochs = [x.epoch for x in self.logs]
        g_total = [x.g_total for x in self.logs]
        g_recon = [x.g_recon for x in self.logs]
        g_id = [x.g_id for x in self.logs]
        d_total = [x.d_total for x in self.logs]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, g_total, label='g_total')
        plt.plot(epochs, d_total, label='d_total')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Generator vs Discriminator Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gan_loss_curve.png', dpi=160)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, g_recon, label='reconstruction_l1')
        plt.plot(epochs, g_id, label='identity_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Reconstruction and Identity Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'component_loss_curve.png', dpi=160)
        plt.close()

    def train(self):
        best_score = float('inf')
        for epoch in range(1, self.args.epochs + 1):
            g_total, g_adv, g_recon, g_id, g_perc, d_total, last_src, last_tgt, last_fake = self._train_one_epoch(epoch)
            val_metrics = self.evaluate()

            row = EpochLog(
                epoch=epoch,
                g_total=g_total,
                g_adv=g_adv,
                g_recon=g_recon,
                g_id=g_id,
                g_perc=g_perc,
                d_total=d_total,
            )
            self.logs.append(row)
            self._save_preview(last_src, last_tgt, last_fake, epoch)

            summary = asdict(row)
            summary.update(val_metrics)
            with open(self.output_dir / 'metrics.jsonl', 'a') as f:
                f.write(json.dumps(summary) + '\n')

            score = val_metrics['val_recon_l1'] + self.args.val_id_weight * val_metrics['val_id_loss']
            ckpt = {
                'epoch': epoch,
                'generator_state_dict': self.G.state_dict(),
                'discriminator_state_dict': self.D.state_dict(),
                'g_optimizer_state_dict': self.g_opt.state_dict(),
                'd_optimizer_state_dict': self.d_opt.state_dict(),
                'style_dim': self.style_dim,
                'image_size': self.args.image_size,
                'base_channels': self.args.base_channels,
            }
            torch.save(ckpt, self.output_dir / 'last_simswap_lite.pt')
            if score < best_score:
                best_score = score
                torch.save(ckpt, self.output_dir / 'best_simswap_lite.pt')

        self._save_plots()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--id_ckpt', required=True, help='Checkpoint from train_faceswap.py')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--min_images_per_identity', type=int, default=5)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--lambda_adv', type=float, default=1.0)
    parser.add_argument('--lambda_recon', type=float, default=10.0)
    parser.add_argument('--lambda_id', type=float, default=8.0)
    parser.add_argument('--lambda_perc', type=float, default=2.0)
    parser.add_argument('--val_id_weight', type=float, default=2.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
