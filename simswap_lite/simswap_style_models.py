import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class IdentityEncoder(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 256):
        super().__init__()
        backbone = models.resnet18(weights=None)
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
        emb = F.normalize(self.embedding(feat), dim=1)
        logits = self.classifier(emb)
        return emb, logits


class AdaIN(nn.Module):
    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.to_style = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_style(style).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * self.norm(x) + beta


class ResidualAdaINBlock(nn.Module):
    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.adain1 = AdaIN(channels, style_dim)
        self.adain2 = AdaIN(channels, style_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.adain1(h, style)
        h = self.act(h)
        h = self.conv2(h)
        h = self.adain2(h, style)
        return self.act(h + x)


class DownBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 4, 2, 1),
            nn.InstanceNorm2d(c_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c_in, c_out, 3, 1, 1),
            nn.InstanceNorm2d(c_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimSwapLiteGenerator(nn.Module):
    """Compact SimSwap-style generator.

    Inputs:
      - target_aligned: face whose expression/pose/background we preserve
      - id_emb: source identity embedding to inject

    Output:
      - swapped aligned face in [-1, 1]
    """
    def __init__(self, style_dim: int = 256, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 7, 1, 3),
            nn.InstanceNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down1 = DownBlock(c, c * 2)
        self.down2 = DownBlock(c * 2, c * 4)
        self.down3 = DownBlock(c * 4, c * 4)

        self.res_blocks = nn.ModuleList([
            ResidualAdaINBlock(c * 4, style_dim),
            ResidualAdaINBlock(c * 4, style_dim),
            ResidualAdaINBlock(c * 4, style_dim),
            ResidualAdaINBlock(c * 4, style_dim),
        ])

        self.up1 = UpBlock(c * 4, c * 2)
        self.up2 = UpBlock(c * 2, c)
        self.up3 = UpBlock(c, c)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(c, 3, 7, 1, 3),
            nn.Tanh(),
        )

    def forward(self, target_aligned: torch.Tensor, id_emb: torch.Tensor) -> torch.Tensor:
        x = self.stem(target_aligned)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        for blk in self.res_blocks:
            x = blk(x, id_emb)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.to_rgb(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.InstanceNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1),
            nn.InstanceNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 4, 4, 1, 1),
            nn.InstanceNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg11(weights=models.VGG11_Weights.DEFAULT).features[:10]
        for p in vgg.parameters():
            p.requires_grad = False
        self.net = vgg.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected in [-1,1], convert to ImageNet-ish range
        x = (x + 1.0) / 2.0
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.net(x)
