import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        filt = torch.stack([ll, lh, hl, hh], dim=0)  # 4x2x2
        filt = filt.unsqueeze(1)  # 4x1x2x2
        self.register_buffer("filt", filt)
        self.in_channels = in_channels

    def forward(self, x):
        # depthwise conv to compute 4 subbands per channel
        weight = self.filt.repeat(self.in_channels, 1, 1, 1)
        x = F.conv2d(x, weight, stride=2, groups=self.in_channels)
        # shape: (B, 4*C, H/2, W/2)
        return x


class MultiLevelDWT(nn.Module):
    """
    Multi-level Haar DWT that returns a list of level feature maps.
    Each level returns 4*C channels: [LL, LH, HL, HH] for each input channel.
    """
    def __init__(self, in_channels=3, levels=3):
        super().__init__()
        self.in_channels = in_channels
        self.levels = levels
        self.dwt = HaarDWT(in_channels=in_channels)

    def forward(self, x):
        levels = []
        cur = x
        for _ in range(self.levels):
            y = self.dwt(cur)
            b, c4, h, w = y.size()
            y = y.view(b, self.in_channels, 4, h, w)
            # Re-pack back to (B, 4*C, H, W) for this level
            level = y.view(b, self.in_channels * 4, h, w)
            levels.append(level)
            # next level operates on LL band
            cur = y[:, :, 0, :, :]
        return levels


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetAggregation(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.down = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.enc2 = ConvBlock(base_channels * 2, base_channels * 2)
        self.up = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec = ConvBlock(base_channels * 2, base_channels)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        u = self.up(e2)
        if u.size(-1) != e1.size(-1):
            u = F.interpolate(u, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([u, e1], dim=1)
        return self.dec(x)


class LiteVAEEncoderFull(nn.Module):
    """
    LiteVAE encoder following the diagram: multi-level DWT -> per-level feature extraction
    -> U-Net aggregation -> latent (mu, logvar).
    """
    def __init__(self, in_channels=3, base_channels=64, latent_dim=128, dwt_levels=3):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.dwt_levels = dwt_levels

        self.dwt = MultiLevelDWT(in_channels=in_channels, levels=dwt_levels)
        self.level_extractors = nn.ModuleList([
            ConvBlock(in_channels * 4, base_channels) for _ in range(dwt_levels)
        ])
        self.agg = UNetAggregation(base_channels * dwt_levels, base_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(base_channels, latent_dim)
        self.fc_logvar = nn.Linear(base_channels, latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        levels = self.dwt(x)
        feats = []
        for lvl, extractor in zip(levels, self.level_extractors):
            feats.append(extractor(lvl))
        target_size = feats[0].shape[-2:]
        feats = [
            f if f.shape[-2:] == target_size
            else F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            for f in feats
        ]
        fused = torch.cat(feats, dim=1)
        agg = self.agg(fused)
        pooled = self.pool(agg).view(agg.size(0), -1)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z


class LiteVAEEncoderSimple(nn.Module):
    """
    A simplified LiteVAE encoder (fallback).
    """
    def __init__(self, in_channels=3, base_channels=64, latent_dim=128, dwt_levels=1):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.dwt_levels = dwt_levels

        self.dwt = HaarDWT(in_channels=in_channels)
        feat_in = in_channels * (4 ** dwt_levels)
        self.encoder = nn.Sequential(
            nn.Conv2d(feat_in, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        enc_out = base_channels * 4
        self.fc_mu = nn.Linear(enc_out, latent_dim)
        self.fc_logvar = nn.Linear(enc_out, latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        for _ in range(self.dwt_levels):
            x = self.dwt(x)
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z


class LiteDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, base_channels=64, image_size=224):
        super().__init__()
        self.image_size = image_size
        self.base_channels = base_channels
        self.fc_decode = nn.Linear(latent_dim, base_channels * 4 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        y = self.fc_decode(z).view(-1, self.base_channels * 4, 7, 7)
        y = self.decoder(y)
        if y.size(-1) != self.image_size:
            y = F.interpolate(y, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return y


class LiteVAE(nn.Module):
    """
    LiteVAE wrapper with configurable encoder variant.
    """
    def __init__(
        self,
        image_size=224,
        in_channels=3,
        base_channels=64,
        latent_dim=128,
        dwt_levels=3,
        variant="full",
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.dwt_levels = dwt_levels
        self.variant = variant

        if variant == "full":
            self.encoder = LiteVAEEncoderFull(
                in_channels=in_channels,
                base_channels=base_channels,
                latent_dim=latent_dim,
                dwt_levels=dwt_levels,
            )
        else:
            self.encoder = LiteVAEEncoderSimple(
                in_channels=in_channels,
                base_channels=base_channels,
                latent_dim=latent_dim,
                dwt_levels=dwt_levels,
            )

        self.decoder = LiteDecoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            base_channels=base_channels,
            image_size=image_size,
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar, z = self.encode(x)
        recon = self.decode(z)
        return mu, logvar, z, recon
