import torch
import torch.nn as nn


class AuxVAE(nn.Module):
    def __init__(self, in_features, latent_dim=128, image_size=224, base_channels=256):
        super(AuxVAE, self).__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.base_channels = base_channels

        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, base_channels * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, features):
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        z = self.reparameterize(mu, logvar)
        x = self.fc_decode(z)
        x = x.view(-1, self.base_channels, 7, 7)
        x = self.decoder(x)
        if x.size(-1) != self.image_size:
            x = nn.functional.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return mu, logvar, x
