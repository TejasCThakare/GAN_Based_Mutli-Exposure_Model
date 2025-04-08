import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=15):  # 5 exposure levels * 3 channels
        super(Generator, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(x.size(0), 5, 3, x.size(2), x.size(3))  # (Batch, 5 exposures, Channels, H, W)
