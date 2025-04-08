import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=18):  # (3 LDR input + 15 generated exposures)
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img_input, img_generated):
        img_concat = torch.cat((img_input, img_generated.view(img_generated.size(0), -1, img_generated.size(3), img_generated.size(4))), 1)
        return self.model(img_concat)
