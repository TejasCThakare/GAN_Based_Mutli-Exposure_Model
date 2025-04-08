import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from dataloader import get_dataloader
from generator import Generator
from discriminator import Discriminator


parser = argparse.ArgumentParser(description="Train GAN for HDR Multi-Exposure Prediction")
parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss functions
criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Load dataset
dataloader = get_dataloader(args.dataset, batch_size=8, num_workers=2)  # Reduced num_workers for stability

# Training loop
epochs = 100
for epoch in range(epochs):
    loop = tqdm(dataloader, leave=True, desc=f"Epoch [{epoch+1}/{epochs}]")
    
    total_loss_G = 0.0
    total_loss_D = 0.0

    for img_input, img_target in loop:
        img_input, img_target = img_input.to(device), img_target.to(device)

        # Train Generator
        optimizer_G.zero_grad()
        generated_exposures = generator(img_input)
        pred_fake = discriminator(img_input, generated_exposures)
        loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_L1 = criterion_L1(generated_exposures, img_target)
        loss_G = loss_GAN + 100 * loss_L1  # Weighted L1 loss
        loss_G.backward()
        optimizer_G.step()
        total_loss_G += loss_G.item()

        # Train Discriminator
        optimizer_D.zero_grad()
        pred_real = discriminator(img_input, img_target)
        loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
        pred_fake = discriminator(img_input, generated_exposures.detach())
        loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()
        total_loss_D += loss_D.item()

        loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())

    # Print average losses every 5 epochs
    if epoch % 5 == 0:
        avg_loss_G = total_loss_G / len(dataloader)
        avg_loss_D = total_loss_D / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss_G: {avg_loss_G:.4f}, Avg Loss_D: {avg_loss_D:.4f}")

    # Save model every 10 epochs
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch}.pth")
