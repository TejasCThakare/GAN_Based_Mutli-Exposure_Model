import argparse
import torch
from generator import Generator
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description="Run Inference on Single LDR Image")
parser.add_argument("--image", type=str, required=True, help="Path to input LDR image")
parser.add_argument("--model", type=str, default="generator.pth", help="Path to trained model")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
# generator.load_state_dict(torch.load(args.model))
generator.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

generator.eval()

# Load input image
img = Image.open(args.image).convert("RGB")
img = ToTensor()(img).unsqueeze(0).to(device)

# Generate multi-exposure images
with torch.no_grad():
    output = generator(img)

# Save outputs as separate images
for i in range(5):
    save_image(output[:, i, :, :, :], f"output_{i}.png")

print("Generated multi-exposure images saved as output_0.png to output_4.png")
