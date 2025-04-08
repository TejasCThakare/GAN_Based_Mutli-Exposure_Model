import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class HDRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        categories = os.listdir(self.root_dir)  # Bathroom, Bear, Table, etc.

        for category in categories:
            image_path = os.path.join(self.root_dir, category, "images")  # Ensure correct folder name
            if not os.path.isdir(image_path):
                continue

            scenes = {}
            for file in os.listdir(image_path):
                if file.endswith(".png"):
                    scene_id = "_".join(file.split("_")[:-1])  # Extract scene identifier (e.g., "0" from "0_0.png")
                    if scene_id not in scenes:
                        scenes[scene_id] = []
                    scenes[scene_id].append(file)

            for scene_id, scene_files in scenes.items():
                if len(scene_files) == 5:  # Ensure all exposure levels exist
                    scene_files.sort()  # Maintain correct order (0_0, 0_1, ..., 0_4)
                    full_paths = [os.path.join(image_path, f) for f in scene_files]
                    data.append(full_paths)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        scene_files = self.data[index]
        images = [Image.open(f).convert("RGB") for f in scene_files]

        if self.transform:
            images = [self.transform(img) for img in images]

        return images[2], torch.stack(images)  # Return mid-exposure as input, all exposures as target

def get_dataloader(root_dir, batch_size=8, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = HDRDataset(root_dir, transform)

    if len(dataset) == 0:
        raise ValueError(f"No valid samples found in {root_dir}. Check dataset structure!")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
