import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import random
from pathlib import Path

class BDDNightDayDataset(Dataset):
    """
    PyTorch Dataset for BDD100K night -> day translation.

    Expects two .txt files:
        - night_paths.txt
        - day_paths.txt

    Each line contains full or relative path to an image.

    Returns:
        night_img: tensor [-1,1]
        day_img: tensor [-1,1]
    """
    def __init__(self, night_list_file, day_list_file, img_size=256):
        # Read paths from split files
        self.night_paths = self._read_list(night_list_file)
        self.day_paths = self._read_list(day_list_file)
        self.img_size = img_size

        # Transformations
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  # [-1,1]
        ])

    def _read_list(self, file_path):
        file_path = Path(file_path)
        with open(file_path, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]
        return paths

    def __len__(self):
        # Return max so that DataLoader can loop over both equally
        return max(len(self.night_paths), len(self.day_paths))

    def __getitem__(self, idx):
        # Cycle through night images
        night_path = self.night_paths[idx % len(self.night_paths)]
        # Randomly sample day image (unpaired)
        day_path = random.choice(self.day_paths)

        night_img = Image.open(night_path).convert("RGB")
        day_img = Image.open(day_path).convert("RGB")

        return self.transform(night_img), self.transform(day_img)
    

