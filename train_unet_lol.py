import os
import cv2
import kagglehub
import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ============================================================
# 1. DOWNLOAD DATASET TO SPECIFIED PATH
# ============================================================

print("CUDA Available:", torch.cuda.is_available())

base_path = r"C:\Users\gsocc\Documents\DigitalImageProcessing\midterm\Night-Time-Image-Enhancement"

os.makedirs(base_path, exist_ok=True)

path = kagglehub.dataset_download("soumikrakshit/lol-dataset")
print("Original download path:", path)

# ============================================================
# 2. DEVICE (CUDA)
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 3. DATASET CLASS
# ============================================================

class LOLDataset(Dataset):
    def __init__(self, root_dir):
        self.low_dir = os.path.join(root_dir, "lol_dataset", "our485", "low")
        self.high_dir = os.path.join(root_dir, "lol_dataset", "our485", "high")

        self.images = sorted(os.listdir(self.low_dir))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_dir, self.images[idx])
        high_path = os.path.join(self.high_dir, self.images[idx])

        low = cv2.imread(low_path)
        high = cv2.imread(high_path)

        low = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
        high = cv2.cvtColor(high, cv2.COLOR_BGR2RGB)

        low = self.transform(low)
        high = self.transform(high)

        return low, high

# ============================================================
# 4. U-NET MODEL
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final(d1))


# ============================================================
# 5. CHANNEL-WISE MSE FUNCTION
# ============================================================

def channelwise_mse(pred, target):
    mse_r = torch.mean((pred[:, 0, :, :] - target[:, 0, :, :]) ** 2)
    mse_g = torch.mean((pred[:, 1, :, :] - target[:, 1, :, :]) ** 2)
    mse_b = torch.mean((pred[:, 2, :, :] - target[:, 2, :, :]) ** 2)

    final_mse = (mse_r + mse_g + mse_b) / 3.0
    return final_mse, mse_r, mse_g, mse_b


# ============================================================
# 6. TRAINING LOOP
# ============================================================

def train():

    dataset = LOLDataset(path)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        loop = tqdm(loader)

        for low, high in loop:
            low = low.to(device)
            high = high.to(device)

            optimizer.zero_grad()
            output = model(low)

            loss, r, g, b = channelwise_mse(output, high)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(MSE=loss.item())

        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(loader)}")

    torch.save(model.state_dict(), "unet_lol.pth")
    print("Model saved.")


# ============================================================
# 8. INFERENCE + MSE EVALUATION ON night.jpg
# ============================================================

def evaluate_on_custom_image():
    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_lol.pth", map_location=device))
    model.eval()

    transform = transforms.ToTensor()

    # Load night and day images
    night_path = os.path.join(base_path, "night.jpg")
    day_path = os.path.join(base_path, "day.jpg")

    night_img = cv2.imread(night_path)
    day_img   = cv2.imread(day_path)

    if night_img is None or day_img is None:
        raise ValueError("night.jpg or day.jpg not found in base directory.")

    # Save original size
    orig_h, orig_w = night_img.shape[:2]

    # Resize to multiples of 8 for U-Net
    new_h = math.ceil(orig_h / 8) * 8
    new_w = math.ceil(orig_w / 8) * 8
    night_img_resized = cv2.resize(night_img, (new_w, new_h))
    day_img_resized   = cv2.resize(day_img, (new_w, new_h))

    # Convert BGR -> RGB and tensor
    night_tensor = transform(cv2.cvtColor(night_img_resized, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    day_tensor   = transform(cv2.cvtColor(day_img_resized, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(night_tensor)

    # Resize output back to original night image size
    output_resized = torch.nn.functional.interpolate(
        output, size=(orig_h, orig_w), mode='bilinear', align_corners=False
    )

    # Convert to numpy image for saving
    output_img = output_resized.squeeze(0).cpu().permute(1, 2, 0).numpy()
    output_img = (output_img * 255).clip(0, 255).astype("uint8")

    save_path = os.path.join(base_path, "generated_day.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(f"Generated image saved at: {save_path}")

    # Resize day_tensor to match original size for MSE computation
    day_tensor_resized = torch.nn.functional.interpolate(
        day_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=False
    )

    # Compute channel-wise MSE
    mse_total, mse_r, mse_g, mse_b = channelwise_mse(output_resized, day_tensor_resized)

    print("\n===== RGB Channel-wise MSE =====")
    print(f"MSE_R: {mse_r.item():.6f}")
    print(f"MSE_G: {mse_g.item():.6f}")
    print(f"MSE_B: {mse_b.item():.6f}")
    print("--------------------------------")
    print(f"Final Averaged MSE: {mse_total.item():.6f}")
    print("================================\n")


# ============================================================
# 9. RUN FULL PIPELINE
# ============================================================

if __name__ == "__main__":
    # train()
    evaluate_on_custom_image()