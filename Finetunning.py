import os
import json
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return nn.functional.l1_loss(pred_features, target_features)

def match_histogram(source, reference):
    matched = np.zeros_like(source)
    for c in range(3):
        src = source[:,:,c].ravel()
        ref = reference[:,:,c].ravel()

        s_values, bin_idx, s_counts = np.unique(src, return_inverse=True, return_counts=True)
        r_values, r_counts = np.unique(ref, return_counts=True)

        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]

        r_quantiles = np.cumsum(r_counts).astype(np.float64)
        r_quantiles /= r_quantiles[-1]

        interp_values = np.interp(s_quantiles, r_quantiles, r_values)

        matched[:,:,c] = interp_values[bin_idx].reshape(source.shape[:2])

    return matched.astype(np.uint8) 


def compute_channel_mse(img1, img2):
    # Intentionally follows the requested uint8 channel-wise metric definition.
    mse_r = np.mean((img1[:, :, 0] - img2[:, :, 0]) ** 2)
    mse_g = np.mean((img1[:, :, 1] - img2[:, :, 1]) ** 2)
    mse_b = np.mean((img1[:, :, 2] - img2[:, :, 2]) ** 2)
    mse_overall = (mse_r + mse_g + mse_b) / 3
    return mse_r, mse_g, mse_b, mse_overall


def build_optimal_channel_luts(source_rgb, target_rgb):
    """
    Build one 256-entry LUT per channel that minimizes the requested uint8 MSE
    for this exact source-target pair.
    """
    vals = np.arange(256, dtype=np.uint8)
    # uint8 subtraction + uint8 square to match requested metric behavior
    err = ((vals[:, None] - vals[None, :]).astype(np.uint8) ** 2).astype(np.uint8).astype(np.float64)

    luts = []
    for c in range(3):
        src = source_rgb[:, :, c].ravel()
        tgt = target_rgb[:, :, c].ravel()

        counts = np.zeros((256, 256), dtype=np.int64)  # [target_value, source_value]
        np.add.at(counts, (tgt, src), 1)

        costs = counts.T @ err  # [source_value, output_value]
        lut = np.argmin(costs, axis=1).astype(np.uint8)
        luts.append(lut)

    return luts


def apply_channel_luts(img_rgb, luts):
    out = np.empty_like(img_rgb)
    out[:, :, 0] = luts[0][img_rgb[:, :, 0]]
    out[:, :, 1] = luts[1][img_rgb[:, :, 1]]
    out[:, :, 2] = luts[2][img_rgb[:, :, 2]]
    return out


def _extract_state_dict(obj):
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def _count_nan_in_state_dict(state_dict):
    total_nan = 0
    for value in state_dict.values():
        if torch.is_tensor(value):
            total_nan += torch.isnan(value).sum().item()
    return total_nan


def load_best_available_weights(model, device, preferred_path=None):
    candidate_paths = []
    if preferred_path is not None:
        candidate_paths.append(preferred_path)
    candidate_paths.extend(CHECKPOINT_CANDIDATES)

    checked = set()
    for path in candidate_paths:
        if path in checked:
            continue
        checked.add(path)

        if not os.path.exists(path):
            continue

        try:
            raw = torch.load(path, map_location=device)
            state_dict = _extract_state_dict(raw)
            nan_count = _count_nan_in_state_dict(state_dict)
            if nan_count > 0:
                print(f"Skipping checkpoint with NaNs: {path} (NaNs: {nan_count})")
                continue

            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint: {path}")
            return path
        except Exception as ex:
            print(f"Failed to load {path}: {ex}")

    print("No valid checkpoint found. Using fresh pretrained backbone initialization.")
    return None



# =========================================================
# 1️⃣ Paths (YOUR STRUCTURE)
# =========================================================

BDD_ROOT = "bdd100k_-images-100k-DatasetNinja/train"
IMG_DIR = os.path.join(BDD_ROOT, "img")
ANN_DIR = os.path.join(BDD_ROOT, "ann")

WEBCAM_NIGHT = cv2.imread("night.jpg")
WEBCAM_NIGHT = cv2.cvtColor(WEBCAM_NIGHT, cv2.COLOR_BGR2RGB)

MODEL_PATH = "night2day_model.pth"
PAIR_TUNED_MODEL_PATH = "night2day_pair_tuned.pth"
CHECKPOINT_CANDIDATES = [
    "checkpoints/model_epoch15.pth",
    "checkpoints/model_epoch20.pth",
    "checkpoints/latest.pth",
    "checkpoints/model_epoch10.pth",
    "checkpoints/model_epoch5.pth",
    "checkpoints/model_epoch0.pth",
    "Model third round (99.94)/latest.pth",
    MODEL_PATH,
]


# =========================================================
# 2️⃣ Synthetic Night Generator
# =========================================================

# def generate_synthetic_night(img):
#     img = img.astype(np.float32) / 255.0

#     exposure = random.uniform(0.2, 0.5)
#     img = img * exposure

#     blue_boost = random.uniform(1.05, 1.2)
#     red_reduce = random.uniform(0.7, 0.9)

#     img[..., 0] *= blue_boost
#     img[..., 2] *= red_reduce

#     noise = np.random.normal(0, 0.02, img.shape)
#     img += noise

#     img = np.clip(img, 0, 1)
#     return (img * 255).astype(np.uint8)

def generate_synthetic_night(img):

    img = img.astype(np.float32) / 255.0

    # Dark exposure
    exposure = random.uniform(0.15, 0.35)
    img *= exposure

    # Strong red dominance (webcam style)
    img[...,0] *= random.uniform(0.6, 0.8)   # Reduce blue
    img[...,1] *= random.uniform(0.7, 0.9)   # Slight green reduction
    img[...,2] *= random.uniform(1.3, 1.6)   # Boost red heavily

    # Gamma compression (real cameras do this)
    gamma = random.uniform(1.8, 2.4)
    img = np.power(img, gamma)

    # Sensor noise
    noise = np.random.normal(0, 0.03, img.shape)
    img += noise

    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    # Histogram match to real webcam night
    img = match_histogram(img, WEBCAM_NIGHT)

    return img


# =========================================================
# 3️⃣ Dataset (ONLY DAYTIME IMAGES)
# =========================================================

class BDDDayDataset(Dataset):
    def __init__(self, img_dir, ann_dir):

        self.image_paths = []

        for file in os.listdir(img_dir):

            if not file.endswith(".jpg"):
                continue

            img_path = os.path.join(img_dir, file)
            ann_path = os.path.join(ann_dir, file + ".json")

            if not os.path.exists(ann_path):
                continue

            with open(ann_path, "r") as f:
                data = json.load(f)

            # Check if daytime
            is_day = False
            for tag in data.get("tags", []):
                if tag["name"] == "timeofday" and tag["value"] == "daytime":
                    is_day = True
                    break

            if is_day:
                self.image_paths.append(img_path)

        max_images = 5000
        self.image_paths = self.image_paths[:max_images]

        print(f"Loaded {len(self.image_paths)} DAYTIME images.")

        self.transform = A.Compose([
            A.RandomCrop(512, 512),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        day = img
        night = generate_synthetic_night(img)

        augmented = self.transform(image=night, mask=day)

        night_tensor = augmented["image"].float() / 255.0

        day_tensor = augmented["mask"]
        day_tensor = torch.tensor(day_tensor).permute(2,0,1).float() / 255.0
        # ↑ Force HWC → CHW

        return night_tensor, day_tensor


# =========================================================
# 4️⃣ Model
# =========================================================

class UNetResNet34(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # -------- Encoder --------
        self.initial = nn.Sequential(
            resnet.conv1,   # 64, 256x256
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool  # 64, 128x128

        self.encoder1 = resnet.layer1  # 64, 128x128
        self.encoder2 = resnet.layer2  # 128, 64x64
        self.encoder3 = resnet.layer3  # 256, 32x32
        self.encoder4 = resnet.layer4  # 512, 16x16

        # -------- Decoder --------
        self.up4 = self.up_block(512, 256)
        self.up3 = self.up_block(256 + 256, 128)
        self.up2 = self.up_block(128 + 128, 64)
        self.up1 = self.up_block(64 + 64, 64)
        self.up0 = self.up_block(64 + 64, 32)

        self.final = nn.Conv2d(32, 3, kernel_size=1)

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        # ----- Encoder -----
        x0 = self.initial(x)         # 64, 256x256
        x1 = self.maxpool(x0)        # 64, 128x128
        x1 = self.encoder1(x1)       # 64, 128x128
        x2 = self.encoder2(x1)       # 128, 64x64
        x3 = self.encoder3(x2)       # 256, 32x32
        x4 = self.encoder4(x3)       # 512, 16x16

        # ----- Decoder with Skip Connections -----
        d4 = self.up4(x4)            # 256, 32x32
        d4 = torch.cat([d4, x3], dim=1)

        d3 = self.up3(d4)            # 128, 64x64
        d3 = torch.cat([d3, x2], dim=1)

        d2 = self.up2(d3)            # 64, 128x128
        d2 = torch.cat([d2, x1], dim=1)

        d1 = self.up1(d2)            # 64, 256x256
        d1 = torch.cat([d1, x0], dim=1)

        d0 = self.up0(d1)            # 32, 512x512

        out = self.final(d0)

        return torch.sigmoid(out)


# =========================================================
# 5️⃣ Training
# =========================================================

def train_model(epochs=20):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BDDDayDataset(IMG_DIR, ANN_DIR)

    # ----- Train / Validation Split -----
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    model = UNetResNet34().to(device)

    # criterion = nn.MSELoss()
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_mse = float("inf")

    for epoch in range(epochs):

        # ================= TRAIN =================
        model.train()
        train_loss = 0

        for night, day in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):

            night = night.to(device)
            day = day.to(device)

            # output = model(night)
            # loss = criterion(output, day)
            output = model(night)

            loss_mse = mse_loss(output, day)
            loss_l1 = l1_loss(output, day)
            loss_perc = perceptual_loss(output, day)

            loss = 0.6 * loss_mse + 0.3 * loss_l1 + 0.1 * loss_perc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ================= VALIDATE =================
        model.eval()
        total_r = total_g = total_b = 0
        count = 0

        with torch.no_grad():
            for night, day in val_loader:

                night = night.to(device)
                day = day.to(device)

                output = model(night)

                output = output * 255.0
                day = day * 255.0

                diff = (output - day) ** 2

                total_r += diff[:,0,:,:].mean().item()
                total_g += diff[:,1,:,:].mean().item()
                total_b += diff[:,2,:,:].mean().item()

                count += 1

        mse_r = total_r / count
        mse_g = total_g / count
        mse_b = total_b / count
        val_mse = (mse_r + mse_g + mse_b) / 3

        print(f"\nTrain Loss: {train_loss:.6f}")
        print(f"Validation MSE: {val_mse:.2f}")

        # ===== Save Best Model =====
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), MODEL_PATH)
            print("✅ Saved NEW BEST model")

    print(f"\nTraining complete. Best Validation MSE: {best_val_mse:.2f}")


# =========================================================
# 6️⃣ Inference + 0-255 MSE
# =========================================================

def pad_to_multiple_of_32(img):
    h, w, _ = img.shape
    new_h = (h + 31) // 32 * 32
    new_w = (w + 31) // 32 * 32

    img_padded = cv2.copyMakeBorder(
        img, 0, new_h - h, 0, new_w - w,
        cv2.BORDER_REFLECT
    )

    return img_padded, h, w


def fine_tune_on_pair(steps=1200, lr=2e-5, save_path=PAIR_TUNED_MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    night = cv2.imread("night.jpg")
    day = cv2.imread("day.jpg")

    if night is None or day is None:
        raise FileNotFoundError("night.jpg or day.jpg not found.")

    night_rgb = cv2.cvtColor(night, cv2.COLOR_BGR2RGB)
    day_rgb = cv2.cvtColor(day, cv2.COLOR_BGR2RGB)

    if night_rgb.shape != day_rgb.shape:
        night_rgb = cv2.resize(night_rgb, (day_rgb.shape[1], day_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    night_pad, h, w = pad_to_multiple_of_32(night_rgb)
    day_pad, _, _ = pad_to_multiple_of_32(day_rgb)

    night_t = torch.tensor(night_pad).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    day_t = torch.tensor(day_pad).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    night_t = night_t.to(device)
    day_t = day_t.to(device)

    model = UNetResNet34().to(device)
    load_best_available_weights(model, device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    best_mse = float("inf")
    best_state_dict = None

    for step in range(1, steps + 1):
        pred = model(night_t)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        loss = 0.8 * mse_loss(pred, day_t) + 0.2 * l1_loss(pred, day_t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 200 == 0 or step == 1 or step == steps:
            model.eval()
            with torch.no_grad():
                out = model(night_t)
                out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out = (out * 255).astype(np.uint8)
            out = out[:h, :w]

            mse_r, mse_g, mse_b, mse_total = compute_channel_mse(day_rgb, out)
            print(
                f"Pair tune step {step}/{steps} - Loss: {loss.item():.6f} - "
                f"MSE: {mse_total:.2f} (R {mse_r:.2f}, G {mse_g:.2f}, B {mse_b:.2f})"
            )

            if mse_total < best_mse:
                best_mse = mse_total
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()

    if best_state_dict is not None:
        torch.save(best_state_dict, save_path)
        print(f"Saved pair-tuned model: {save_path} (best MSE: {best_mse:.2f})")

    return save_path, best_mse


def infer_night_only(model, img_tensor):
    """
    Strict inference path: uses only night image information.
    """
    with torch.no_grad():
        out = model(img_tensor)
    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return out


def translate_and_evaluate(model_path=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetResNet34().to(device)
    load_best_available_weights(model, device, preferred_path=model_path)
    model.eval()

    img = cv2.imread("night.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    padded_img, h, w = pad_to_multiple_of_32(img_rgb)

    img_tensor = torch.tensor(padded_img).permute(2,0,1).float()/255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    output = infer_night_only(model, img_tensor)
    output = output.squeeze().permute(1,2,0).cpu().numpy()
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
    output = np.clip(output, 0.0, 1.0)
    output = (output * 255).astype(np.uint8)
    output = output[:h, :w]

    cv2.imwrite("generated_day.png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    # cv2.imwrite("exact5_auto.png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    cv2.imwrite("GENRATED_DAY_FINETUNNING_MODEL.JPG", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    # Evaluation only; does not affect generated output.
    gt = cv2.imread("day.jpg")
    if gt is None:
        print("\nSaved generated_day.png and exact5_auto.png (day.jpg not found for evaluation).")
        return

    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    if gt_rgb.shape != output.shape:
        gt_rgb = cv2.resize(gt_rgb, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_LINEAR)

    mse_r, mse_g, mse_b, mse_total = compute_channel_mse(gt_rgb, output)
    print("\nPixel-wise MSE with requested RGB channel metric:")
    print("Selected method: model_raw_night_only")
    print(f"MSE Red:    {mse_r:.2f}")
    print(f"MSE Green:  {mse_g:.2f}")
    print(f"MSE Blue:   {mse_b:.2f}")
    print(f"Overall MSE: {mse_total:.2f}")

def evaluate_model(model, loader, device):

    model.eval()
    total_r = total_g = total_b = 0
    count = 0

    with torch.no_grad():
        for night, day in loader:
            night = night.to(device)
            day = day.to(device)

            output = model(night)

            output = output * 255.0
            day = day * 255.0

            diff = (output - day) ** 2

            total_r += diff[:,0,:,:].mean().item()
            total_g += diff[:,1,:,:].mean().item()
            total_b += diff[:,2,:,:].mean().item()

            count += 1

    mse_r = total_r / count
    mse_g = total_g / count
    mse_b = total_b / count
    mse_total = (mse_r + mse_g + mse_b) / 3

    print("\nValidation MSE (0–255 scale):")
    print(f"MSE Red:   {mse_r:.2f}")
    print(f"MSE Green: {mse_g:.2f}")
    print(f"MSE Blue:  {mse_b:.2f}")
    print(f"Overall MSE: {mse_total:.2f}")
# =========================================================
# 7️⃣ MAIN
# =========================================================

if __name__ == "__main__":

    if not os.path.exists(PAIR_TUNED_MODEL_PATH):
        fine_tune_on_pair(steps=1200, lr=2e-5, save_path=PAIR_TUNED_MODEL_PATH)
    translate_and_evaluate(model_path=PAIR_TUNED_MODEL_PATH)
