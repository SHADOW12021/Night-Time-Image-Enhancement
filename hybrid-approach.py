import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from glob import glob
import random
import matplotlib.pyplot as plt

# ----------------------------
# Classical Image Processing
# ----------------------------

def classical_preprocess(night_img):
    """Apply classical preprocessing: denoise and illumination adjustment"""
    # Denoising
    denoised = cv2.fastNlMeansDenoisingColored(night_img, None, 10, 10, 7, 21)

    # Convert to LAB for illumination correction
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    preprocessed = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return preprocessed

# ----------------------------
# Learning-Based Module
# ----------------------------

class GlobalColorCorrection(nn.Module):
    """Simple global color correction using 3x3 color matrix + bias"""
    def __init__(self):
        super().__init__()
        # Initialize as identity
        self.color_matrix = nn.Parameter(torch.eye(3))
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        # x: [B, C, H, W] float32 tensor in [0,1]
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)  # [B,3,H*W]
        y_flat = torch.matmul(self.color_matrix, x_flat) + self.bias.view(3,1)
        y = y_flat.view(B, C, H, W)
        return torch.clamp(y, 0, 1)

# ----------------------------
# Synthetic Data Generation
# ----------------------------

def generate_synthetic_night(img, num_variations=5):
    """Generate synthetic night images from a well-lit image"""
    imgs = []
    for _ in range(num_variations):
        img_copy = img.astype(np.float32) / 255.0

        # Random brightness reduction
        brightness = random.uniform(0.2, 0.7)
        img_copy *= brightness

        # Random contrast compression
        contrast = random.uniform(0.5, 1.0)
        mean = np.mean(img_copy, axis=(0,1), keepdims=True)
        img_copy = (img_copy - mean) * contrast + mean

        # Random color casting
        color_cast = np.random.uniform(0.9, 1.1, size=(1,1,3))
        img_copy *= color_cast

        # Additive Gaussian noise
        noise = np.random.normal(0, 0.02, img_copy.shape)
        img_copy += noise
        img_copy = np.clip(img_copy, 0, 1)
        imgs.append(img_copy)
    return imgs

# ----------------------------
# Training Loop
# ----------------------------

def train_correction_model(model, synthetic_day_images, device='cpu', epochs=200):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for day_img in synthetic_day_images:
            # Generate synthetic night version
            night_imgs = generate_synthetic_night((day_img*255).astype(np.uint8))
            for night_img in night_imgs:
                # Prepare tensors
                x = torch.tensor(night_img.transpose(2,0,1)[None], dtype=torch.float32, device=device)
                y = torch.tensor(day_img.transpose(2,0,1)[None], dtype=torch.float32, device=device)
                # Forward
                pred = model(x)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(synthetic_day_images):.6f}")

# ----------------------------
# Inference
# ----------------------------

def enhance_night_image(night_img, model, device='cpu'):
    preprocessed = classical_preprocess(night_img)
    x = torch.tensor(preprocessed.astype(np.float32).transpose(2,0,1)[None]/255.0, device=device)
    with torch.no_grad():
        enhanced = model(x).cpu().numpy()[0].transpose(1,2,0) * 255.0
    return np.clip(enhanced, 0, 255).astype(np.uint8)

# ----------------------------
# Evaluation
# ----------------------------

def compute_mse(img1, img2):
    """Compute channel-wise MSE"""
    mse_channels = ((img1.astype(np.float32) - img2.astype(np.float32))**2).mean(axis=(0,1))
    return mse_channels.mean()


# ----------------------------
# Visualization
# ----------------------------

def visualize_results(night_img, enhanced_img, day_img=None):
    """Display Night, Enhanced, and optionally Day images side by side"""
    images = [night_img, enhanced_img]
    titles = ['Night Image', 'Enhanced Image']

    if day_img is not None:
        images.append(day_img)
        titles.append('Reference Day Image')

    plt.figure(figsize=(15,5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ----------------------------
# Example Usage
# ----------------------------

if __name__ == "__main__":
    # Load day-time image for synthetic training
    day_img = cv2.imread("day.jpg")
    day_img = cv2.cvtColor(day_img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

    # Initialize and train model
    model = GlobalColorCorrection()
    train_correction_model(model, [day_img])

    # Load night-time image for inference
    night_img = cv2.imread("night.jpg")
    night_img_rgb = cv2.cvtColor(night_img, cv2.COLOR_BGR2RGB)

    # Enhance
    enhanced_img = enhance_night_image(night_img_rgb, model)

    # Save enhanced result
    cv2.imwrite("enhanced_result.jpg", cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))

    # Evaluate (optional)
    ref_img = (day_img*255).astype(np.uint8)
    mse = compute_mse(enhanced_img, ref_img)
    print(f"Channel-wise MSE: {mse:.4f}")

    # Visualize results
    visualize_results(night_img_rgb, enhanced_img, ref_img)