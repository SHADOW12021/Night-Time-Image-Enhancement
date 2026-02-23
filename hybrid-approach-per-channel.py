import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

# ----------------------------
# Classical Image Processing
# ----------------------------

def classical_preprocess(night_img):
    """
    Preprocessing: initial denoising and LAB conversion
    """
    # 1. Initial denoising
    denoised = cv2.fastNlMeansDenoisingColored(night_img, None, 10, 10, 7, 21)
    
    # 2. Convert to LAB
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 3. Apply CLAHE to the whole L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)

    # 4. Blend: only brighten dark/medium pixels, leave bright areas untouched
    mask = l < 200
    l_eq = l.copy()
    l_eq[mask] = l_clahe[mask]
    
    lab_eq = cv2.merge((l_eq, a, b))
    preprocessed = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return preprocessed


# ----------------------------
# Learning-Based Module
# ----------------------------

class ColorRatioCorrection(nn.Module):
    """
    Predict per-channel gain based on luminance
    This avoids global overexposure while restoring RGB
    """
    def __init__(self):
        super().__init__()
        # Simple 3-layer network for gain prediction per channel
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid()  # output gain in [0,1]
        )

    def forward(self, L_channel):
        # L_channel: [B,1,H,W] normalized [0,1]
        B, _, H, W = L_channel.shape
        x = L_channel.view(B, -1, 1)  # [B, H*W,1]
        gains = self.net(x)  # [B,H*W,3]
        gains = gains.view(B, 3, H, W)
        return gains

def apply_color_ratio(img_lab, gains):
    """
    Apply predicted gains to LAB image to restore RGB
    """
    lab = img_lab.astype(np.float32)
    L = lab[:,:,0] / 255.0
    a = lab[:,:,1]
    b = lab[:,:,2]
    # Apply per-channel gain to Luminance
    R = np.clip(L * gains[0] * 255.0, 0, 255)
    G = np.clip(L * gains[1] * 255.0, 0, 255)
    B = np.clip(L * gains[2] * 255.0, 0, 255)
    rgb = np.stack([R,G,B], axis=2).astype(np.uint8)
    return rgb

# ----------------------------
# Synthetic Data Generation
# ----------------------------

def generate_synthetic_night(img, num_variations=5):
    """
    Generate synthetic night images from well-lit images
    Adds random brightness reduction, contrast compression, color casting, and noise
    """
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

        # Random color cast
        color_cast = np.random.uniform(0.9,1.1, size=(1,1,3))
        img_copy *= color_cast

        # Add Gaussian noise
        noise = np.random.normal(0,0.02,img_copy.shape)
        img_copy += noise
        img_copy = np.clip(img_copy,0,1)
        imgs.append(img_copy)
    return imgs

# ----------------------------
# Training
# ----------------------------

def train_color_ratio_model(model, day_images, device='cpu', epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for day_img in day_images:
            night_variants = generate_synthetic_night((day_img*255).astype(np.uint8))
            for night_img in night_variants:
                # Convert to LAB
                night_lab = cv2.cvtColor(night_img, cv2.COLOR_RGB2LAB)
                day_lab = cv2.cvtColor((day_img*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
                
                # Prepare L channel tensor
                L_night = torch.tensor(night_lab[:,:,0][None,None]/255.0, dtype=torch.float32, device=device)
                L_day = torch.tensor(day_lab[:,:,0][None,None]/255.0, dtype=torch.float32, device=device)

                gains_pred = model(L_night)
                # Reconstruct predicted RGB from predicted gains
                pred_rgb = L_night * gains_pred  # simplified for training
                loss = criterion(pred_rgb, L_day.repeat(1,3,1,1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(day_images):.6f}")

# ----------------------------
# Inference
# ----------------------------

def enhance_night_image(night_img, model, device='cpu'):
    """
    Full pipeline: preprocess, predict gains, apply, denoise post-enhancement
    """
    # Classical preprocessing
    preprocessed = classical_preprocess(night_img)
    lab = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:,:,0]/255.0
    L_tensor = torch.tensor(L[None,None,:,:], dtype=torch.float32, device=device)

    with torch.no_grad():
        gains = model(L_tensor).cpu().numpy()[0]  # [3,H,W]

    enhanced = apply_color_ratio(lab, gains)
    
    # Post-denoise after enhancement
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    return enhanced

# ----------------------------
# Visualization
# ----------------------------

def visualize_results(night_img, enhanced_img, day_img=None):
    images = [night_img, enhanced_img]
    titles = ['Night Image', 'Enhanced Image']
    if day_img is not None:
        images.append(day_img)
        titles.append('Reference Day Image')
    plt.figure(figsize=(15,5))
    for i,(img,title) in enumerate(zip(images,titles)):
        plt.subplot(1,len(images),i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.show()

# ----------------------------
# Example Usage
# ----------------------------

if __name__=="__main__":
    # Load day image for synthetic training
    day_img = cv2.imread("day.jpg")
    day_img = cv2.cvtColor(day_img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

    # Initialize model
    model = ColorRatioCorrection()

    # Train
    train_color_ratio_model(model, [day_img], epochs=200)

    # Load night image
    night_img = cv2.imread("night.jpg")
    night_rgb = cv2.cvtColor(night_img, cv2.COLOR_BGR2RGB)

    # Enhance
    enhanced_img = enhance_night_image(night_rgb, model)

    # Save
    cv2.imwrite("enhanced_result_per_channel.jpg", cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))

    # Visualize
    visualize_results(night_rgb, enhanced_img, (day_img*255).astype(np.uint8))
