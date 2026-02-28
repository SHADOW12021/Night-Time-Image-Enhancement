# n2d3_train.py
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import json
import pytorch_msssim
from skimage.exposure import match_histograms
from disentanglement import N2D3Disentanglement  # Your previous module

# -------------------------------
# 1. Dataset
# -------------------------------
class Night2DayDataset(Dataset):
    """
    Unpaired dataset:
    - Night images → domain A
    - Day images → domain B
    """
    def __init__(self, root, image_size=256):
        super().__init__()
        self.img_dir = os.path.join(root, "train", "img")
        self.ann_dir = os.path.join(root, "train", "ann")

        # self.transform = transform if transform else transforms.Compose([
        #     # transforms.Resize((256, 256)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        # ])
        
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.RandomCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        self.night_imgs = []
        self.day_imgs = []

        for img_name in os.listdir(self.img_dir):
            ann_path = os.path.join(self.ann_dir, img_name + ".json")
            if not os.path.exists(ann_path):
                continue

            with open(ann_path, "r") as f:
                ann = json.load(f)

            for tag in ann.get("tags", []):
                if tag.get("name") == "timeofday":
                    if tag.get("value") == "night":
                        self.night_imgs.append(img_name)
                    elif tag.get("value") == "daytime":
                        self.day_imgs.append(img_name)

        print(f"Night images: {len(self.night_imgs)}")
        print(f"Day images: {len(self.day_imgs)}")
        
        random.shuffle(self.night_imgs)
        random.shuffle(self.day_imgs)

        self.night_imgs = self.night_imgs[:5000]
        self.day_imgs   = self.day_imgs[:5000]

        print(f"Using Night subset: {len(self.night_imgs)}")
        print(f"Using Day subset: {len(self.day_imgs)}")

    def __len__(self):
        return max(len(self.night_imgs), len(self.day_imgs))

    def __getitem__(self, idx):
        night_img = Image.open(os.path.join(
            self.img_dir,
            self.night_imgs[idx % len(self.night_imgs)]
        )).convert("RGB")

        day_img = Image.open(os.path.join(
            self.img_dir,
            random.choice(self.day_imgs)
        )).convert("RGB")

        return self.transform(night_img), self.transform(day_img)
# -------------------------------
# 2. Generator & Discriminator (ResNet GAN skeletons)
# -------------------------------
class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_res_blocks=9):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )

        self.res_blocks = nn.Sequential(
            *[ResNetBlock(256) for _ in range(n_res_blocks)]
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, out_channels, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x, return_features=False):
        f1 = self.init_conv(x)     # 64 x 256x256
        f2 = self.down1(f1)        # 128 x 128x128
        f3 = self.down2(f2)        # 256 x 64x64
        f4 = self.res_blocks(f3)   # 256 x 64x64

        out = self.up(f4)

        if return_features:
            return out, [f1, f2, f3, f4]
        return out

class Discriminator(nn.Module):
    """
    PatchGAN discriminator.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        def conv_block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
            if norm: layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv_block(in_channels, 64, norm=False),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------------
# 3. Loss Functions
# -------------------------------

def edge_loss(fake, real):
    def sobel(x):
        kernel_x = torch.tensor([[1,0,-1],
                                 [2,0,-2],
                                 [1,0,-1]], dtype=torch.float32,
                                 device=x.device).view(1,1,3,3)

        kernel_y = torch.tensor([[1,2,1],
                                 [0,0,0],
                                 [-1,-2,-1]], dtype=torch.float32,
                                 device=x.device).view(1,1,3,3)

        # Apply per channel
        grads = []
        for c in range(x.shape[1]):
            ch = x[:, c:c+1]
            gx = F.conv2d(ch, kernel_x, padding=1)
            gy = F.conv2d(ch, kernel_y, padding=1)
            grads.append(torch.sqrt(gx**2 + gy**2 + 1e-6))

        return torch.cat(grads, dim=1)

    edge_fake = sobel(fake)
    edge_real = sobel(real)

    return F.l1_loss(edge_fake, edge_real)

def gan_loss(pred, target_is_real):
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)

def feature_patch_nce_loss(fake_feats, real_feats, masks,
                           mlps, num_patches=256,
                           temperature=0.07):

    total_loss = 0.0
    device = fake_feats[0].device

    for feat_fake, feat_real, mlp in zip(fake_feats, real_feats, mlps):

        B, C, H, W = feat_fake.shape

        # flatten spatial dims
        feat_fake = feat_fake.permute(0, 2, 3, 1).reshape(B, -1, C)
        feat_real = feat_real.permute(0, 2, 3, 1).reshape(B, -1, C)

        for b in range(B):

            # random spatial sampling
            idx = torch.randperm(feat_fake.shape[1])[:num_patches]

            f_fake = feat_fake[b, idx]
            f_real = feat_real[b, idx]

            # projection head
            z_fake = mlp(f_fake)
            z_real = mlp(f_real)

            z_fake = F.normalize(z_fake, dim=1)
            z_real = F.normalize(z_real, dim=1)

            logits = torch.mm(z_fake, z_real.t()) / temperature
            labels = torch.arange(len(idx), device=device)

            loss = F.cross_entropy(logits, labels)
            total_loss += loss

    return total_loss / len(fake_feats)


# -------------------------------
class PatchSampleMLP(nn.Module):
    def __init__(self, in_channels, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)
    
# -------------------------------
# 4. Training Loop
# -------------------------------
def train_n2d3(data_root="bdd100k_-images-100k-DatasetNinja",
               epochs=20,
               batch_size=8,
               image_size=256):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    G = Generator().to(device)
    D = Discriminator().to(device)
    disentangler = N2D3Disentanglement().to(device)

    # ----------------------------
    # Create MLP projection heads
    # ----------------------------
    mlps = nn.ModuleList([
        PatchSampleMLP(64),
        PatchSampleMLP(128),
        PatchSampleMLP(256),
        PatchSampleMLP(256)
    ]).to(device)

    # Initialize MLP weights
    for m in mlps.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ----------------------------
    # Create optimizers
    # ----------------------------
    lr = 3e-4 if image_size == 256 else 2e-4 if image_size == 512 else 1e-4
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_F = torch.optim.Adam(mlps.parameters(), lr=lr, betas=(0.5, 0.999))
    
    start_epoch = 0
    checkpoint_path = "checkpoints/latest.pth"

    if os.path.exists(checkpoint_path):
        print("🔄 Loading checkpoint...")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        G.load_state_dict(checkpoint["G"])
        D.load_state_dict(checkpoint["D"])
        mlps.load_state_dict(checkpoint["mlps"])

        opt_G.load_state_dict(checkpoint["opt_G"])
        opt_D.load_state_dict(checkpoint["opt_D"])
        opt_F.load_state_dict(checkpoint["opt_F"])

        start_epoch = checkpoint["epoch"] + 1

        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting fresh training")


    dataset = Night2DayDataset(data_root, image_size = image_size)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=8, pin_memory=True)

    lambda_gan = 0.05      
    lambda_nce = 1.0
    lambda_id = 25.0      
    lambda_edge = 5.0  
    lambda_l1 = 20.0
    lambda_ssim = 10.0

    for epoch in range(start_epoch, epochs):
        for i, (real_night, real_day) in enumerate(loader):

            real_night = real_night.to(device)
            real_day = real_day.to(device)
            

            # ----------------------------------
            # 1️⃣ Train Discriminator
            # ----------------------------------
            opt_D.zero_grad()

            # Real prediction
            pred_real = D(real_day)
            loss_D_real = gan_loss(pred_real, True)

            # Generate fake day image (detach for D)
            fake_day = G(real_night).detach()  # <-- generate here
            pred_fake = D(fake_day)
            loss_D_fake = gan_loss(pred_fake, False)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            # ----------------------------------
            # 2️⃣ Train Generator
            # ----------------------------------
            opt_G.zero_grad()
            opt_F.zero_grad()

            # Forward with features
            fake_day, fake_feats = G(real_night, return_features=True)
            
            loss_L1 = F.l1_loss(fake_day, real_day)
            ssim_loss = 1 - pytorch_msssim.ssim(fake_day, real_day, data_range=2.0)

            # GAN loss
            pred_fake = D(fake_day)
            loss_GAN = gan_loss(pred_fake, True)

            # Real features (encoder only)
            with torch.no_grad():
                _, real_feats = G(real_night, return_features=True)

            # Optional: physics masks (if you want to integrate later)
            with torch.no_grad():
                masks = disentangler(real_night)

            loss_NCE = feature_patch_nce_loss(
                fake_feats, real_feats, masks, mlps
            )

            # Identity
            identity = G(real_day)
            loss_EDGE = edge_loss(fake_day, real_night)
            loss_ID = F.l1_loss(identity, real_day)

            loss_G = (
            lambda_gan * loss_GAN +
            lambda_nce * loss_NCE +
            lambda_id * loss_ID +
            lambda_edge * loss_EDGE +
            lambda_l1 * loss_L1 + 
            lambda_ssim * ssim_loss
            )

            loss_G.backward()

            opt_G.step()
            opt_F.step()
            
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] "
                        f"[Iter {i}/{len(loader)}] "
                        f"[G: {loss_G.item():.4f}] "
                        f"[D: {loss_D.item():.4f}] "
                        f"[GAN: {loss_GAN.item():.4f}] "
                        f"[NCE: {loss_NCE.item():.4f}] "
                        f"[ID: {loss_ID.item():.4f}] "
                        f"[EDGE: {loss_EDGE.item():.4f}]")

        os.makedirs("checkpoints", exist_ok=True)

        checkpoint = {
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "mlps": mlps.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "opt_F": opt_F.state_dict()
        }
        
        torch.save(checkpoint, "checkpoints/latest.pth")
        
        if epoch % 5 == 0:
            torch.save(checkpoint, f"checkpoints/model_epoch{epoch}.pth")

    print("Training complete.")
    
    # mlps = nn.ModuleList([
    # PatchSampleMLP(64),
    # PatchSampleMLP(128),
    # PatchSampleMLP(256),
    # PatchSampleMLP(256)
    # ]).to(device)
    
    # for m in mlps.modules():
    #     if isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight, 0, 0.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
                
    # opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    # opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    # opt_F = torch.optim.Adam(mlps.parameters(), lr=2e-4, betas=(0.5, 0.999))

# -------------------------------
# 5. Night2Day Translation Utility
# -------------------------------
def translate_night_to_day(image_path, model_weights_path="checkpoints/latest.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Generator().to(device)
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint["G"])
    model.eval()

    transform = transforms.Compose([
        # transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    # output_img = transforms.ToPILImage()(
    #     (output.squeeze().cpu() + 1.0) / 2.0
    # )
    
    fake_np = (output.squeeze().cpu().numpy() + 1.0) / 2.0
    fake_np = np.transpose(fake_np, (1,2,0))

    real_np = np.array(img) / 255.0

    matched = match_histograms(fake_np, real_np, channel_axis=-1)

    output_img = Image.fromarray((matched * 255).astype(np.uint8))

    save_path = "translated_day.jpg"
    output_img.save(save_path)
 
    print(f"Saved translated image as {save_path}")
    

# -------------------------------
# 6. Run training
# -------------------------------
if __name__ == "__main__":
    # train_n2d3()
    
    # Phase 1: 256 resolution
    train_n2d3(epochs=10, batch_size=8, image_size=256)

    # Phase 2: 512 resolution
    train_n2d3(epochs=7, batch_size=4, image_size=512)

    # Phase 3: 720 resolution (near full BDD)
    train_n2d3(epochs=5, batch_size=2, image_size=720)

    # After training, translate sample image
    translate_night_to_day("night.jpg")
