import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_norm=True):
        super().__init__()

        if down:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )

        self.norm = nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class NightToDayGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_residuals=6):
        super().__init__()

        # -------- Encoder --------
        self.enc1 = ConvBlock(in_channels, 64, down=True, use_norm=False)
        self.enc2 = ConvBlock(64, 128, down=True)
        self.enc3 = ConvBlock(128, 256, down=True)
        self.enc4 = ConvBlock(256, 512, down=True)

        # -------- Bottleneck --------
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(512) for _ in range(num_residuals)]
        )

        # -------- Decoder --------
        self.dec4 = ConvBlock(512, 256, down=False)
        self.dec3 = ConvBlock(512, 128, down=False)
        self.dec2 = ConvBlock(256, 64, down=False)

        self.final_conv = nn.ConvTranspose2d(
            128,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)     # [B, 64, H/2, W/2]
        e2 = self.enc2(e1)    # [B, 128, H/4, W/4]
        e3 = self.enc3(e2)    # [B, 256, H/8, W/8]
        e4 = self.enc4(e3)    # [B, 512, H/16, W/16]

        # Bottleneck
        b = self.res_blocks(e4)

        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e3], dim=1)

        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        out = self.final_conv(d2)

        return self.tanh(out)   