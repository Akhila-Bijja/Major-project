import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)

class AppearanceGenerator(nn.Module):
    """
    Module 5: Appearance Generation Model
    Takes fused structural and semantic features and synthesizes realistic visible RGB images.
    Also produces an optional Confidence Map indicating generation reliability (Module 7).
    """
    def __init__(self, in_fused_channels=128, out_rgb_channels=3, num_res_blocks=6):
        super().__init__()
        
        # Initial Downsampling / Encoding
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_fused_channels, 128, kernel_size=7),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        model += [
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        ]
        
        # Residual Bottleneck Blocks
        for _ in range(num_res_blocks):
            model.append(ResNetBlock(256))
            
        # Upsampling
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        ]
        
        self.backbone = nn.Sequential(*model)
        
        # Output Head: Visible RGB Image
        self.rgb_head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(128, out_rgb_channels, kernel_size=7),
            nn.Tanh()  # Output range [-1, 1]
        )
        
        # Output Head: Module 7 Confidence Map (Optional reliability estimation)
        self.confidence_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 1, kernel_size=3),
            nn.Sigmoid()  # Output range [0, 1]
        )

    def forward(self, fused_features):
        feat = self.backbone(fused_features)
        visible_img = self.rgb_head(feat)
        confidence_map = self.confidence_head(feat)
        
        return visible_img, confidence_map

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Adversarial Training.
    Classifies image patches as Real or Generated/Fake.
    """
    def __init__(self, in_channels=6):  # Thermal + Visible concatenated
        super().__init__()
        
        def discriminator_block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, thermal_img, visible_img):
        # Concatenate Thermal input and Visible candidate
        img_input = torch.cat([thermal_img, visible_img], dim=1)
        return self.model(img_input)

if __name__ == "__main__":
    gen = AppearanceGenerator()
    disc = PatchGANDiscriminator()
    fused = torch.randn(2, 128, 256, 256)
    rgb, conf = gen(fused)
    print(f"Generated RGB Shape: {rgb.shape}, Confidence Map Shape: {conf.shape}")
    
    thermal = torch.randn(2, 3, 256, 256)
    validity = disc(thermal, rgb)
    print(f"Discriminator Output Shape: {validity.shape}")
