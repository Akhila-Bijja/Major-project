import torch
import torch.nn as nn

class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class FeatureFusionModule(nn.Module):
    """
    Module 4: Feature Fusion Module
    Fuses structural features (3A) and semantic features (3B) along with raw thermal image inputs.
    Combines representations using channel concatenation, attention-weighted fusion, and residual Conv layers.
    """
    def __init__(self, in_thermal_channels=3, struct_channels=64, semantic_channels=64, fused_channels=128):
        super().__init__()
        total_in_channels = in_thermal_channels + struct_channels + semantic_channels
        
        # Initial projection layer
        self.proj = nn.Sequential(
            nn.Conv2d(total_in_channels, fused_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True)
        )
        
        # Spatial-Channel Attention Gate
        self.attention = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels // 2, fused_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Refinement Residual Blocks
        self.res1 = ResidualConvBlock(fused_channels)
        self.res2 = ResidualConvBlock(fused_channels)

    def forward(self, thermal_input, structure_features, semantic_features):
        # Concatenate along channel dimension
        cat_features = torch.cat([thermal_input, structure_features, semantic_features], dim=1)
        
        # Initial feature fusion
        fused = self.proj(cat_features)
        
        # Apply self-attention map
        att = self.attention(fused)
        fused = fused * att
        
        # Residual refinement
        fused = self.res1(fused)
        fused = self.res2(fused)
        
        return fused  # (B, fused_channels, H, W)

if __name__ == "__main__":
    fusion = FeatureFusionModule()
    t = torch.randn(2, 3, 256, 256)
    s = torch.randn(2, 64, 256, 256)
    m = torch.randn(2, 64, 256, 256)
    out = fusion(t, s, m)
    print(f"Fused Features Shape: {out.shape}")
