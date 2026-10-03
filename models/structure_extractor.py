import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class StructureExtractionNetwork(nn.Module):
    """
    Module 3A: Structure Extraction Network (U-Net style)
    Extracts high-frequency spatial details (edges, contours, shape information).
    Outputs both intermediate feature maps (for fusion) and 1-channel predicted edge map.
    """
    def __init__(self, in_channels=3, feature_channels=64):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = ConvBlock(128, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, feature_channels)
        
        # Edge prediction head
        self.edge_head = nn.Sequential(
            nn.Conv2d(feature_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        structure_features = self.dec1(d1)  # (B, feature_channels, H, W)
        
        edge_map = self.edge_head(structure_features)  # (B, 1, H, W)
        
        return structure_features, edge_map

if __name__ == "__main__":
    net = StructureExtractionNetwork()
    x = torch.randn(2, 3, 256, 256)
    feat, edge = net(x)
    print(f"Structure Features Shape: {feat.shape}, Edge Map Shape: {edge.shape}")
