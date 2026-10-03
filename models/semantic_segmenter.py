import torch
import torch.nn as nn
import torchvision.models as models

class SemanticSegmentationNetwork(nn.Module):
    """
    Module 3B: Semantic Segmentation / Context Network
    Recognizes scene objects and extracts contextual semantic feature maps.
    Can leverage a ResNet/DeepLab backbone or lightweight encoder.
    """
    def __init__(self, in_channels=3, feature_channels=64, use_pretrained_backbone=True):
        super().__init__()
        self.use_pretrained = use_pretrained_backbone
        
        if use_pretrained_backbone:
            # Use ResNet18 backbone for robust pre-trained feature extraction
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if hasattr(models, 'ResNet18_Weights') else True)
            self.initial = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            )
            self.layer1 = resnet.layer1  # 64 channels
            self.layer2 = resnet.layer2  # 128 channels
            self.layer3 = resnet.layer3  # 256 channels
            
            # Upsampling and feature projection to (B, feature_channels, H, W)
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, feature_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
            )
        else:
            # Standalone CNN Encoder
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, feature_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.use_pretrained:
            h = self.initial(x)
            h = self.layer1(h)
            h = self.layer2(h)
            h = self.layer3(h)
            semantic_features = self.upsample(h)  # Resized back to original resolution
        else:
            semantic_features = self.net(x)
            
        return semantic_features

if __name__ == "__main__":
    net = SemanticSegmentationNetwork(use_pretrained_backbone=False)
    x = torch.randn(2, 3, 256, 256)
    out = net(x)
    print(f"Semantic Features Shape: {out.shape}")
