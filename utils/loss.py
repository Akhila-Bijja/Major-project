import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using pre-trained VGG16 features to measure feature similarity.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT if hasattr(models, 'VGG16_Weights') else True).features
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(4)]).eval()
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)]).eval()
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 16)]).eval()
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # Normalize from [-1, 1] to VGG expectation [0, 1]
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        
        h_pred1 = self.slice1(pred)
        h_target1 = self.slice1(target)
        
        h_pred2 = self.slice2(h_pred1)
        h_target2 = self.slice2(h_target1)
        
        h_pred3 = self.slice3(h_pred2)
        h_target3 = self.slice3(h_target2)
        
        loss = F.l1_loss(h_pred1, h_target1) + \
               F.l1_loss(h_pred2, h_target2) + \
               F.l1_loss(h_pred3, h_target3)
               
        return loss

class SSIMLoss(nn.Module):
    """
    Differentiable 1D/2D Structural Similarity Index (SSIM) Loss.
    """
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, img1, img2):
        # Rescale inputs to [0, 1]
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0

        mu1 = F.avg_pool2d(img1, self.window_size, stride=1, padding=self.window_size // 2)
        mu2 = F.avg_pool2d(img2, self.window_size, stride=1, padding=self.window_size // 2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 * img1, self.window_size, stride=1, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, self.window_size, stride=1, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size, stride=1, padding=self.window_size // 2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)) / \
                   ((mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2))
                   
        return 1.0 - ssim_map.mean()

class GANLoss(nn.Module):
    """
    Least-Squares GAN (LSGAN) or Vanilla BCE GAN Loss.
    """
    def __init__(self, gan_mode="lsgan"):
        super().__init__()
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        target_val = 1.0 if target_is_real else 0.0
        return torch.full_like(prediction, target_val)

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

if __name__ == "__main__":
    print("Loss functions defined successfully.")
