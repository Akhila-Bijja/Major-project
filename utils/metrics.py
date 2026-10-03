import torch
import torch.nn.functional as F
import numpy as np
import cv2

def calculate_psnr(img1, img2):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between two PyTorch Tensors in [-1, 1].
    """
    # Convert from [-1, 1] to [0, 255]
    img1 = ((img1 + 1.0) * 127.5).clamp(0, 255)
    img2 = ((img2 + 1.0) * 127.5).clamp(0, 255)
    
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, window_size=11):
    """
    Computes Structural Similarity Index Metric (SSIM) value between 0 and 1.
    """
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
               
    return ssim_map.mean().item()

if __name__ == "__main__":
    t1 = torch.randn(1, 3, 256, 256)
    t2 = t1 + torch.randn(1, 3, 256, 256) * 0.1
    print(f"PSNR: {calculate_psnr(t1, t2):.2f} dB")
    print(f"SSIM: {calculate_ssim(t1, t2):.4f}")
