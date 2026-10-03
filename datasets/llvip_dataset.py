import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class LLVIPDataset(Dataset):
    """
    PyTorch Dataset for LLVIP (Low-Light Visible-Infrared Paired) Dataset.
    Loads paired thermal (IR) and visible (RGB) images.
    """
    def __init__(self, thermal_dir, visible_dir, img_size=(256, 256), is_train=True):
        self.thermal_dir = thermal_dir
        self.visible_dir = visible_dir
        self.img_size = img_size
        self.is_train = is_train

        # Support both 'thermal' and 'infrared' directory names automatically
        if not os.path.exists(thermal_dir):
            alt_thermal_dir = thermal_dir.replace("thermal", "infrared")
            if os.path.exists(alt_thermal_dir):
                thermal_dir = alt_thermal_dir
            else:
                raise FileNotFoundError(f"Thermal/Infrared directory not found: {thermal_dir} or {alt_thermal_dir}")

        if not os.path.exists(visible_dir):
            raise FileNotFoundError(f"Visible directory not found: {visible_dir}")

        self.thermal_dir = thermal_dir
        self.visible_dir = visible_dir

        # List all image files (assuming matching file names in both folders)
        valid_extensions = ('.jpg', '.png', '.jpeg', '.bmp')
        self.image_filenames = sorted([
            f for f in os.listdir(thermal_dir)
            if f.lower().endswith(valid_extensions)
        ])
        
        # Base transforms
        self.transform = T.Compose([
            T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Scale to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_filenames)

    def extract_canny_edges(self, np_img):
        """Extract Canny edges from an image array for structure training."""
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY) if len(np_img.shape) == 3 else np_img
        edges = cv2.Canny(gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0  # Scale to [0, 1]
        return torch.tensor(edges).unsqueeze(0)  # Shape (1, H, W)

    def __getitem__(self, index):
        filename = self.image_filenames[index]
        thermal_path = os.path.join(self.thermal_dir, filename)
        visible_path = os.path.join(self.visible_dir, filename)

        # Open Images
        thermal_pil = Image.open(thermal_path).convert("RGB")
        visible_pil = Image.open(visible_path).convert("RGB")

        # Dynamic Data Augmentation (Synchronized Flip)
        if self.is_train and np.random.rand() > 0.5:
            thermal_pil = T.functional.hflip(thermal_pil)
            visible_pil = T.functional.hflip(visible_pil)

        # Apply standard transforms
        thermal_tensor = self.transform(thermal_pil)
        visible_tensor = self.transform(visible_pil)

        # Generate edge map from visible/thermal for Structure Model training
        visible_np = np.array(visible_pil.resize(self.img_size))
        edge_tensor = self.extract_canny_edges(visible_np)

        return {
            "filename": filename,
            "thermal": thermal_tensor,    # Input X: (3, H, W) in [-1, 1]
            "visible": visible_tensor,    # Ground Truth Y: (3, H, W) in [-1, 1]
            "edges": edge_tensor          # Structure Target: (1, H, W) in [0, 1]
        }

if __name__ == "__main__":
    print("LLVIP Dataset module initialized successfully.")
