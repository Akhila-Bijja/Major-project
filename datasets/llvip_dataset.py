import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

def resolve_directory(path):
    """Dynamically resolves directory paths, handling 'thermal' vs 'infrared', case sensitivity, and subfolders."""
    if os.path.exists(path):
        return path

    # Case 1: Try replacing 'thermal' with 'infrared' or 'Infrared'
    candidates = [
        path,
        path.replace("thermal", "infrared"),
        path.replace("thermal", "Infrared"),
        path.replace("visible", "Visible"),
        path.replace("LLVIP", "llvip"),
        path.replace("llvip-images", "llvip"),
        path.replace("llvip-images", "LLVIP")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Case 2: Deep search inside /kaggle/input or local directories
    target_split = path.replace("\\", "/").split("/")
    folder_type = target_split[-2].lower() if len(target_split) >= 2 else "" # e.g., 'thermal', 'infrared', 'visible'
    sub_type = target_split[-1].lower() if len(target_split) >= 1 else ""     # e.g., 'train', 'test'

    for search_root in ["/kaggle/input", "./data", "."]:
        if os.path.exists(search_root):
            for root, dirs, _ in os.walk(search_root):
                root_name = os.path.basename(root).lower()
                # Check if root is thermal/infrared/visible (or contains it)
                if (folder_type in root_name or (folder_type in ["thermal", "infrared"] and root_name in ["thermal", "infrared"])):
                    for d in dirs:
                        if d.lower() == sub_type:
                            resolved = os.path.join(root, d)
                            return resolved

    return path

class LLVIPDataset(Dataset):
    """
    PyTorch Dataset for LLVIP (Low-Light Visible-Infrared Paired) Dataset.
    Loads paired thermal (IR) and visible (RGB) images.
    """
    def __init__(self, thermal_dir, visible_dir, img_size=(256, 256), is_train=True):
        self.thermal_dir = resolve_directory(thermal_dir)
        self.visible_dir = resolve_directory(visible_dir)
        self.img_size = img_size
        self.is_train = is_train

        if not os.path.exists(self.thermal_dir):
            raise FileNotFoundError(f"Thermal/Infrared directory not found: {thermal_dir} (Resolved to: {self.thermal_dir})")
        if not os.path.exists(self.visible_dir):
            raise FileNotFoundError(f"Visible directory not found: {visible_dir} (Resolved to: {self.visible_dir})")

        print(f"[{'Train' if is_train else 'Test'}] Loaded Thermal from: {self.thermal_dir}")
        print(f"[{'Train' if is_train else 'Test'}] Loaded Visible from: {self.visible_dir}")

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
