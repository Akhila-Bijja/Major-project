import os
import torch

class Config:
    # Environment & Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    
    # Dataset Configuration (Adjust paths according to Kaggle or Local environment)
    # Default Kaggle LLVIP path: /kaggle/input/llvip-dataset/LLVIP/
    # Local path alternative: ./data/LLVIP
    DATASET_ROOT = os.getenv("LLVIP_ROOT", "/kaggle/input/llvip-dataset/LLVIP")
    THERMAL_TRAIN_DIR = os.path.join(DATASET_ROOT, "thermal", "train")
    VISIBLE_TRAIN_DIR = os.path.join(DATASET_ROOT, "visible", "train")
    THERMAL_TEST_DIR = os.path.join(DATASET_ROOT, "thermal", "test")
    VISIBLE_TEST_DIR = os.path.join(DATASET_ROOT, "visible", "test")
    
    # Image Dimensions
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IN_CHANNELS = 3    # Thermal image loaded as 3-channel or RGB converted
    OUT_CHANNELS = 3   # Generated Visible RGB Image
    
    # Feature Channels
    STRUCTURE_FEAT_CHANNELS = 64
    SEMANTIC_FEAT_CHANNELS = 64
    FUSED_FEAT_CHANNELS = 128
    
    # Hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE_G = 2e-4
    LEARNING_RATE_D = 2e-4
    BETA1 = 0.5
    BETA2 = 0.999
    
    # Loss Weights
    LAMBDA_L1 = 100.0
    LAMBDA_PERCEPTUAL = 10.0
    LAMBDA_SSIM = 5.0
    LAMBDA_GAN = 1.0
    
    # Epochs
    STAGE1_EPOCHS = 15   # Pre-train Structure Extraction Model
    STAGE2_EPOCHS = 40   # Train Fusion + Generator (3A & 3B frozen)
    STAGE3_EPOCHS = 20   # End-to-End Fine-Tuning
    
    # Save & Checkpoint Paths
    CHECKPOINT_DIR = "./checkpoints"
    RESULTS_DIR = "./results"
    
    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)

if __name__ == "__main__":
    Config.create_dirs()
    print(f"Using device: {Config.DEVICE}")
