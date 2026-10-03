import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import Config
from datasets.llvip_dataset import LLVIPDataset
from models.structure_extractor import StructureExtractionNetwork

def train_structure_model():
    Config.create_dirs()
    print("--- STAGE 1: Training Structure Extraction Network (Module 3A) ---")
    
    # Dataset & DataLoader
    train_dataset = LLVIPDataset(
        thermal_dir=Config.THERMAL_TRAIN_DIR,
        visible_dir=Config.VISIBLE_TRAIN_DIR,
        img_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        is_train=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Model & Optimizer
    structure_model = StructureExtractionNetwork(
        in_channels=Config.IN_CHANNELS,
        feature_channels=Config.STRUCTURE_FEAT_CHANNELS
    ).to(Config.DEVICE)
    
    optimizer = torch.optim.Adam(structure_model.parameters(), lr=1e-3)
    bce_loss = nn.BCELoss()
    
    structure_model.train()
    for epoch in range(1, Config.STAGE1_EPOCHS + 1):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            thermal_imgs = batch["thermal"].to(Config.DEVICE)
            gt_edges = batch["edges"].to(Config.DEVICE)  # Shape (B, 1, H, W)
            
            optimizer.zero_grad()
            _, pred_edges = structure_model(thermal_imgs)
            
            loss = bce_loss(pred_edges, gt_edges)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{Config.STAGE1_EPOCHS}] - Structure Loss: {avg_loss:.4f}")
        
    # Save Checkpoint
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "structure_model_stage1.pth")
    torch.save(structure_model.state_dict(), ckpt_path)
    print(f"Structure Model saved successfully at: {ckpt_path}")

if __name__ == "__main__":
    train_structure_model()
