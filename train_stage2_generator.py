import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import Config
from datasets.llvip_dataset import LLVIPDataset
from models.structure_extractor import StructureExtractionNetwork
from models.semantic_segmenter import SemanticSegmentationNetwork
from models.fusion_module import FeatureFusionModule
from models.generator import AppearanceGenerator, PatchGANDiscriminator
from utils.loss import PerceptualLoss, SSIMLoss, GANLoss

def train_generator_stage():
    Config.create_dirs()
    print("--- STAGE 2: Training Fusion Module & Appearance Generator (Modules 4 & 5) ---")

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

    # Instantiate Models
    structure_model = StructureExtractionNetwork().to(Config.DEVICE)
    # Load Stage 1 checkpoint if available
    ckpt_struct = os.path.join(Config.CHECKPOINT_DIR, "structure_model_stage1.pth")
    if os.path.exists(ckpt_struct):
        structure_model.load_state_dict(torch.load(ckpt_struct, map_location=Config.DEVICE))
        print(f"Loaded Structure model weights from {ckpt_struct}")
    structure_model.eval()
    for param in structure_model.parameters():
        param.requires_grad = False

    semantic_model = SemanticSegmentationNetwork(use_pretrained_backbone=True).to(Config.DEVICE)
    semantic_model.eval()
    for param in semantic_model.parameters():
        param.requires_grad = False

    fusion_module = FeatureFusionModule().to(Config.DEVICE)
    generator = AppearanceGenerator().to(Config.DEVICE)
    discriminator = PatchGANDiscriminator().to(Config.DEVICE)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        list(fusion_module.parameters()) + list(generator.parameters()),
        lr=Config.LEARNING_RATE_G, betas=(Config.BETA1, Config.BETA2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=Config.LEARNING_RATE_D, betas=(Config.BETA1, Config.BETA2)
    )

    # Loss Functions
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(Config.DEVICE)
    ssim_loss = SSIMLoss().to(Config.DEVICE)
    gan_loss = GANLoss(gan_mode="lsgan").to(Config.DEVICE)

    for epoch in range(1, Config.STAGE2_EPOCHS + 1):
        g_loss_total, d_loss_total = 0.0, 0.0
        
        fusion_module.train()
        generator.train()
        discriminator.train()

        for i, batch in enumerate(train_loader):
            thermal_imgs = batch["thermal"].to(Config.DEVICE)
            visible_gt = batch["visible"].to(Config.DEVICE)

            # -----------------------------------------------
            # 1. Feature Extraction (Frozen 3A & 3B)
            # -----------------------------------------------
            with torch.no_grad():
                struct_feats, _ = structure_model(thermal_imgs)
                sem_feats = semantic_model(thermal_imgs)

            # -----------------------------------------------
            # 2. Train Generator & Fusion Module
            # -----------------------------------------------
            optimizer_G.zero_grad()
            
            fused_feats = fusion_module(thermal_imgs, struct_feats, sem_feats)
            fake_visible, conf_map = generator(fused_feats)

            # Discriminator predictions
            pred_fake = discriminator(thermal_imgs, fake_visible)
            
            # Calculate Combined Loss Functions
            loss_gan = gan_loss(pred_fake, target_is_real=True)
            loss_l1 = l1_loss(fake_visible, visible_gt)
            loss_perc = perceptual_loss(fake_visible, visible_gt)
            loss_ssim = ssim_loss(fake_visible, visible_gt)

            loss_G = (Config.LAMBDA_GAN * loss_gan) + \
                     (Config.LAMBDA_L1 * loss_l1) + \
                     (Config.LAMBDA_PERCEPTUAL * loss_perc) + \
                     (Config.LAMBDA_SSIM * loss_ssim)

            loss_G.backward()
            optimizer_G.step()

            # -----------------------------------------------
            # 3. Train Discriminator
            # -----------------------------------------------
            optimizer_D.zero_grad()
            
            # Real loss
            pred_real = discriminator(thermal_imgs, visible_gt)
            loss_D_real = gan_loss(pred_real, target_is_real=True)
            
            # Fake loss
            pred_fake_d = discriminator(thermal_imgs, fake_visible.detach())
            loss_D_fake = gan_loss(pred_fake_d, target_is_real=False)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            g_loss_total += loss_G.item()
            d_loss_total += loss_D.item()

        avg_g = g_loss_total / len(train_loader)
        avg_d = d_loss_total / len(train_loader)
        print(f"Epoch [{epoch}/{Config.STAGE2_EPOCHS}] - G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")

    # Save Checkpoints
    torch.save(fusion_module.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "fusion_module_stage2.pth"))
    torch.save(generator.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "generator_stage2.pth"))
    print("Stage 2 Checkpoints saved successfully.")

if __name__ == "__main__":
    train_generator_stage()
