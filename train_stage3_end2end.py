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

def train_end2end():
    Config.create_dirs()
    print("--- STAGE 3: End-to-End Joint Fine-Tuning ---")

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

    # Instantiate Models & Load Checkpoints
    structure_model = StructureExtractionNetwork().to(Config.DEVICE)
    ckpt_struct = os.path.join(Config.CHECKPOINT_DIR, "structure_model_stage1.pth")
    if os.path.exists(ckpt_struct):
        structure_model.load_state_dict(torch.load(ckpt_struct, map_location=Config.DEVICE))

    semantic_model = SemanticSegmentationNetwork(use_pretrained_backbone=True).to(Config.DEVICE)

    fusion_module = FeatureFusionModule().to(Config.DEVICE)
    ckpt_fusion = os.path.join(Config.CHECKPOINT_DIR, "fusion_module_stage2.pth")
    if os.path.exists(ckpt_fusion):
        fusion_module.load_state_dict(torch.load(ckpt_fusion, map_location=Config.DEVICE))

    generator = AppearanceGenerator().to(Config.DEVICE)
    ckpt_gen = os.path.join(Config.CHECKPOINT_DIR, "generator_stage2.pth")
    if os.path.exists(ckpt_gen):
        generator.load_state_dict(torch.load(ckpt_gen, map_location=Config.DEVICE))

    discriminator = PatchGANDiscriminator().to(Config.DEVICE)

    # UNFREEZE ALL PARAMETERS FOR JOINT FINE-TUNING
    all_gen_params = (
        list(structure_model.parameters()) +
        list(semantic_model.parameters()) +
        list(fusion_module.parameters()) +
        list(generator.parameters())
    )

    # Smaller learning rate for fine-tuning stability
    optimizer_G = torch.optim.Adam(all_gen_params, lr=Config.LEARNING_RATE_G * 0.2, betas=(Config.BETA1, Config.BETA2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=Config.LEARNING_RATE_D * 0.2, betas=(Config.BETA1, Config.BETA2))

    # Loss Functions
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(Config.DEVICE)
    ssim_loss = SSIMLoss().to(Config.DEVICE)
    gan_loss = GANLoss(gan_mode="lsgan").to(Config.DEVICE)

    for epoch in range(1, Config.STAGE3_EPOCHS + 1):
        structure_model.train()
        semantic_model.train()
        fusion_module.train()
        generator.train()
        discriminator.train()

        g_loss_total, d_loss_total = 0.0, 0.0

        for i, batch in enumerate(train_loader):
            thermal_imgs = batch["thermal"].to(Config.DEVICE)
            visible_gt = batch["visible"].to(Config.DEVICE)

            # 1. Forward Pass Through Full Pipeline
            optimizer_G.zero_grad()

            struct_feats, pred_edges = structure_model(thermal_imgs)
            sem_feats = semantic_model(thermal_imgs)
            fused_feats = fusion_module(thermal_imgs, struct_feats, sem_feats)
            fake_visible, conf_map = generator(fused_feats)

            pred_fake = discriminator(thermal_imgs, fake_visible)

            # Combined Losses
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

            # 2. Train Discriminator
            optimizer_D.zero_grad()
            
            pred_real = discriminator(thermal_imgs, visible_gt)
            loss_D_real = gan_loss(pred_real, target_is_real=True)

            pred_fake_d = discriminator(thermal_imgs, fake_visible.detach())
            loss_D_fake = gan_loss(pred_fake_d, target_is_real=False)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            g_loss_total += loss_G.item()
            d_loss_total += loss_D.item()

        avg_g = g_loss_total / len(train_loader)
        avg_d = d_loss_total / len(train_loader)
        print(f"Fine-Tuning Epoch [{epoch}/{Config.STAGE3_EPOCHS}] - G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")

    # Save Final Models
    torch.save(structure_model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "structure_final.pth"))
    torch.save(semantic_model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "semantic_final.pth"))
    torch.save(fusion_module.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "fusion_final.pth"))
    torch.save(generator.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "generator_final.pth"))
    print("All Final End-to-End Models saved successfully!")

if __name__ == "__main__":
    train_end2end()
