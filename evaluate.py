import os
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from config import Config
from datasets.llvip_dataset import LLVIPDataset
from models.structure_extractor import StructureExtractionNetwork
from models.semantic_segmenter import SemanticSegmentationNetwork
from models.fusion_module import FeatureFusionModule
from models.generator import AppearanceGenerator
from utils.metrics import calculate_psnr, calculate_ssim

def evaluate():
    Config.create_dirs()
    print("--- Evaluating Model on Test Dataset ---")

    test_dataset = LLVIPDataset(
        thermal_dir=Config.THERMAL_TEST_DIR,
        visible_dir=Config.VISIBLE_TEST_DIR,
        img_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        is_train=False
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load Models
    structure_model = StructureExtractionNetwork().to(Config.DEVICE)
    semantic_model = SemanticSegmentationNetwork().to(Config.DEVICE)
    fusion_module = FeatureFusionModule().to(Config.DEVICE)
    generator = AppearanceGenerator().to(Config.DEVICE)

    # Load weights
    def load_weights(model, name):
        path = os.path.join(Config.CHECKPOINT_DIR, name)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=Config.DEVICE))
            print(f"Loaded weights: {name}")

    load_weights(structure_model, "structure_final.pth")
    load_weights(semantic_model, "semantic_final.pth")
    load_weights(fusion_module, "fusion_final.pth")
    load_weights(generator, "generator_final.pth")

    structure_model.eval()
    semantic_model.eval()
    fusion_module.eval()
    generator.eval()

    psnr_scores, ssim_scores = [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            thermal = batch["thermal"].to(Config.DEVICE)
            visible_gt = batch["visible"].to(Config.DEVICE)
            filename = batch["filename"][0]

            struct_feats, pred_edges = structure_model(thermal)
            sem_feats = semantic_model(thermal)
            fused = fusion_module(thermal, struct_feats, sem_feats)
            generated_visible, confidence_map = generator(fused)

            # Compute Metrics
            psnr_val = calculate_psnr(generated_visible, visible_gt)
            ssim_val = calculate_ssim(generated_visible, visible_gt)
            
            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)

            # Save qualitative visual comparison every 10 images or for first 50
            if i < 50:
                # Expand confidence map to 3-channels for visualization
                conf_3ch = confidence_map.repeat(1, 3, 1, 1) * 2.0 - 1.0
                edge_3ch = pred_edges.repeat(1, 3, 1, 1) * 2.0 - 1.0

                comparison = torch.cat([
                    thermal,             # Thermal Input
                    edge_3ch,            # Structure Map
                    generated_visible,   # Generated Visible
                    visible_gt,          # Ground Truth Visible
                    conf_3ch             # Confidence Map
                ], dim=3)  # Concatenate horizontally

                save_path = os.path.join(Config.RESULTS_DIR, f"result_{i:04d}_{filename}")
                vutils.save_image(comparison, save_path, normalize=True, value_range=(-1, 1))

    mean_psnr = sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0
    mean_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0

    print("================ EVALUATION RESULTS ================")
    print(f"Mean PSNR: {mean_psnr:.4f} dB")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print(f"Saved evaluation comparison images to: {Config.RESULTS_DIR}")

if __name__ == "__main__":
    evaluate()
