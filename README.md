# Structure-Aware Semantic-Guided Thermal-to-Visible Image Translation (LLVIP)

This repository implements the multi-stage deep learning pipeline for translating Thermal Infrared (IR) images into realistic Visible (RGB) spectrum images using:
1. **Module 3A**: Structure Extraction Network (U-Net + Edge prediction)
2. **Module 3B**: Semantic Context Network (DeepLabv3+ / ResNet Backbone)
3. **Module 4**: Feature Fusion Module (Attention Gate + Residual Convolutions)
4. **Module 5**: Appearance Generator (ResNet Generator + LSGAN PatchGAN Discriminator)
5. **Module 7**: Reliability Confidence Map Estimator

---

## 📁 Repository Structure

```
Major Project/
├── config.py                   # Central Configuration & Hyperparameters
├── datasets/
│   └── llvip_dataset.py        # PyTorch Dataset Loader for LLVIP Paired Images
├── models/
│   ├── structure_extractor.py  # Module 3A: Structure Extraction Network
│   ├── semantic_segmenter.py   # Module 3B: Semantic Context Network
│   ├── fusion_module.py        # Module 4: Spatial-Channel Feature Fusion
│   └── generator.py            # Module 5 & 7: Appearance Generator & PatchGAN Discriminator
├── utils/
│   ├── loss.py                 # L1, Perceptual (VGG16), SSIM, and LSGAN Losses
│   └── metrics.py              # PSNR and SSIM Evaluation Metrics
├── train_stage1_structure.py   # Stage 1: Pre-train Structure Network
├── train_stage2_generator.py   # Stage 2: Train Fusion + Generator (Modules 3A & 3B Frozen)
├── train_stage3_end2end.py     # Stage 3: End-to-End Joint Fine-Tuning
└── evaluate.py                 # Evaluation & Visual Comparison Generator
```

---

## ⚡ How to Run on Kaggle (Step-by-Step)

### Step 1: Create a Kaggle Notebook
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code) and click **New Notebook**.
2. Set Environment **GPU** (NVIDIA T4 or P100 GPU).

### Step 2: Add LLVIP Dataset
1. In the right panel of Kaggle Notebook, click **+ Add Data**.
2. Search for `LLVIP` or `LLVIP Dataset`.
3. Add the dataset to your notebook. The path will automatically be:
   `/kaggle/input/llvip-dataset/LLVIP`

### Step 3: Run Training & Evaluation

Copy and run the following cells in your Kaggle Notebook:

```python
# Cell 1: Clone or Copy repository files & Setup environment
!git clone <YOUR_GITHUB_REPO_URL> repo || true
%cd repo

# Cell 2: Run Stage 1 (Pre-train Structure Model)
!python train_stage1_structure.py

# Cell 3: Run Stage 2 (Train Fusion + Appearance Generator)
!python train_stage2_generator.py

# Cell 4: Run Stage 3 (End-to-End Joint Fine-Tuning)
!python train_stage3_end2end.py

# Cell 5: Evaluate & Save Output Images
!python evaluate.py
```

---

## 📊 Outputs & Visualizations
The `evaluate.py` script automatically computes:
* **PSNR (Peak Signal-to-Noise Ratio)**
* **SSIM (Structural Similarity Index Metric)**

And generates side-by-side comparison images:
`[ Thermal Input | Structure Edge Map | Generated Visible RGB | Ground Truth RGB | Confidence Map ]`
saved directly in `./results/`.
