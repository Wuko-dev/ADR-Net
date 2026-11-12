# ADR-Net

**ADR-Net: Attention-Decoupled Detail Restoration Network for Low-Light Image Enhancement**

Kang Liu, Zhe Yang, Zhihao Xv, and Runshi Ji

PyTorch implementation of ADR-Net for low-light image enhancement.

## Overview

ADR-Net combines efficient global channel relationship estimation with cross-stage detail aggregation in an encoder-decoder architecture.

- **Decoupled Self-Attention Block (DSAB):** its D-MSA operator estimates channel relationships from spatially compressed Query and Key features while retaining full-resolution Value features. This decouples relationship estimation from the transmission of restoration content.
- **Cross-Stage Detail Aggregation Module (CDAM):** aligns encoder features with the current decoder resolution, recalibrates individual-stage features, and refines their aggregated representation to provide complementary structural and detail cues.

## Results reported in the manuscript

The following LOL results use the **GT-Mean evaluation protocol** described in the manuscript. PSNR and SSIM are higher-is-better; LPIPS is lower-is-better.

| Dataset | PSNR (dB) | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| LOL-v1 | 27.84 | 0.883 | 0.075 |
| LOL-v2-real | 29.66 | 0.908 | 0.084 |
| LOL-v2-synthetic | 31.36 | 0.961 | 0.029 |

GT-Mean evaluation aligns an enhanced image's mean luminance with its reference before calculating metrics. It uses ground-truth information for benchmark evaluation and is not part of inference on an unseen image. Raw and GT-Mean scores should be reported separately.

## Installation

```bash
git clone https://github.com/Wuko-dev/ADR-Net.git
cd ADR-Net
```

Use an isolated Python environment and install a compatible PyTorch/torchvision pair using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). Select a CUDA-enabled build for training and the supplied LPIPS evaluation code.

Install the remaining dependencies:

```bash
python -m pip install numpy pillow opencv-python einops thop lpips scipy tqdm tensorboard
```

Check PyTorch and CUDA availability:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The VGG perceptual loss and LPIPS may download their pretrained feature-extractor weights on first use. These are separate from ADR-Net enhancement checkpoints.

## Repository structure

```text
ADR-Net/
├── model/
│   └── net.py                 # ADRNet, DSAB/D-MSA, and CDAM
├── data/                      # Dataset loaders and image transforms
├── scripts/
│   ├── options.py             # Training options and dataset paths
│   ├── losses.py              # Reconstruction, SSIM, edge, perceptual losses
│   ├── scheduler.py           # Learning-rate schedules
│   └── niqe_utils.py          # NIQE utilities
├── pretrained_weights/        # Training checkpoints
├── train.py                   # Training and validation
├── test.py                    # Dataset-specific evaluation entry point
├── measure.py                 # Paired PSNR, SSIM, and LPIPS evaluation
└── README.md
```

Datasets and pretrained ADR-Net checkpoints are not included in the source archive. Place an ADR-Net checkpoint in `pretrained_weights/` before running pretrained inference, or train a model using the instructions below.

## Datasets

### Download sources

Dataset links are collected from the official [HVI-CIDNet data preparation instructions](https://github.com/Fediory/HVI-CIDNet#data-preparation) and [Retinexformer dataset instructions](https://github.com/caiyuanhao1998/Retinexformer#2-prepare-dataset). Please follow the original dataset terms and cite the dataset publications when using them.

| Dataset | Download / source | Notes |
| --- | --- | --- |
| LOL-v1 | [Original dataset page](https://daooshee.github.io/BMVC2018website/) · [Google Drive mirror](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing) | 485 training pairs and 15 test pairs |
| LOL-v2-real / LOL-v2-synthetic | [Baidu Disk](https://pan.baidu.com/s/17KTa-6GUUW22Q49D5DhhWw?pwd=yixu) (`yixu`) · [Google Drive](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing) | Use the Real_captured and Synthetic subsets, respectively |
| SID / Sony-Total-Dark | [HVI data preparation](https://github.com/Fediory/HVI-CIDNet#data-preparation) · [Retinexformer data preparation](https://github.com/caiyuanhao1998/Retinexformer#2-prepare-dataset) | This implementation requires processed RGB image pairs; see the SID note below |
| DICM, LIME, MEF, NPE, VV | [Baidu Disk](https://pan.baidu.com/s/1FZ5HWT30eghGuaAqqpJGaw?pwd=yixu) (`yixu`) · [HVI data preparation](https://github.com/Fediory/HVI-CIDNet#data-preparation) | Unpaired evaluation images |
| ExDark | [Official dataset repository](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) | Downstream low-light object detection; detector training code is not included in this archive |

The cloud links above are third-party dataset mirrors listed by the cited repositories. If a mirror changes or requires sign-in, consult the corresponding source repository for updated links. Downloaded archives may use different directory names; arrange the files for the ADR-Net loaders as described below.

### Directory layout

Create a `datasets/` directory under the repository root. The following is an example layout; configure the actual paths in `scripts/options.py`.

```text
datasets/
├── LOLv1/
│   ├── our485/
│   │   ├── low/
│   │   └── high/
│   └── eval15/
│       ├── low/
│       └── high/
├── LOLv2/
│   ├── Real_captured/
│   │   ├── Train/
│   │   │   ├── Low/
│   │   │   └── Normal/
│   │   └── Test/
│   │       ├── Low/
│   │       └── Normal/
│   └── Synthetic/
│       ├── Train/
│       │   ├── Low/
│       │   └── Normal/
│       └── Test/
│           ├── Low/
│           └── Normal/
├── processed_sid_png/
│   ├── train/
│   │   ├── low/
│   │   └── high/
│   └── eval/
│       ├── low/
│       └── high/
├── DICM/
├── LIME/
├── MEF/
├── NPE/
└── VV/
```

Folder names are case-sensitive on Linux. Keep only image files in the input/target directories. For paired evaluation, predictions and references must have matching filenames.

**Training pair order:** the LOL training loaders pair entries by directory enumeration rather than by filename lookup. Verify that the low-light and reference lists returned by the loaders correspond image by image before training; equal folder sizes alone do not establish correct pairing.

**SID format:** `data/SICE_blur_SID.py` reads processed RGB images from flat `low/` and `high/` folders. RAW files, `.npy` arrays, or nested `short/long` folders from other implementations are not directly compatible. Prepare the same processed image pairs and split as the manuscript experiment, preserving the input/reference correspondence. A RAW-to-RGB preprocessing pipeline is not included in this archive.

## Training

### Configure the dataset

Edit the `default=` values in `scripts/options.py`. Enable exactly one dataset flag:

| Dataset | Flag to set to `True` | Training root | Validation input | Validation reference |
| --- | --- | --- | --- | --- |
| LOL-v1 | `lol_v1` | `data_train_lol_v1` | `data_val_lol_v1` | `data_valgt_lol_v1` |
| LOL-v2-real | `lolv2_real` | `data_train_lolv2_real` | `data_val_lolv2_real` | `data_valgt_lolv2_real` |
| LOL-v2-synthetic | `lolv2_syn` | `data_train_lolv2_syn` | `data_val_lolv2_syn` | `data_valgt_lolv2_syn` |
| SID | `SID` | `data_train_SID` | `data_val_SID` | `data_valgt_SID` |

For example, for LOL-v2-real, set the corresponding path defaults to:

```text
data_train_lolv2_real = ./datasets/LOLv2/Real_captured/Train
data_val_lolv2_real   = ./datasets/LOLv2/Real_captured/Test/Low
data_valgt_lolv2_real = ./datasets/LOLv2/Real_captured/Test/Normal/
```

Keep the trailing `/` on reference-directory paths: `measure.py` currently joins these paths and image filenames by string concatenation.

### Default training settings in the supplied code

| Setting | Option | Default |
| --- | --- | ---: |
| Batch size | `batchSize` | 16 |
| Training crop size | `cropSize` | 200 × 200 |
| Epochs | `nEpochs` | 2000 |
| Initial learning rate | `lr` | 0.0006 |
| Warm-up epochs | `warmup_epochs` | 3 |
| Reconstruction loss weight | `L1_weight` | 1.0 |
| SSIM loss weight | `ssim_weight` | 0.5 |
| Edge loss weight | `edge_weight` | 50.0 |
| Perceptual loss weight | `per_weight` | 0.005 |

The code uses Adam and enables the warm-up/cosine-restart schedule and GT-Mean loss by default. Configure `gpu_id` and `threads` for your machine. To select the configured seed, set `init_seed=True`; the default selects a random seed.

Run training from the repository root after configuring the defaults:

```bash
python train.py
```

In this version, `train.py` imports `test.py`, whose argument parser runs during import. Training-only command-line arguments can therefore be rejected by the test parser. Configure training through `scripts/options.py` and launch without extra arguments. The boolean options also use `type=bool`; edit their defaults instead of passing strings such as `--lolv2_real False`.

Outputs are written to timestamped directories:

```text
pretrained_weights/<run>/     # state_dict checkpoints
results/<run>/               # Validation images
runs/<run>/                  # TensorBoard events
```

The training script saves periodic `epoch_*.pth` checkpoints and separate best checkpoints for raw and GT-Mean metrics. Checkpoint files contain model parameters only; `start_epoch` does not restore model, optimizer, or scheduler state.

View training logs with:

```bash
tensorboard --logdir runs
```

## Inference

Use a checkpoint trained for the intended dataset. Training saves a plain PyTorch `state_dict`, which can be loaded with `ADRNet().load_state_dict(...)`.

The following example enhances a folder of RGB images without importing the dataset-specific parser in `test.py`. Save it as `infer_folder.py` in the repository root, set the three paths, and run `python infer_folder.py`.

```python
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from model.net import ADRNet

checkpoint = Path("pretrained_weights/your_run/best_gt_psnr.pth")
input_dir = Path("datasets/LOLv2/Real_captured/Test/Low")
output_dir = Path("output/LOLv2_real")

if not checkpoint.is_file():
    raise FileNotFoundError(f"ADR-Net checkpoint not found: {checkpoint}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ADRNet().to(device)
state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=True)
model.eval()
output_dir.mkdir(parents=True, exist_ok=True)

extensions = {".png", ".jpg", ".jpeg", ".bmp"}
images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in extensions)
if not images:
    raise RuntimeError(f"No input images found in {input_dir}")

with torch.no_grad():
    for path in images:
        with Image.open(path) as image:
            tensor = to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
        enhanced = model(tensor).clamp(0, 1)
        to_pil_image(enhanced[0].cpu()).save(output_dir / path.name)

print(f"Saved {len(images)} enhanced images to {output_dir}")
```

The model pads images internally and crops the output back to the original spatial size. The example uses `weights_only=True`, which requires a PyTorch version supporting that argument. For benchmark evaluation, retain lossless PNG output and the reference filenames.

### Using the supplied `test.py`

`test.py` contains dataset-specific input paths and checkpoint assignments. Before using it, replace those paths and set a valid `weight_path` in the selected branch. Empty checkpoint paths skip weight loading. Also move the model to the intended device, and pass the selected `norm_size` value into the final `eval(...)` call so that externally padded unpaired images are cropped correctly. The folder-inference example above handles device selection and preserves input dimensions without these manual changes.

## Paired evaluation

`measure.py` provides `metrics(im_dir, label_dir)`, returning six dataset averages: raw PSNR, SSIM, LPIPS, followed by GT-Mean PSNR, SSIM, LPIPS. Its LPIPS implementation requires CUDA.

Save the following as `evaluate_results.py` in the repository root, update the paths, and run `python evaluate_results.py`:

```python
from measure import metrics

scores = metrics(
    "./output/LOLv2_real/*.png",
    "./datasets/LOLv2/Real_captured/Test/Normal/",
)

print("Raw:     PSNR {:.4f} dB | SSIM {:.4f} | LPIPS {:.4f}".format(*scores[:3]))
print("GT-Mean: PSNR {:.4f} dB | SSIM {:.4f} | LPIPS {:.4f}".format(*scores[3:]))
```

Evaluate the complete test split, with exactly one prediction per reference. The current function computes both sets of metrics regardless of its `use_GT_mean` argument. It also resizes predictions to the reference size, so check that saved predictions already have the correct dimensions before interpreting benchmark scores.

For unpaired DICM, LIME, MEF, NPE, and VV evaluation, no reference image is available; do not apply GT-Mean alignment. NIQE utilities are included in `scripts/niqe_utils.py`, but this archive does not include a standalone unpaired-metric command or the downstream object-detection training pipeline.

## Acknowledgments

We thank the authors of [HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet) and [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) for making their research resources and dataset preparation instructions available, and the original dataset authors for providing the benchmarks.
