# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LHM (Large Human Model) is a PyTorch implementation for photorealistic 3D human avatar reconstruction and animation from single images. It uses 3D Gaussian Splatting, SMPL-X body models, and large vision encoders (DINO, Sapiens).

## Environment Setup

Requires Python 3.10 and CUDA 11.8 or 12.1.

```bash
python -m venv lhm_env
source lhm_env/bin/activate
bash install_cu121.sh   # or install_cu118.sh for CUDA 11.8
```

Manual dependency installation order matters:
```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip uninstall basicsr && pip install git+https://github.com/XPixelGroup/BasicSR
pip install git+https://github.com/hitsz-zuoqi/sam2/
```

## Running Inference

```bash
# Gradio UI (image preprocessing + static animation)
python app.py [--model_name LHM-1B]

# Gradio UI with pose estimation (needs ~24GB VRAM)
python app_motion.py [--model_name LHM-1B-HF]

# Memory-saving motion animation
python app_motion_ms.py

# Batch inference
bash inference.sh <MODEL_NAME> <IMAGE_PATH> <MOTION_SEQS_DIR>

# Export 3D mesh
bash inference_mesh.sh <MODEL_NAME>

# Extract motion from video
python engine/pose_estimation/video2motion.py --video_path <VIDEO> --output_path <OUTPUT>

# CLI runner (used internally by inference.sh)
python -m LHM.launch infer.human_lrm --config configs/inference/<config>.yaml
```

## Model Variants

| Model | Layers | Min VRAM | Notes |
|-------|--------|----------|-------|
| LHM-MINI | 2 | 16 GB | Fastest |
| LHM-500M | 5 | 18 GB | Medium |
| LHM-500M-HF | 5 | 18 GB | Half-body capable |
| LHM-1B | 15 | 22 GB | Highest quality |
| LHM-1B-HF | 15 | 22 GB | Half-body capable |

Model weights are auto-downloaded from HuggingFace on first run via `utils/model_download_utils.py`.

## Architecture

### Core Pipeline

1. **Input** → image preprocessing (background removal via BiRefNet, segmentation via SAM2)
2. **Encoding** → DINO + Sapiens dual-encoder extracts image features
3. **Decoding** → Transformer decoder queries 3D Gaussian parameters anchored to SMPL-X body
4. **Rendering** → 3D Gaussian Splatting renders novel views / animations

### Key Modules

**`LHM/models/`** — Neural architectures
- `modeling_human_lrm.py` — Main model classes (`ModelHumanLRM`, `ModelHumanLRMSapdinoBodyHeadSD3_5`)
- `encoders/` — DINO and Sapiens vision encoders
- `transformer.py` / `transformer_dit.py` — Transformer decoder variants
- `rendering/` — GS renderer, triplane synthesis, SMPL-X Gaussian avatar

**`LHM/runners/infer/human_lrm.py`** — Inference orchestration; loads model, processes inputs, drives rendering
- `utils.py` in same dir handles motion preparation and image pre/post-processing

**`LHM/launch.py`** — Registry-based CLI entry point; maps runner names (e.g. `infer.human_lrm`) to runner classes via `REGISTRY_RUNNERS`

**`engine/`** — External processing pipelines (not part of the LHM module)
- `pose_estimation/` — Video → SMPL-X pose parameters using ViTPose + YOLO
- `SegmentAPI/` — SAM2-based segmentation
- `BiRefNet/` — Background removal

**`configs/`** — YAML configs for training and inference; inference configs live in `configs/inference/`

### Data Flow for Animation

```
Input image → Preprocess (segment/background) → Encode (DINO+Sapiens)
→ Transformer decoder → SMPL-X Gaussian avatar
→ Apply motion sequence (SMPL-X poses) → GS renderer → Output video
```

### Pretrained Models Layout

`pretrained_models/` (auto-populated on first run):
- `huggingface/` — LHM model weights by variant
- `sam2/`, `sapiens/` — Encoder checkpoints
- `human_model_files/` — SMPL-X body model files
- `dense_sample_points/` — Query point initialization for GS avatar

## Licenses

- Code: Apache 2.0
- Model weights: CC-BY-NC 4.0 (non-commercial only)
