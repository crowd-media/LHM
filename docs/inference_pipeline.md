# LHM Inference Pipeline

This document describes the end-to-end inference pipeline for the Large Human Model (LHM): from a raw input image and motion sequence to a rendered animation video.

---

## Overview

The pipeline has seven stages:

```
Input Image + Motion Sequence
        │
        ▼
1. Image Preprocessing          (segmentation, crop, normalize)
        │
        ▼
2. Body Pose & Shape Estimation (SMPL-X params from image)
        │
        ▼
3. Feature Extraction           (DINOv2 + Sapiens dual encoder)
        │
        ▼
4. Transformer Decoding         (query points → latent features)
        │
        ▼
5. Gaussian Splatting Decode    (latent → 3D Gaussian attributes)
        │
        ▼
6. Motion Sequence Loading      (per-frame SMPL-X pose params)
        │
        ▼
7. Animated Rendering           (animate Gaussians → frames → video)
```

**Entry point:** `LHM/launch.py` dispatches to `HumanLRMInferrer` (registered as `"infer.human_lrm"` in `REGISTRY_RUNNERS`).

**CLI invocation** (used by `inference.sh`):
```bash
python -m LHM.launch infer.human_lrm \
    model_name=$MODEL_NAME \
    image_input=$IMAGE_PATH \
    motion_seqs_dir=$MOTION_SEQS_DIR
```

---

## Stage 1 — Image Preprocessing

**Location:** `LHM/runners/infer/human_lrm.py` → `infer_preprocess_image()`

### Background Removal

Two options, tried in order:
1. **SAM2** (`engine/SegmentAPI/SAM.py`) — semantic segmentation, preferred
2. **rembg** (U²-Net fallback) — if SAM2 is unavailable

Output: binary mask `[H, W]` in `[0, 1]`.

### Crop & Resize

```
Raw image [H, W, 3]
    → Extract bounding box from mask (scaled ×1.1 for context)
    → Pad to 5:3 aspect ratio (white fill)
    → resize_image_keepaspect_np() to max_tgt_size=896
    → center_crop_according_to_mask() to enlarge human region
    → Final resize to render_tgt_size, rounded to nearest multiple of 14
Output: [1, 3, 1024, 1024]  (or nearest multiple-of-14 resolution)
```

The multiple-of-14 constraint aligns with DINOv2-ViT-14 patch size.

### Alpha Compositing

```python
rgb = rgb * mask + bg_color * (1 - mask)   # white background
```

### Camera Intrinsics

`intr` is an `[4, 4]` matrix updated at each crop/resize step to remain consistent with the final pixel coordinate frame.

---

## Stage 2 — Body Pose & Shape Estimation

**Location:** `LHM/runners/infer/human_lrm.py` → `HumanLRMInferrer.infer()`

### Pose Estimator

`PoseEstimator` wraps ViTPose + YOLO to extract SMPL-X parameters from the input image:

```python
shape_pose = self.pose_estimator(image_path)
# shape_pose.beta        : [10]   SMPL-X shape coefficients
# shape_pose.ratio       : float  body/image ratio (must be > 0.4)
# shape_pose.is_full_body: bool
```

### Face Detection & Crop

`FaceDetector` (VGG-based) crops a `[112, 112]` head region used by the head encoder branch:

```python
src_head_rgb = self.crop_face_image(image_path)
# Shape after normalization: [1, 3, 112, 112]  in [0, 1]
```

---

## Stage 3 — Feature Extraction (Encoder)

**Location:** `LHM/models/modeling_human_lrm.py` → `forward_encode_image()`

### Dual-Encoder Architecture (LHM-1B / LHM-500M)

| Branch | Model | Input | Output |
|--------|-------|-------|--------|
| Body   | Fine-tuned DINOv2-ViT-L-14 | `[B, 3, 1024, 1024]` | `[B, 1024, H_f, W_f]` |
| Detail | Sapiens-1B | `[B, 3, 1024, 1024]` | `[B, 1536, H_f, W_f]` |
| Head   | DINOv2 (small) | `[B, 3, 112, 112]` | `[B, 1536, H_h, W_h]` |

Features from the two body branches are fused (`encoder_type: dinov2_fusion`), producing a combined `image_feats: [B, 1536, H_f, W_f]`.

---

## Stage 4 — Transformer Decoding

**Location:** `LHM/models/modeling_human_lrm.py` → `forward_latent_points()`

### Query Points

Dense point samples on the SMPL-X body surface in canonical pose (`latent_query_points_type: e2e_smplx_sub1`):

```python
query_points, smplx_params = self.renderer.get_query_points(smplx_params, device)
# query_points: [1, Np, 3]   Np ≈ 40 000 (dense subdivision)
```

### Motion Tokens

Body encoder features are projected to modulation tokens via MLP:

```python
motion_tokens = self.forward_moitonembed(body_feats)
# Shape: [B, 2 * pcl_dim]  — used as scale/shift in transformer layers
```

### Transformer Forward

```python
tokens = self.transformer(
    x=query_point_embeddings,   # [B, Np, 1024]
    cond=image_feats,           # [B, 1536, H_f, W_f]
    temb=motion_tokens          # modulation signal
)
# Output: [B, Np, 1024]
```

Config (`sd3_mm_bh_cond` variant, LHM-1B): `transformer_layers=15`, `transformer_dim=1024`, `transformer_heads=16`. Gradient checkpointing is enabled by default.

---

## Stage 5 — 3D Gaussian Splatting Decode

**Location:** `LHM/models/rendering/gs_renderer.py` → `GSLayer`, `forward_gs()`

`GSLayer` maps transformer output tokens to per-Gaussian attributes:

| Attribute | Shape per point | Description |
|-----------|----------------|-------------|
| `xyz`      | `[3]`           | Position offset from canonical |
| `opacity`  | `[1]`           | Alpha, after sigmoid |
| `scaling`  | `[3]`           | Log-scale axis lengths |
| `rotation` | `[4]`           | Unit quaternion |
| `shs`      | `[3 × (deg+1)²]`| Spherical harmonics (degree 3) |

Decoded attributes are wrapped in a `GaussianModel` object. This model can be exported to `.ply` via `GaussianModel.save_ply()`.

---

## Stage 6 — Motion Sequence Loading

**Location:** `LHM/runners/infer/utils.py` → `prepare_motion_seqs()`

Motion sequences live in a directory of per-frame JSON files:

```json
{
  "root_pose":  [[rx, ry, rz]],
  "body_pose":  [[...21 joints × 3 axis-angle...]],
  "betas":      [[...10 shape coefficients...]],
  "trans":      [[tx, ty, tz]],
  "jaw_pose":   [[jx, jy, jz]],
  "leye_pose":  [[...]], "reye_pose": [[...]],
  "lhand_pose": [[...15 × 3...]], "rhand_pose": [[...15 × 3...]],
  "expr":       [[...100 expression coefficients...]]
}
```

`prepare_motion_seqs()` stacks all frames into tensors:

```python
{
    "render_c2ws":      [1, Nframes, 4, 4],   # camera poses
    "render_intrs":     [1, Nframes, 4, 4],   # camera intrinsics
    "render_bg_colors": [1, Nframes, 3],       # white
    "smplx_params": {
        "betas":      [1, 10],
        "root_pose":  [1, Nframes, 1, 3],
        "body_pose":  [1, Nframes, 21, 3],
        "jaw_pose":   [1, Nframes, 1, 3],
        "leye_pose":  [1, Nframes, 1, 3],
        "reye_pose":  [1, Nframes, 1, 3],
        "lhand_pose": [1, Nframes, 15, 3],
        "rhand_pose": [1, Nframes, 15, 3],
        "trans":      [1, Nframes, 3],
        "expr":       [1, Nframes, 100],
        "focal":      [1, Nframes, 2],
        "princpt":    [1, Nframes, 2],
        "img_size_wh":[1, Nframes, 2]
    }
}
```

Custom motion can be extracted from video using:
```bash
python engine/pose_estimation/video2motion.py \
    --video_path <input.mp4> \
    --output_path <output_dir>
```

---

## Stage 7 — Animated Rendering

**Location:** `LHM/models/modeling_human_lrm.py` → `animation_infer()`, `LHM/models/rendering/gs_renderer.py` → `forward_animate_gs()`

### Per-Frame Animation Loop

Frames are rendered in batches of 40 to manage GPU memory:

```python
for batch_i in range(0, Nframes, batch_size=40):
    # 1. Deform canonical Gaussians to target pose
    animate_gs_model = self.animate_gs_model(
        gs_attr, query_points, smplx_params[batch_i]
    )
    # 2. Rasterize
    rendered = rasterizer(
        means3D=gs.xyz,
        scales=gs.scaling,
        rotations=gs.rotation,
        shs=gs.shs,
        opacities=gs.opacity,
    )
```

### Rasterization

Uses `diff_gaussian_rasterization`. Rendering settings per camera:

| Setting | Value |
|---------|-------|
| znear | 0.01 |
| zfar  | 100.0 |
| background | white `[1, 1, 1]` |
| sh_degree | 3 |

Output per frame:
```python
{
    "comp_rgb":   [H, W, 3],   # float32, [0, 1]
    "comp_mask":  [H, W, 1],   # alpha
    "comp_depth": [H, W, 1]    # depth
}
```

### Video Assembly

Frames are concatenated `[Nframes, H, W, 3]`, clamped to `[0, 1]`, cast to `uint8`, and encoded at 30 fps:

```python
images_to_video(rgb_frames, output_path, fps=30)
```

---

## Key Configuration Parameters

Configs live in `configs/inference/human-lrm-{SIZE}.yaml`.

| Parameter | LHM-1B value | Description |
|-----------|-------------|-------------|
| `source_image_res` | 1024 | Input resolution |
| `render_image.high` | 512 | Render output resolution |
| `src_head_size` | 112 | Face crop size |
| `transformer_dim` | 1024 | Token width |
| `transformer_layers` | 15 | Transformer depth |
| `transformer_heads` | 16 | Attention heads |
| `gs_sh` | 3 | SH degree for appearance |
| `dense_sample_pts` | 40 000 | Number of Gaussians |
| `shape_param_dim` | 10 | SMPL-X shape PCA |
| `expr_param_dim` | 100 | Facial expression PCA |
| `smplx_type` | `smplx_2` | Body model variant |

---

## Data Flow Summary

```
Raw Image [H, W, 3]
    ──preprocess──►  [1, 3, 1024, 1024]
    ──encode──►      image_feats [1, 1536, 73, 73]
                     head_feats  [1, 1536, 8, 8]
    ──transformer──► latent [1, Np, 1024]    (Np ≈ 40 000)
    ──GSLayer──►     xyz      [1, Np, 3]
                     opacity  [1, Np, 1]
                     scaling  [1, Np, 3]
                     rotation [1, Np, 4]
                     shs      [1, Np, 48]
    ──per frame──►   animate → rasterize → RGB [H, W, 3]
    ──assemble──►    output.mp4 @ 30 fps
```

---

## Relevant Source Files

| File | Role |
|------|------|
| `LHM/launch.py` | CLI entry point, runner registry |
| `LHM/runners/infer/human_lrm.py` | Top-level inference orchestration |
| `LHM/runners/infer/utils.py` | Motion sequence loading, image helpers |
| `LHM/models/modeling_human_lrm.py` | Model forward passes (`infer_single_view`, `animation_infer`) |
| `LHM/models/encoders/` | DINOv2 and Sapiens encoder wrappers |
| `LHM/models/transformer.py` | Transformer decoder |
| `LHM/models/rendering/gs_renderer.py` | GSLayer, GaussianModel, rasterization |
| `LHM/models/rendering/smpl_x.py` | SMPL-X body model, point cloud sampling |
| `engine/SegmentAPI/SAM.py` | SAM2 background segmentation |
| `engine/pose_estimation/video2motion.py` | Video → SMPL-X pose extraction |
| `configs/inference/` | Per-model YAML configs |
