# Main App Dependency Analysis

## Entry Point: `main_app.py`

### Direct Dependencies
- `unik3d_viewer.main_viewer.LiveViewerWindow`

### Traced Dependency Tree

#### 1. `unik3d_viewer/main_viewer.py` imports:
- **External**: pyglet, numpy, torch, cv2, imgui, tkinter, pytorch_wavelets
- **Core UniK3D**: `unik3d.models.UniK3D`
- **Local viewer modules**:
  - `.config` (settings management)
  - `.camera_handler` (Camera class)
  - `.inference_logic` (inference_thread_func)
  - `.rendering` (Renderer class)
  - `.ui` (UIManager class)

#### 2. `unik3d_viewer` local dependencies:
- **config.py** ✅ USED
- **camera_handler.py** ✅ USED  
- **inference_logic.py** ✅ USED
  - imports: `unik3d.models.UniK3D`, `.file_io`
- **rendering.py** ✅ USED
  - imports: `.shaders`
- **ui.py** ✅ USED
  - imports: `.gl_utils`
- **file_io.py** ✅ USED (by inference_logic)
- **gl_utils.py** ✅ USED (by ui)
- **shaders.py** ✅ USED (by rendering)

#### 3. Core `unik3d` dependencies (traced from UniK3D model):

**Used by Live App:**
- `unik3d/models/` ✅ (UniK3D, decoder, encoder, backbones)
- `unik3d/layers/` ✅ (MLP, AttentionBlock, etc.)  
- `unik3d/utils/camera.py` ✅ (Camera classes)
- `unik3d/utils/coordinate.py` ✅ (coords_grid)
- `unik3d/utils/geometric.py` ✅ (flat_interpolate)
- `unik3d/utils/misc.py` ✅ (get_params, last_stack, etc.)
- `unik3d/utils/positional_embedding.py` ✅ (generate_fourier_features)
- `unik3d/utils/sht.py` ✅ (rsh_cart_3)
- `unik3d/utils/constants.py` ✅ (IMAGENET_DATASET_MEAN, etc.)
- `unik3d/ops/` ✅ (some components used by models)

## 🚮 **Files NOT Used by Main App**

### 1. Training/Evaluation Scripts
- ❌ `scripts/train.py` (24KB) - ML training
- ❌ `scripts/eval.py` (6.5KB) - Model evaluation  
- ❌ `scripts/infer.py` (3.8KB) - Batch inference
- ❌ `scripts/demo.py` (4.5KB) - Static demo script

### 2. Training/Evaluation Configs
- ❌ `configs/train/` - Training configurations
- ❌ `configs/eval/` - Evaluation configurations

### 3. Dataset Loaders (Training Only)
- ❌ `unik3d/datasets/` - All dataset classes (training/eval only)
  - arkit.py, kitti.py, nyuv2.py, hypersim.py, etc. (~50+ files)

### 4. Export/Training Utilities  
- ❌ `unik3d/models/export.py` - ONNX export
- ❌ `unik3d/models/camera_augmenter.py` - Training augmentation
- ❌ `unik3d/utils/validation.py` - Validation metrics
- ❌ `unik3d/utils/evaluation_depth.py` - Evaluation utilities
- ❌ `unik3d/utils/visualization.py` - Training visualization  
- ❌ `unik3d/utils/pose.py` - Pose utilities
- ❌ `unik3d/utils/distributed.py` - Multi-GPU training
- ❌ `unik3d/utils/ema_torch.py` - Exponential moving averages
- ❌ `unik3d/ops/scheduler.py` - Learning rate scheduling
- ❌ `unik3d/ops/losses/` - Training loss functions

### 5. Other Demos/Apps (Already Removed)
- ❌ `live_slam_viewer.py` - Keep as requested but it's similar to main_app
- ❌ `play_3d_video.py` - Video player demo

### 6. Build Artifacts
- ❌ `recording_output/` - Runtime output directory
- ❌ `video_outputs/` - Runtime output directory  
- ❌ `docs/` - Documentation
- ❌ `assets/` - Demo assets

## ✅ **Essential Files for Main App**

### Core Application
- ✅ `main_app.py` - Main entry point
- ✅ `live_slam_viewer.py` - Alternative entry (keep as requested)
- ✅ `unik3d_viewer/` - Complete viewer package (all files used)

### Core Model
- ✅ `unik3d/models/` - Neural network components
- ✅ `unik3d/layers/` - Neural network layers
- ✅ `unik3d/utils/` (subset) - Camera, geometry, misc utilities  
- ✅ `unik3d/ops/` (subset) - Operations used by models
- ✅ `unik3d/__init__.py` - Package initialization

### Configuration
- ✅ `requirements.txt` - Dependencies
- ✅ `pyproject.toml` - Package metadata
- ✅ `hubconf.py` - HuggingFace Hub integration
- ✅ `configs/config_*.json` - Model configurations (needed for model loading)

## 📊 **Estimated Space Savings**

- Training scripts: ~38KB
- Dataset loaders: ~200KB+ (50+ files)
- Training utilities: ~100KB+
- Evaluation utilities: ~50KB+
- Config directories: Variable
- **Total estimated**: ~400KB+ of code cleanup

## 🎯 **Cleanup Recommendation**

**Phase 1 - Safe Removals:**
1. Remove `scripts/` directory (training/eval only)
2. Remove `configs/train/` and `configs/eval/` 
3. Remove `unik3d/datasets/` directory (training only)

**Phase 2 - Targeted Removals:**
4. Remove training-specific utilities from `unik3d/utils/`
5. Remove training-specific ops from `unik3d/ops/`
6. Remove unused model files from `unik3d/models/`

**Phase 3 - Final Cleanup:**
7. Remove demo assets and output directories
8. Clean up documentation if not needed 