# 📸 Wid3R: Wide Field-of-View 3D Reconstruction via Camera Model Conditioning

<p align="center">
  <a href="https://arxiv.org/abs/2602.05321"><img src="https://img.shields.io/badge/arXiv-2602.05321-b31b1b.svg" alt="Paper"></a>
  <a href="https://jdk9405.github.io/Wid3R/"><img src="https://img.shields.io/badge/Project-Page-f59e0b.svg" alt="Project Page"></a>
  <a href="https://github.com/jdk9405/Wid3R"><img src="https://img.shields.io/badge/GitHub-Wid3R-181717.svg" alt="GitHub"></a>
</p>

<p align="center">
  <b><a href="https://jdk9405.github.io/">Dongki Jung</a><sup>1</sup>, <a href="https://jh-choi.github.io/">Jaehoon Choi</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=YU1z_eEAAAAJ&amp;hl=en">Adil Qureshi</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=Pur3SOQAAAAJ&amp;hl">Somi Jeong</a><sup>2</sup>, <a href="https://scholar.google.com/citations?user=Nqawxa0AAAAJ&amp;hl">Dinesh Manocha</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=fkp6OZgAAAAJ&amp;hl">Suyong Yeon</a><sup>2</sup></b>
</p>

<p align="center">
  <sup>1</sup>University of Maryland, College Park &nbsp;&nbsp; <sup>2</sup>NAVER LABS
</p>


## Highlights

- **Wide-FoV reconstruction:** reconstructs geometry directly from fisheye and 360-degree imagery.
- **Camera model conditioning:** uses a camera model token to adapt reconstruction to different projection models.
- **Ray-based representation:** combines image features with ray geometry represented using spherical harmonics.
- **Feed-forward inference:** predicts dense point maps and camera poses from multiple input views in a single model pass.

## TODO

- [ ] Release the data preprocessing code.

## Overview

Wid3R is a feed-forward network for multi-view 3D reconstruction from wide field-of-view imagery. Unlike methods built around rectified pinhole inputs, Wid3R directly handles distorted wide-angle observations without explicit camera calibration or image undistortion. It supports wide field-of-view cameras through camera-conditioned geometry prediction. 


<p align="center">
  <img src="https://jdk9405.github.io/Wid3R/assets/main.png" alt="Wid3R architecture" width="100%">
</p>

## Installation

The provided Conda environment uses Python 3.10, PyTorch 2.3.1, CUDA 12.1 packages, and Gradio 5.45.0.

```bash
git clone https://github.com/jdk9405/Wid3R.git
cd Wid3R

conda env create -f environments.yml
conda activate wid3r
```

Verify that PyTorch can access the allocated GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Wid3R inference and distributed training require CUDA. Run the commands below from a GPU-enabled compute node or allocation.

## Interactive Demo

The Gradio application accepts a collection of images, reconstructs a colored 3D point cloud, and exports the result as GLB. The current interface provides fisheye and 360-degree camera modes.

Download the Wid3R checkpoint [`wid3r.bin`](https://drive.google.com/file/d/1N6nneTNLg-On_TRR1Yt2EwAEKtBd6sVf/view?usp=sharing) and save it under `pretrained_weights/`:

```text
pretrained_weights/wid3r.bin
```

Before launching, set `CKPT_DIR` in `demo_gradio.py` to a compatible Accelerate checkpoint:

```python
CKPT_DIR = "pretrained_weights/wid3r.bin"
```

Then run:

```bash
python demo_gradio.py
```

In the web interface:

1. Upload an ordered set of overlapping images, or select the bundled `360 Example`.
2. Choose the camera model that matches the input images.
3. Adjust the image sampling interval when needed.
4. Click **Reconstruct**.
5. Inspect the generated point cloud and camera trajectory in the 3D viewer.
6. Tune the confidence threshold or select an individual frame for visualization.


## Model Input and Output

The model receives a batch of image sequences and a corresponding batch of camera objects.

```text
images:  B x N x 3 x H x W, values in [0, 1]
cameras: B x N camera parameters represented by the selected camera model
```

The prediction dictionary contains the following principal values:

| Key | Description |
| --- | --- |
| `local_points` | Per-view camera-space point maps |
| `camera_poses` | Predicted camera-to-world transformations |
| `points` | Globally aligned 3D point maps |
| `uncertain` | Predicted per-point uncertainty used to derive confidence |
| `images` | Input colors used to render the reconstructed point cloud |

The demo converts uncertainty to confidence and uses the selected confidence threshold to remove unreliable geometry before GLB export.

## Training

Training is initialized from the [Pi3 pretrained model](https://huggingface.co/yyfz233/Pi3/tree/main). Download `model.safetensors` and save it under `pretrained_weights/`:

```bash
mkdir -p pretrained_weights
wget https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors \
  -O pretrained_weights/model.safetensors
```

Training is managed with Hydra and Hugging Face Accelerate. The example below uses one GPU process:

```bash
# number of GPUs
accelerate launch --num_processes 1 \
  --config_file configs/accelerate/ddp.yaml \
  --num_machines 1 \
  scripts/train_wid3r.py \
  train=train_wid3r \
  model.ckpt=pretrained_weights/model.safetensors \
  hydra/job_logging=custom \
  name=exp
```


The training mixture is configured in `configs/data/example.yaml` and includes [TartanAirV2](https://tartanair.org/), [ASE](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/), [Hypersim](https://github.com/apple/ml-hypersim), [KITTI-360](https://github.com/autonomousvision/kitti360Scripts), [Loc360](https://huajianup.github.io/research/360Loc/), [Matterport3D](https://niessner.github.io/Matterport/), [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/), [EDEN](https://github.com/lhoangan/eden-generation), and [Virtual KITTI 2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/). Update every `data_root` entry to match your local dataset layout before training.

Important training options are located in:

| Configuration | Purpose |
| --- | --- |
| `configs/default.yaml` | Hydra configuration composition |
| `configs/general/default.yaml` | output directories, logging, and random seed |
| `configs/data/example.yaml` | dataset mixture, paths, sampling, and augmentation |
| `configs/model/wid3r.yaml` | model architecture and optional pretrained checkpoint |
| `configs/train/train_wid3r.yaml` | resolution, sequence length, and learning rates |
| `configs/accelerate/ddp.yaml` | distributed execution and mixed precision |

## Evaluation

Evaluation utilities are grouped by task under `evaluation/`:

- `evaluation/mv_recon/`: multi-view reconstruction evaluation.
- `evaluation/monodepth/`: monocular depth inference and evaluation.
- `evaluation/relpose/`: relative camera pose evaluation.
- `evaluation/datasets/`: evaluation dataset adapters.

Dataset paths and checkpoint locations must be configured for the target environment before running these scripts.

## Troubleshooting

### Out of memory

Reduce `train.max_img_per_gpu`, narrow `train.image_num_range`, or lower the configured training resolution in `configs/train/train_wid3r.yaml`.

## Citation

If Wid3R is useful in your research, please cite:

```bibtex
@article{jung2026wid3r,
  title   = {Wid3R: Wide Field-of-View 3D Reconstruction via Camera Model Conditioning},
  author  = {Jung, Dongki and Choi, Jaehoon and Qureshi, Adil and Jeong, Somi and Manocha, Dinesh and Yeon, Suyong},
  journal = {arXiv preprint arXiv:2602.05321},
  year    = {2026}
}
```

## Acknowledgements

This repository builds on ideas and open-source components from the broader feed-forward visual geometry ecosystem, including [Pi3](https://github.com/yyfz/Pi3), [VGGT](https://github.com/facebookresearch/vggt), [DUSt3R](https://github.com/naver/dust3r), and related multi-view reconstruction projects.
