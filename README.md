# UniWAM: Unified World-Action Model (OLA-SEM)

This repository is the official implementation of UniWAM on RoboTwin. It
employs three expert MoT models to jointly supervise physical language
understanding, image generation, and action prediction.



## Contents

- `models/`, `train/`, `utils/`, `bak/wan/`: physical language model and training runtime.
- `data/robotwin2/`: RoboTwin loader and conversion utilities (code only).
- `configs/`: RoboTwin physical language, IDM, history-flow, and future-noise examples.
- `inference/robotwin/uniwam/`: self-contained RoboTwin policy deployment.

Bridge, DROID, Fractal, and real-world inference are intentionally out
of scope for this release.

## Installation

Python 3.10 and a CUDA-capable PyTorch installation are recommended. Install
PyTorch for your CUDA version first, then install the remaining dependencies:

```bash
conda create -n ola-sem python=3.10 -y
conda activate ola-sem
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```

## Model weights

### Pretrained assets for training

Standard training uses the following pretrained assets:

- [Kosmos524/d0_v on ModelScope](https://www.modelscope.cn/models/Kosmos524/d0_v/files): initialization weights supervised on mixed robot datasets. Use these weights to initialize training or fine-tuning; they are not the released RoboTwin evaluation checkpoint.
- [Wan-AI/Wan2.2-TI2V-5B on Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B): Wan video backbone, VAE, and UMT5 components.
- [Qwen/Qwen3-VL-2B-Instruct on Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct): vision-language backbone.

Keep the downloaded directory structure as follows so that it matches the
default paths in the training configs:

```text
pretrained_models/
├── d0_v/
├── Qwen3-VL-2B-Instruct/
└── Wan2.2-TI2V-5B/
    └── Wan2.2_VAE.pth
```

`d0_v` is the normal training initialization. Point the training
fine-tuning checkpoint setting at `pretrained_models/d0_v/`.

## Post-training

### RoboTwin

See the [RoboTwin post-training guide](docs/robotwin/README.md)
for data preparation, training, and inference.

### LIBERO

Coming soon.

## Acknowledgements

This repository is based on and modified from
[Motus](https://github.com/thu-ml/Motus). We sincerely thank the Motus authors
for their valuable open-source work and contribution to the community.
