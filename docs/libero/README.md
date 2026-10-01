# LIBERO training and evaluation

This repository trains on the **original LIBERO** four-suite demonstrations,
using either the official HDF5 files converted below or a compatible LeRobot
v2.1 dataset. It evaluates a trained checkpoint on either
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) or
[LIBERO-plus](https://github.com/sylvestf/LIBERO-plus). No benchmark environment, dataset, pretrained model, or
checkpoint is bundled here.

## Install

Install this repository as described in the root README. Clone and install the
benchmark you want separately, following its own README and asset download
instructions. By default, the evaluation script looks for `../LIBERO` or
`../LIBERO-plus` beside this repository; `--libero-plus-root` overrides that.
Keep the benchmark's `assets`, `bddl_files`, and `init_files` directories in
the locations expected by that benchmark. The evaluation script generates a
LIBERO config in `outputs/.libero_configs/`, avoiding dependence on a user's
`~/.libero` configuration. A separate Python environment for the benchmark is
supported via `--libero-plus-python`.

## Original LIBERO training

For the [official LIBERO HDF5 demonstrations](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets), first download the four directories `libero_10`, `libero_goal`, `libero_object`, and `libero_spatial` using [LIBERO's download instructions](https://github.com/Lifelong-Robot-Learning/LIBERO#datasets). Convert them into a **new** directory (the converter refuses to overwrite an existing suite). This produces the minimal v2.1-style layout required by this repository's loader, not a fully populated general-purpose LeRobot release:

```bash
python -m scripts.libero.convert_official_libero \
  --source /path/to/official/LIBERO-datasets \
  --output /path/to/libero_official_lerobot
DATASET_DIR=/path/to/libero_official_lerobot \
  bash scripts/libero/train_libero.sh
```

For a quick check, pass `--suites libero_goal --max-episodes 2` and use a separate output directory. The official recorder stores each observation after executing the same-index action, so the converter pairs that observation with the **next** action and drops the last frame. It also flips the raw camera images by 180° to match evaluation, converts gripper `-1/+1` to policy `1/0` when needed, and preserves all recorded timesteps. It does **not** reproduce the no-op removal or exact episode selection of the IPEC-COMMUNITY release; training results from the two preparations are not directly identical. The converter needs `h5py`, `pyarrow`, and PyAV with H.264 encoding.

The converter was tested on two episodes from an official LIBERO HDF5 file: 174 output frames, with action/state alignment and video counts checked against the source. Run a small conversion on your own download before converting all four suites.

Alternatively, use an existing four-suite LeRobot v2.1 conversion under one parent:

```text
data/libero_dataset/
├── libero_10_no_noops*/
├── libero_goal_no_noops*/
├── libero_object_no_noops*/
└── libero_spatial_no_noops*/
```

The official converter above writes the same layout with the shorter child names `libero_10/`, `libero_goal/`, `libero_object/`, and `libero_spatial/`.

Each suite needs `meta/info.json`, `meta/tasks.jsonl`, `meta/episodes.jsonl`,
`data/chunk-*/episode_*.parquet`, and the two camera videos under
`videos/chunk-*/observation.images.image/` and
`videos/chunk-*/observation.images.wrist_image/`. This loader expects 20 Hz,
8D state, and 7D raw-OSC actions. It will reject incompatible datasets.
The original benchmark's environment assets alone are not a training dataset;
users must supply data in this format or use the official HDF5 converter above.

Download the Wan2.2-TI2V-5B and Qwen3-VL-2B-Instruct weights listed in the
root README to `pretrained_models/`. The optional UniWAM initialization
checkpoint is intentionally unset (`finetune.checkpoint_path: null`); set it
to your own checkpoint when available. Then:

```bash
DATASET_DIR=/path/to/libero_dataset \
  bash scripts/libero/prepare_libero_cache.sh
DATASET_DIR=/path/to/libero_dataset \
  bash scripts/libero/train_libero.sh
```

The training script also prepares any missing cache automatically; the first
run can take time. Set `CACHE_DIR` to move action statistics, LAP text, and T5
embeddings elsewhere. Set `NPROC_PER_NODE` to select the GPU count, `WAN_PATH`
to the parent directory containing `Wan2.2-TI2V-5B`, and `CONFIG_FILE` to use
another config. 

## Evaluation

Run from this repository root, with an existing checkpoint and its matching
config. A short smoke test is:

```bash
bash scripts/libero/eval_libero.sh \
  --checkpoint /path/to/UniWAM-libero/pytorch_model \
  --benchmark-mode original --task libero_goal --max-tasks 1 --num-trials 1 \
  --action-stats /path/to/UniWAM-libero/raw_osc_action_stats.json
```

For LIBERO-plus, use its own checkpoint and action statistics:

```bash
bash scripts/libero/eval_libero.sh \
  --checkpoint /path/to/UniWAM-libero-plus/pytorch_model \
  --benchmark-mode plus --task libero_goal --max-tasks 1 --num-trials 1 \
  --action-stats /path/to/UniWAM-libero-plus/raw_osc_action_stats.json
```

The two benchmarks share the policy server but use separately installed
environments, checkpoints, and normalization statistics.  Use `--task all` for all four suites. Original
LIBERO defaults to 50 trials per task; LIBERO-plus defaults to one. 
