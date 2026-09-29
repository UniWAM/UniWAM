# UniWAM Policy Evaluation on RoboTwin

This guide explains how to deploy and evaluate UniWAM (OLA-SEM) on RoboTwin 2.0.
The policy directory and RoboTwin policy name are `uniwam`. The underlying
Motus model classes are retained, and checkpoint loading is unchanged.

## Environment

Install RoboTwin 2.0 by following its official installation guide. A separate
RoboTwin environment is recommended because simulator and training dependencies
may differ.

After activating the RoboTwin environment, install the policy dependencies:

```bash
cd /path/to/OLA-Sem/inference/robotwin/uniwam
pip install -r requirements.txt
```

## Deployment

Copy the complete policy directory into RoboTwin:

```bash
cp -a /path/to/OLA-Sem/inference/robotwin/uniwam \
  /path/to/RoboTwin/policy/
cd /path/to/RoboTwin/policy/uniwam
cp paths_config.example.yml paths_config.yml
```

The deployed layout should include:

```text
RoboTwin/
├── policy/uniwam/
│   ├── deploy_policy.py
│   ├── deploy_policy.yml
│   ├── eval.sh
│   ├── auto_eval.sh
│   ├── paths_config.yml
│   ├── tasks_all.txt
│   ├── models/
│   ├── utils/
│   └── bak/wan/
└── script/eval_policy.py
```

## Configuration

Edit `paths_config.yml` before evaluation:

```yaml
robotwin_root: "/path/to/RoboTwin"
conda_env: "/path/to/conda/envs/RoboTwin"
checkpoint_path: "/path/to/checkpoint_step_N/pytorch_model"
wan_path: "/path/to/Wan2.2-TI2V-5B"
vlm_path: "/path/to/Qwen3-VL-2B-Instruct"

gpu_ids: []
task_config: "demo_randomized"
seed: 42
tasks_file: "tasks_all.txt"
instruction_type: "unseen"
test_num: 100

inference_mode: "history_flow"
num_inference_timesteps: 4
history_action_noise_std: 0.02
future_video_denoise_fraction: 1.0
```

Required paths:

- `robotwin_root`: the RoboTwin repository root.
- `checkpoint_path`: the exported `pytorch_model/` directory containing
  `mp_rank_00_model_states.pt`.
- `wan_path`: Wan2.2 directory containing the VAE, T5 weights, and tokenizer.
- `vlm_path`: Qwen3-VL directory used to load its configuration and processor.
- `conda_env`: RoboTwin environment path or name. Leave it empty if the correct
  environment is already active.

`gpu_ids: []` enables `nvidia-smi` auto-detection. Set an explicit list such as
`[0, 1, 2, 3]` when needed. Batch evaluation assigns one task to each GPU at a
time.

## Inference modes

### Legacy

```yaml
inference_mode: "legacy"
```

Legacy mode uses Gaussian flow sources and supports checkpoints created before
history-flow training was introduced.

### History flow

```yaml
inference_mode: "history_flow"
```

History-flow mode conditions the next prediction on qpos values actually
executed by the simulator. The checkpoint layout must be:

```text
checkpoint_step_N/
├── config.json
└── pytorch_model/
    └── mp_rank_00_model_states.pt
```

The adjacent `config.json` must contain `flow_source.mode: "history"` and a
`history_length` equal to the action chunk size. Its `video_mode` may be
`gaussian` or `history`. `history_action_noise_std` overrides the checkpoint
noise value and must be non-negative.

`num_inference_timesteps` must be positive.
`future_video_denoise_fraction` must be between `0.0` and `1.0`; `1.0` denoises
the future-video branch throughout the full inference schedule.

## Running evaluation

Evaluate one task:

```bash
cd /path/to/RoboTwin/policy/uniwam
bash eval.sh hanging_mug
```

Evaluate all tasks listed in `tasks_all.txt` across the configured GPUs:

```bash
bash auto_eval.sh
```

To use another configuration file:

```bash
CONFIG_FILE=/absolute/path/to/paths_config.yml bash eval.sh hanging_mug
CONFIG_FILE=/absolute/path/to/paths_config.yml bash auto_eval.sh
```

Common values can also be overridden without editing YAML:

```bash
GPU_ID=1 TEST_NUM=10 NUM_INFERENCE_TIMESTEPS=4 bash eval.sh hanging_mug
```

Other supported overrides include `ROBOTWIN_ROOT`, `CONDA_ENV`,
`CHECKPOINT_PATH`, `WAN_PATH`, `VLM_PATH`, `TASK_CONFIG`, `SEED`,
`INSTRUCTION_TYPE`, `INFERENCE_MODE`, `HISTORY_ACTION_NOISE_STD`,
`FUTURE_VIDEO_DENOISE_FRACTION`, `TASKS_FILE`, and `GPU_IDS`.

## Logs and troubleshooting

Each run writes logs to `logs_YYYYMMDD_HHMMSS/` under the policy directory
unless `LOG_DIR` is set.
