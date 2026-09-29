# RoboTwin Post-training

[Back to the main README](../../README.md)

Run the commands below from the repository root.

## RoboTwin data

No data is checked into this repository. The default training path is
`data/robotwin_dataset`. Prepare it with the Python utilities documented in
[`data/robotwin2/robotwin_data_convert/README.md`](../../data/robotwin2/robotwin_data_convert/README.md),
or point `dataset.dataset_dir` in a config at an existing converted dataset.

The loader expects `clean/` and/or `randomized/` task directories containing
videos, qpos tensors, instruction metadata, and optional cached language
features. The converter README contains the concrete tree.

## RoboTwin training

Edit one of the YAML files in `configs/` to set the dataset and model paths,
then run:

```bash
CONFIG_FILE=configs/robotwin_lap.yaml \
NUM_GPUS=8 \
bash scripts/train_lap.sh
```

Useful overrides are `DEEPSPEED_CONFIG`, `OUTPUT_DIR`, `RUN_NAME`,
`MASTER_ADDR`, `MASTER_PORT`, and `REPORT_TO`. Available examples include:

- `robotwin_lap.yaml`: standard physical language training.
- `robotwin_lap_clean.yaml`: clean split.
- `robotwin_lap_history_flow*.yaml`: executed-qpos history as action source.
- `robotwin_lap_future_noise.yaml`: history initialization with future-video
  noise augmentation.

Set `resume.checkpoint_path` for a full Accelerator/DeepSpeed state, or
`finetune.checkpoint_path` for model-weight initialization. Training writes a
`config.json` beside exported model weights. History-flow inference requires
that metadata and validates `flow_source.mode`, `video_mode`, and
`history_length` against the action chunk size.

## RoboTwin inference

### Post-trained checkpoint for evaluation

Use the released
[Kosmos524/ola_sem checkpoint on ModelScope](https://www.modelscope.cn/models/Kosmos524/ola_sem/files)
for RoboTwin evaluation.

See the [RoboTwin inference guide](../../inference/robotwin/uniwam/README.md) for setup
and configuration.
