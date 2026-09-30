#!/usr/bin/env python3
"""Generate per-timestep LAP text for LeRobot LIBERO raw OSC actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from data.libero.libero_dataset import load_jsonl, suite_name_from_root
from data.robotwin2.robotwin_data_convert.robotwin_generate_language_action import summarize_numeric_actions


def raw_osc_window_to_lap(actions: np.ndarray) -> str:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected raw OSC [T,7], got {actions.shape}")
    scaled = actions.copy()
    scaled[:, :3] *= 0.05
    scaled[:, 3:6] *= 0.5
    text = summarize_numeric_actions(
        scaled,
        sum_decimal="0f",
        include_rotation=True,
        rotation_precision=10,
        include_gripper_action=True,
    )
    if not text:
        return "keep the end effector still"
    return text


def generate(
    dataset_dir: Path,
    cache_dir: Path,
    horizon: int,
    overwrite: bool,
    lap_subdir: str = "lap",
) -> None:
    suite_roots = sorted(
        path for path in dataset_dir.iterdir()
        if path.is_dir() and (path / "meta" / "info.json").exists()
    )
    for root in suite_roots:
        with (root / "meta" / "info.json").open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        suite = suite_name_from_root(root)
        out_dir = cache_dir / lap_subdir / suite
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = load_jsonl(root / "meta" / "episodes.jsonl")
        for row in rows:
            episode_index = int(row["episode_index"])
            output_path = out_dir / f"episode_{episode_index:06d}.txt"
            if output_path.exists() and not overwrite:
                continue
            episode_chunk = episode_index // int(info.get("chunks_size", 1000))
            parquet_path = root / "data" / f"chunk-{episode_chunk:03d}" / f"episode_{episode_index:06d}.parquet"
            table = pq.read_table(parquet_path, columns=["action"])
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            lines = [
                raw_osc_window_to_lap(actions[t : min(t + horizon, len(actions))])
                for t in range(len(actions))
            ]
            temporary = output_path.with_suffix(".txt.tmp")
            temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
            temporary.replace(output_path)
        print(f"Generated LAP cache for {suite}: {len(rows)} episodes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--lap-subdir", default="lap")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.horizon <= 0:
        raise ValueError("LIBERO LAP horizon must be positive")
    lap_subdir = Path(args.lap_subdir)
    if lap_subdir.is_absolute() or ".." in lap_subdir.parts:
        raise ValueError("--lap-subdir must stay within the cache directory")
    generate(
        args.dataset_dir.resolve(),
        args.cache_dir.resolve(),
        args.horizon,
        args.overwrite,
        args.lap_subdir,
    )


if __name__ == "__main__":
    main()
