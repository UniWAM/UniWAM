"""Generate one global raw-OSC min/max file across the four LIBERO suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pyarrow.parquet as pq

from data.libero.action_normalization import ACTION_DIM, EXPECTED_MASK


EXPECTED_SUITES = {"libero_10", "libero_goal", "libero_object", "libero_spatial"}


def suite_name_from_root(root: Path) -> str:
    return root.name.split("_no_noops", 1)[0]


def discover_suite_roots(dataset_dir: Path) -> List[Path]:
    roots = sorted(
        path for path in dataset_dir.iterdir()
        if path.is_dir() and (path / "meta" / "info.json").exists()
    )
    suites = {suite_name_from_root(path) for path in roots}
    if suites != EXPECTED_SUITES:
        raise ValueError(
            f"Expected exactly the four standard LIBERO suites, got {sorted(suites)} "
            f"under {dataset_dir}"
        )
    return roots


def compute_global_action_stats(dataset_dir: Path) -> dict:
    minimum = np.full(ACTION_DIM, np.inf, dtype=np.float64)
    maximum = np.full(ACTION_DIM, -np.inf, dtype=np.float64)
    sample_count = 0
    roots = discover_suite_roots(dataset_dir)

    for root in roots:
        parquet_files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No episode parquet files found in {root}")
        for parquet_path in parquet_files:
            table = pq.read_table(parquet_path, columns=["action"])
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                raise ValueError(f"Expected action [T,7], got {actions.shape} in {parquet_path}")
            if not np.all(np.isfinite(actions)):
                raise ValueError(f"Non-finite action in {parquet_path}")
            minimum = np.minimum(minimum, actions.min(axis=0))
            maximum = np.maximum(maximum, actions.max(axis=0))
            sample_count += actions.shape[0]

    if sample_count == 0 or np.any(maximum[EXPECTED_MASK] <= minimum[EXPECTED_MASK]):
        raise ValueError("Could not compute valid global continuous action ranges")
    return {
        "format_version": 1,
        "action_representation": "raw_osc",
        "normalization": "min_max",
        "normalized_range": [-1.0, 1.0],
        "action_dim": ACTION_DIM,
        "mask": EXPECTED_MASK.tolist(),
        "suites": sorted(EXPECTED_SUITES),
        "num_action_rows": sample_count,
        "min": minimum.tolist(),
        "max": maximum.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    stats = compute_global_action_stats(dataset_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {stats['num_action_rows']} raw-OSC rows from 4 suites to {output}")


if __name__ == "__main__":
    main()
