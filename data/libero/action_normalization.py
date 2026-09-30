"""Shared raw-OSC action normalization for LIBERO training and inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


ACTION_DIM = 7
CONTINUOUS_ACTION_DIMS = 6
EXPECTED_MASK = np.asarray([True] * CONTINUOUS_ACTION_DIMS + [False], dtype=bool)


@dataclass(frozen=True)
class RawOscActionNormalizer:
    minimum: np.ndarray
    maximum: np.ndarray
    mask: np.ndarray

    @classmethod
    def from_file(cls, path: str | Path) -> "RawOscActionNormalizer":
        stats_path = Path(path).expanduser().resolve()
        if not stats_path.exists():
            raise FileNotFoundError(
                f"LIBERO raw-OSC statistics not found: {stats_path}. "
                "Run python -m data.libero.generate_action_stats first."
            )
        with stats_path.open("r", encoding="utf-8") as handle:
            payload: Dict[str, Any] = json.load(handle)

        if payload.get("action_representation") != "raw_osc":
            raise ValueError(f"Expected raw_osc statistics in {stats_path}")
        if payload.get("normalization") != "min_max":
            raise ValueError(f"Expected min_max statistics in {stats_path}")

        minimum = np.asarray(payload.get("min"), dtype=np.float32)
        maximum = np.asarray(payload.get("max"), dtype=np.float32)
        mask = np.asarray(payload.get("mask"), dtype=bool)
        if minimum.shape != (ACTION_DIM,) or maximum.shape != (ACTION_DIM,):
            raise ValueError(f"Expected 7D min/max in {stats_path}")
        if mask.shape != (ACTION_DIM,) or not np.array_equal(mask, EXPECTED_MASK):
            raise ValueError(f"Expected mask [true x6, false] in {stats_path}")
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise ValueError(f"Non-finite action statistics in {stats_path}")
        if np.any(maximum[mask] <= minimum[mask]):
            raise ValueError(f"Degenerate continuous action range in {stats_path}")
        return cls(minimum=minimum, maximum=maximum, mask=mask)

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        values = self._validate_actions(actions).copy()
        values[..., self.mask] = (
            2.0
            * (values[..., self.mask] - self.minimum[self.mask])
            / (self.maximum[self.mask] - self.minimum[self.mask])
            - 1.0
        )
        return values

    def denormalize(self, actions: np.ndarray, clip: bool = True) -> np.ndarray:
        values = self._validate_actions(actions).copy()
        normalized = values[..., self.mask]
        if clip:
            normalized = np.clip(normalized, -1.0, 1.0)
        values[..., self.mask] = (
            0.5
            * (normalized + 1.0)
            * (self.maximum[self.mask] - self.minimum[self.mask])
            + self.minimum[self.mask]
        )
        values[..., -1] = (values[..., -1] > 0.5).astype(np.float32)
        return values

    @staticmethod
    def _validate_actions(actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim < 1 or values.shape[-1] != ACTION_DIM:
            raise ValueError(f"Expected actions [...,{ACTION_DIM}], got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("Actions contain non-finite values")
        return values
