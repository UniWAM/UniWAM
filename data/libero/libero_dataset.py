"""Motus dataset adapter for local LeRobot v2.1 LIBERO data."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from data.libero.action_normalization import RawOscActionNormalizer
from data.libero.image_utils import compose_libero_views, load_libero_video_frames
from data.utils.image_utils import tensor_to_pil
from utils.vlm_utils import preprocess_vlm_messages_lap

try:
    from transformers import AutoProcessor
except Exception:  # pragma: no cover
    AutoProcessor = None

logger = logging.getLogger(__name__)

THIRD_PERSON_KEY = "observation.images.image"
WRIST_KEY = "observation.images.wrist_image"


def suite_name_from_root(root: Path) -> str:
    return root.name.split("_no_noops", 1)[0]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class Episode:
    suite: str
    root: Path
    episode_index: int
    length: int
    task_index: int
    instruction: str
    chunk_size: int

    @property
    def episode_chunk(self) -> int:
        return self.episode_index // self.chunk_size

    @property
    def parquet_path(self) -> Path:
        return self.root / "data" / f"chunk-{self.episode_chunk:03d}" / f"episode_{self.episode_index:06d}.parquet"

    def video_path(self, key: str) -> Path:
        return self.root / "videos" / f"chunk-{self.episode_chunk:03d}" / key / f"episode_{self.episode_index:06d}.mp4"


class LiberoMotusDataset(Dataset):
    """Read all four standard LIBERO suites without requiring the lerobot package."""

    def __init__(
        self,
        dataset_dir: str,
        cache_dir: str,
        action_stats_path: str,
        lap_subdir: str = "lap",
        global_downsample_rate: int = 1,
        video_action_freq_ratio: int = 2,
        num_video_frames: int = 4,
        video_size: Tuple[int, int] = (384, 320),
        val: bool = False,
        val_fraction: float = 0.05,
        split_seed: int = 42,
        max_episodes: Optional[int] = None,
        use_language_action: bool = True,
        require_cache: bool = True,
        normalize_actions: bool = True,
        target_action_dim: int = 7,
        include_history_actions: bool = False,
        history_action_length: Optional[int] = None,
        vlm_checkpoint_path: Optional[str] = None,
        image_aug: bool = False,
    ) -> None:
        if int(global_downsample_rate) != 1:
            raise ValueError("LIBERO raw OSC training requires global_downsample_rate=1")
        if int(num_video_frames) <= 0 or int(video_action_freq_ratio) <= 0:
            raise ValueError("LIBERO video frames and action frequency ratio must be positive")
        if image_aug:
            raise ValueError("LIBERO image augmentation is not implemented for synchronized camera pairs")
        if not 0.0 < float(val_fraction) < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")

        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.lap_subdir = str(lap_subdir)
        if not normalize_actions:
            raise ValueError("LIBERO raw OSC training requires normalize_actions=true")
        self.action_normalizer = RawOscActionNormalizer.from_file(action_stats_path)
        self.target_action_dim = int(target_action_dim)
        if self.target_action_dim < 7:
            raise ValueError(f"LIBERO target_action_dim must be at least 7, got {self.target_action_dim}")
        self.action_chunk_size = int(num_video_frames) * int(video_action_freq_ratio)
        self.include_history_actions = bool(include_history_actions)
        self.history_action_length = int(history_action_length or self.action_chunk_size)
        if self.history_action_length != self.action_chunk_size:
            raise ValueError(
                "history_action_length must match action_chunk_size "
                f"({self.action_chunk_size}), got {self.history_action_length}"
            )
        self.num_video_frames = int(num_video_frames)
        self.video_action_freq_ratio = int(video_action_freq_ratio)
        self.video_size = tuple(map(int, video_size))
        self.val = bool(val)
        self.use_language_action = bool(use_language_action)
        self.require_cache = bool(require_cache)

        self.episodes = self._discover_episodes(float(val_fraction), int(split_seed))
        if max_episodes is not None:
            self.episodes = self.episodes[: int(max_episodes)]
        if not self.episodes:
            raise ValueError(f"No {'validation' if self.val else 'training'} LIBERO episodes found")

        self.samples: List[Tuple[int, int]] = []
        for episode_idx, episode in enumerate(self.episodes):
            # o_t, a configurable contiguous action chunk, and its final future observation.
            for frame_idx in range(max(0, episode.length - self.action_chunk_size)):
                self.samples.append((episode_idx, frame_idx))
        if not self.samples:
            raise ValueError("No valid LIBERO temporal windows found")

        self.vlm_processor = None
        if vlm_checkpoint_path:
            if AutoProcessor is None:
                raise ImportError("transformers.AutoProcessor is required for LAP training")
            self.vlm_processor = AutoProcessor.from_pretrained(
                vlm_checkpoint_path,
                trust_remote_code=True,
            )

        if self.require_cache:
            self._validate_cache()

        logger.info(
            "LIBERO dataset ready: split=%s episodes=%d windows=%d video=%s actions=%d",
            "val" if self.val else "train",
            len(self.episodes),
            len(self.samples),
            self.video_size,
            self.action_chunk_size,
        )

    def _discover_episodes(self, val_fraction: float, split_seed: int) -> List[Episode]:
        suite_roots = sorted(
            path for path in self.dataset_dir.iterdir()
            if path.is_dir() and (path / "meta" / "info.json").exists()
        )
        if not suite_roots:
            raise FileNotFoundError(f"No LeRobot suites under {self.dataset_dir}")

        selected: List[Episode] = []
        for root in suite_roots:
            with (root / "meta" / "info.json").open("r", encoding="utf-8") as handle:
                info = json.load(handle)
            if info.get("codebase_version") != "v2.1":
                raise ValueError(f"Unsupported LeRobot version in {root}: {info.get('codebase_version')}")
            if int(info.get("fps", 0)) != 20:
                raise ValueError(f"Expected 20 Hz LIBERO data in {root}")
            features = info.get("features", {})
            for key in (THIRD_PERSON_KEY, WRIST_KEY):
                if key not in features:
                    raise ValueError(f"Missing {key} camera in {root}")

            tasks = load_jsonl(root / "meta" / "tasks.jsonl")
            task_index_by_text = {str(row["task"]): int(row["task_index"]) for row in tasks}
            rows = load_jsonl(root / "meta" / "episodes.jsonl")
            grouped: Dict[int, List[Dict[str, Any]]] = {}
            for row in rows:
                instruction = str(row["tasks"][0])
                task_index = task_index_by_text[instruction]
                grouped.setdefault(task_index, []).append(row)

            suite = suite_name_from_root(root)
            suite_digest = int(hashlib.sha1(suite.encode("utf-8")).hexdigest()[:8], 16)
            for task_index, task_rows in sorted(grouped.items()):
                shuffled = list(task_rows)
                random.Random(split_seed + suite_digest + task_index).shuffle(shuffled)
                num_val = max(1, int(round(len(shuffled) * val_fraction)))
                chosen = shuffled[:num_val] if self.val else shuffled[num_val:]
                for row in sorted(chosen, key=lambda item: int(item["episode_index"])):
                    instruction = str(row["tasks"][0])
                    selected.append(
                        Episode(
                            suite=suite,
                            root=root,
                            episode_index=int(row["episode_index"]),
                            length=int(row["length"]),
                            task_index=task_index,
                            instruction=instruction,
                            chunk_size=int(info.get("chunks_size", 1000)),
                        )
                    )
        return selected

    def _t5_path(self, episode: Episode) -> Path:
        return self.cache_dir / "t5" / episode.suite / f"task_{episode.task_index:02d}.pt"

    def _lap_path(self, episode: Episode) -> Path:
        return self.cache_dir / self.lap_subdir / episode.suite / f"episode_{episode.episode_index:06d}.txt"

    def _validate_cache(self) -> None:
        missing: List[Path] = []
        for episode in self.episodes:
            if not self._t5_path(episode).exists():
                missing.append(self._t5_path(episode))
            if self.use_language_action and not self._lap_path(episode).exists():
                missing.append(self._lap_path(episode))
            if len(missing) >= 10:
                break
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"LIBERO preprocessing cache is incomplete:\n{formatted}")

    @lru_cache(maxsize=32)
    def _load_episode_arrays(self, parquet_path: str) -> Tuple[np.ndarray, np.ndarray]:
        table = pq.read_table(parquet_path, columns=["observation.state", "action"])
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 8:
            raise ValueError(f"Expected state [T,8], got {states.shape} in {parquet_path}")
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Expected action [T,7], got {actions.shape} in {parquet_path}")
        return states, actions

    @lru_cache(maxsize=128)
    def _load_t5(self, path: str) -> torch.Tensor:
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            value = torch.load(path, map_location="cpu")
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"Expected [S,D] T5 tensor in {path}")
        return value

    @lru_cache(maxsize=32)
    def _load_lap_lines(self, path: str) -> Tuple[str, ...]:
        with open(path, "r", encoding="utf-8") as handle:
            return tuple(line.rstrip("\n") for line in handle)

    def _load_composite_frames(self, episode: Episode, indices: List[int]) -> torch.Tensor:
        third = load_libero_video_frames(
            str(episode.video_path(THIRD_PERSON_KEY)),
            indices,
            fill_missing_with_black=True,
        )
        wrist = load_libero_video_frames(
            str(episode.video_path(WRIST_KEY)),
            indices,
            fill_missing_with_black=True,
        )
        frames = []
        for third_frame, wrist_frame in zip(third, wrist):
            third_np = np.rint(third_frame.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            wrist_np = np.rint(wrist_frame.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            composed = compose_libero_views(third_np, wrist_np, self.video_size)
            frames.append(torch.from_numpy(composed).permute(2, 0, 1).float() / 255.0)
        return torch.stack(frames)

    def __len__(self) -> int:
        return len(self.samples)

    def _history_action_sequence(self, actions: np.ndarray, frame_idx: int) -> torch.Tensor:
        """Build a causal action history ending at ``frame_idx - 1``.

        Missing actions at the start of an episode are left-padded with raw
        OSC zeros before applying the same normalization as the target chunk.
        """
        history = np.zeros(
            (self.history_action_length, actions.shape[-1]),
            dtype=np.float32,
        )
        history_start = max(0, int(frame_idx) - self.history_action_length)
        available = actions[history_start:int(frame_idx)]
        if len(available):
            history[-len(available):] = available
        normalized = torch.from_numpy(self.action_normalizer.normalize(history))
        if normalized.shape[-1] < self.target_action_dim:
            padded = normalized.new_zeros(*normalized.shape[:-1], self.target_action_dim)
            padded[..., : normalized.shape[-1]] = normalized
            normalized = padded
        return normalized

    def __getitem__(self, index: int) -> Dict[str, Any]:
        episode_position, frame_idx = self.samples[index]
        episode = self.episodes[episode_position]
        states, actions = self._load_episode_arrays(str(episode.parquet_path))

        action_end = frame_idx + self.action_chunk_size
        raw_action_sequence = actions[frame_idx:action_end].copy()
        action_sequence = torch.from_numpy(self.action_normalizer.normalize(raw_action_sequence))
        action_mask = torch.ones_like(action_sequence, dtype=torch.bool)
        if action_sequence.shape[-1] < self.target_action_dim:
            padded = action_sequence.new_zeros(*action_sequence.shape[:-1], self.target_action_dim)
            padded[..., : action_sequence.shape[-1]] = action_sequence
            action_sequence = padded
            padded_mask = torch.zeros_like(action_sequence, dtype=torch.bool)
            padded_mask[..., : action_mask.shape[-1]] = action_mask
            action_mask = padded_mask
        initial_state = torch.from_numpy(states[frame_idx].copy())
        future_indices = [
            frame_idx + (step + 1) * self.video_action_freq_ratio
            for step in range(self.num_video_frames)
        ]
        frames = self._load_composite_frames(episode, [frame_idx] + future_indices)

        item: Dict[str, Any] = {
            "first_frame": frames[0],
            "video_frames": frames[1:],
            "initial_state": initial_state,
            "action_sequence": action_sequence,
            "action_mask": action_mask,
        }
        if self.include_history_actions:
            item["history_action_sequence"] = self._history_action_sequence(
                actions,
                frame_idx,
            )

        t5_path = self._t5_path(episode)
        if t5_path.exists():
            item["language_embedding"] = self._load_t5(str(t5_path))

        lap_path = self._lap_path(episode)
        if self.use_language_action and lap_path.exists():
            lines = self._load_lap_lines(str(lap_path))
            if frame_idx >= len(lines):
                raise IndexError(f"Missing LAP line {frame_idx} in {lap_path}")
            if self.vlm_processor is not None:
                item["vlm_inputs"] = preprocess_vlm_messages_lap(
                    episode.instruction,
                    tensor_to_pil(frames[0]),
                    self.vlm_processor,
                    lines[frame_idx],
                    supervise_answer=True,
                )
        return item
