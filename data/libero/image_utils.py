"""Shared LIBERO train/eval image composition."""

from __future__ import annotations

import logging
from typing import List, Tuple

import av
import numpy as np
import torch

from data.utils.image_utils import resize_with_padding


logger = logging.getLogger(__name__)


def load_libero_video_frames(
    video_path: str,
    frame_indices: List[int],
    *,
    fill_missing_with_black: bool = False,
) -> torch.Tensor:
    """Decode exact AV1 frame indices with PyAV/libdav1d in RGB order.

    When ``fill_missing_with_black`` is enabled, unavailable indices are
    represented by black frames with the video's native resolution. This lets
    synchronized multi-camera training preserve the valid camera view instead
    of dropping the entire sample.
    """
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")
    if any(index < 0 for index in frame_indices):
        raise ValueError(f"Negative frame index requested from {video_path}: {frame_indices}")

    requested = set(map(int, frame_indices))
    decoded = {}
    with av.open(video_path) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {video_path}")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        stream.codec_context.thread_type = "NONE"
        frame_height = int(stream.codec_context.height)
        frame_width = int(stream.codec_context.width)

        if stream.average_rate is None or stream.time_base is None:
            raise ValueError(f"Missing frame-rate metadata in {video_path}")
        rate = float(stream.average_rate)
        time_base = float(stream.time_base)
        start_pts = int(stream.start_time or 0)
        first_index = min(requested)
        last_index = max(requested)
        target_pts = start_pts + int((first_index / rate) / time_base)
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_index = int(round((int(frame.pts) - start_pts) * time_base * rate))
            if frame_index in requested:
                decoded[frame_index] = frame.to_ndarray(format="rgb24")
            if frame_index >= last_index and requested.issubset(decoded):
                break

    missing = sorted(requested.difference(decoded))
    if missing:
        if not fill_missing_with_black:
            raise ValueError(f"Frames {missing} are unavailable in {video_path}")
        if decoded:
            black_frame = np.zeros_like(next(iter(decoded.values())))
        elif frame_height > 0 and frame_width > 0:
            black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        else:
            raise ValueError(
                f"Frames {missing} are unavailable in {video_path}, and the frame size is unknown"
            )
        for index in missing:
            decoded[index] = black_frame.copy()
        logger.warning(
            "Frames %s are unavailable in %s; replacing them with black frames",
            missing,
            video_path,
        )
    frames = np.stack([decoded[int(index)] for index in frame_indices])
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0


def compose_libero_views(
    third_person: np.ndarray,
    wrist: np.ndarray,
    target_size: Tuple[int, int] = (384, 320),
    rotate_180: bool = False,
) -> np.ndarray:
    """Stack third-person and wrist RGB views without changing aspect ratio."""
    if third_person.ndim != 3 or wrist.ndim != 3:
        raise ValueError("LIBERO views must be HWC images")
    if third_person.shape[2] != 3 or wrist.shape[2] != 3:
        raise ValueError("LIBERO views must have three RGB channels")

    target_h, target_w = map(int, target_size)
    if target_h % 2:
        raise ValueError(f"target height must be even, got {target_h}")

    if rotate_180:
        third_person = np.ascontiguousarray(np.rot90(third_person, 2))
        wrist = np.ascontiguousarray(np.rot90(wrist, 2))

    half_size = (target_h // 2, target_w)
    top = resize_with_padding(third_person, half_size)
    bottom = resize_with_padding(wrist, half_size)
    composed = np.concatenate([top, bottom], axis=0)
    if composed.shape != (target_h, target_w, 3):
        raise RuntimeError(f"unexpected composed shape: {composed.shape}")
    return np.ascontiguousarray(composed)
