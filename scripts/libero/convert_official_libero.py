#!/usr/bin/env python3
"""Convert official LIBERO task HDF5 files to this project's v2.1-style loader layout.

Source: https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
The official recorder stores observation j *after* action j. We therefore
pair observation j with action j+1 and discard the final observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
CAMERAS = {
    "observation.images.image": "agentview_rgb",
    "observation.images.wrist_image": "eye_in_hand_rgb",
}
FPS = 20
CHUNK_SIZE = 1000


def convert_gripper(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite [T,7] actions, got {actions.shape}")
    gripper = actions[:, 6]
    if np.all(np.isclose(np.abs(gripper), 1)):
        actions[:, 6] = (1 - gripper) / 2  # LIBERO -1=open, +1=close -> 1=open, 0=close
    elif not np.all(np.isclose(gripper, 0) | np.isclose(gripper, 1)):
        raise ValueError("Expected gripper actions in {-1,+1} or {0,1}")
    return actions


def write_video(frames: h5py.Dataset, path: Path, length: int) -> None:
    height, width, channels = frames.shape[1:]
    if channels != 3 or height % 2 or width % 2 or frames.dtype != np.uint8:
        raise ValueError(f"Expected even-sized uint8 RGB frames, got {frames.shape} {frames.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for index in range(length):
            # The official recorder saved the raw MuJoCo image; evaluation flips both axes.
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frames[index][::-1, ::-1]), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def convert_suite(source: Path, destination: Path, max_episodes: int | None) -> tuple[int, int, int]:
    files = sorted(source.glob("*_demo.hdf5"))
    if not files:
        raise FileNotFoundError(f"No official *_demo.hdf5 files in {source}")
    if destination.exists() or destination.with_name(destination.name + ".partial").exists():
        raise FileExistsError(f"Destination already exists: {destination} (or .partial)")
    stage = destination.with_name(destination.name + ".partial")
    (stage / "meta").mkdir(parents=True)
    tasks, episodes = [], []
    episode_index = total_frames = 0
    image_shape = None
    for task_index, source_file in enumerate(files):
        instruction = source_file.stem.removesuffix("_demo").replace("_", " ")
        tasks.append({"task_index": task_index, "task": instruction})
        with h5py.File(source_file, "r") as handle:
            for demo_name in sorted(handle["data"], key=lambda name: int(name.split("_")[-1])):
                demo = handle["data"][demo_name]
                actions = convert_gripper(demo["actions"][:])
                obs = demo["obs"]
                ee = np.asarray(obs["ee_states"], dtype=np.float32)
                gripper = np.asarray(obs["gripper_states"], dtype=np.float32)
                if ee.shape != (len(actions), 6) or gripper.shape != (len(actions), 2):
                    raise ValueError(f"Unexpected state shape in {source_file}:{demo_name}")
                length = len(actions) - 1
                if length < 17:
                    continue  # 16-action windows require at least one future observation
                state = np.concatenate([ee[:-1], gripper[:-1]], axis=1)
                if not np.isfinite(state).all():
                    raise ValueError(f"Non-finite state in {source_file}:{demo_name}")
                shape = tuple(obs["agentview_rgb"].shape[1:])
                if image_shape is None:
                    image_shape = shape
                if shape != image_shape:
                    raise ValueError(f"Mixed image shapes in {source_file}:{demo_name}")
                chunk = episode_index // CHUNK_SIZE
                relative = Path(f"chunk-{chunk:03d}") / f"episode_{episode_index:06d}"
                parquet_path = stage / "data" / relative.with_suffix(".parquet")
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.table({
                    "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 8)),
                    "action": pa.array(actions[1:].tolist(), type=pa.list_(pa.float32(), 7)),
                    "timestamp": pa.array(np.arange(length, dtype=np.float32) / FPS),
                    "frame_index": pa.array(np.arange(length, dtype=np.int64)),
                    "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64)),
                    "index": pa.array(np.arange(total_frames, total_frames + length, dtype=np.int64)),
                    "task_index": pa.array(np.full(length, task_index, dtype=np.int64)),
                }), parquet_path)
                for video_key, hdf5_key in CAMERAS.items():
                    frames = obs[hdf5_key]
                    if len(frames) != len(actions) or tuple(frames.shape[1:]) != image_shape:
                        raise ValueError(f"Camera mismatch in {source_file}:{demo_name}:{hdf5_key}")
                    write_video(frames, stage / "videos" / relative.parent / video_key / (relative.name + ".mp4"), length)
                episodes.append({"episode_index": episode_index, "tasks": [instruction], "length": length})
                episode_index += 1
                total_frames += length
                if max_episodes is not None and episode_index >= max_episodes:
                    break
        if max_episodes is not None and episode_index >= max_episodes:
            break
    if not episodes:
        raise ValueError(f"No usable episodes in {source}")
    height, width, _ = image_shape
    features = {
        key: {"dtype": "video", "shape": list(image_shape), "info": {"video.fps": FPS, "video.codec": "h264"}}
        for key in CAMERAS
    }
    features.update({
        "observation.state": {"dtype": "float32", "shape": [8]},
        "action": {"dtype": "float32", "shape": [7]},
    })
    info = {
        "codebase_version": "v2.1", "robot_type": "franka", "fps": FPS,
        "total_episodes": episode_index, "total_frames": total_frames,
        "total_tasks": len(tasks), "total_videos": 2 * episode_index,
        "total_chunks": (episode_index + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE, "splits": {"train": f"0:{episode_index}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (stage / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("tasks", tasks), ("episodes", episodes)):
        (stage / "meta" / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    stage.rename(destination)
    return episode_index, total_frames, len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Official LIBERO dataset parent")
    parser.add_argument("--output", type=Path, required=True, help="New LeRobot four-suite parent")
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=SUITES)
    parser.add_argument("--max-episodes", type=int, help="Per-suite smoke-test limit")
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    for suite in args.suites:
        counts = convert_suite(args.source / suite, args.output / suite, args.max_episodes)
        print(f"{suite}: {counts[0]} episodes, {counts[1]} frames, {counts[2]} tasks")


if __name__ == "__main__":
    main()
