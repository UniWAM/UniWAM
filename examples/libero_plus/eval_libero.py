#!/usr/bin/env python3
"""Evaluate a Motus raw-OSC policy on LIBERO-plus."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from examples.libero_plus.eval_utils import deterministic_episode_seed
from examples.libero_plus.tcp_protocol import PolicyClient

DUMMY_ACTION = [0.0] * 6 + [-1.0]
RESOLUTION = 256


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float32)
    return quat[:3] * (2.0 * math.acos(float(quat[3])) / denominator)


def env_action(raw_action: np.ndarray) -> np.ndarray:
    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if raw_action.shape != (7,):
        raise ValueError(f"Expected raw OSC action [7], got {raw_action.shape}")
    gripper = 1.0 - 2.0 * float(raw_action[6] > 0.5)
    return np.concatenate([raw_action[:6], np.asarray([gripper], dtype=np.float32)])


class ChunkedPolicyClient:
    def __init__(self, host: str, port: int, num_inference_steps: int) -> None:
        self.client = PolicyClient(host, port)
        metadata = self.client.request("metadata")
        expected_metadata = {
            "action_representation": "raw_osc",
            "action_normalization": "denormalized",
            "include_state": True,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(f"Unexpected Motus server metadata: {metadata}")
        self.prediction_horizon = int(metadata["action_chunk_size"])
        self.execution_horizon = int(
            metadata.get("execution_horizon", self.prediction_horizon)
        )
        if not 0 < self.execution_horizon <= self.prediction_horizon:
            raise ValueError(f"Invalid Motus action horizons: {metadata}")
        self.metadata = metadata
        self.requires_history_actions = bool(metadata.get("requires_history_actions", False))
        self.history_action_length = int(
            metadata.get("history_action_length", self.prediction_horizon)
        )
        if self.requires_history_actions and self.history_action_length != self.prediction_horizon:
            raise ValueError(f"Invalid Motus history horizon: {metadata}")
        self.num_inference_steps = int(num_inference_steps)
        self.actions = None
        self.cursor = 0
        self.episode_seed = 0
        self.chunk_index = 0
        self.history_actions = np.zeros((self.history_action_length, 7), dtype=np.float32)

    def reset(self, episode_seed: int) -> None:
        self.actions = None
        self.cursor = 0
        self.episode_seed = int(episode_seed)
        self.chunk_index = 0
        self.history_actions.fill(0.0)

    def step(self, example: dict) -> np.ndarray:
        if self.actions is None or self.cursor >= self.execution_horizon:
            request_example = dict(example)
            if self.requires_history_actions:
                request_example["history_actions"] = self.history_actions.copy()
            response = self.client.request("predict_action", {
                "examples": [request_example],
                "num_inference_steps": self.num_inference_steps,
                "seed": self.episode_seed + self.chunk_index,
            })
            self.chunk_index += 1
            self.actions = np.asarray(response["actions"], dtype=np.float32)[0]
            expected_shape = (self.prediction_horizon, 7)
            if self.actions.shape != expected_shape:
                raise ValueError(
                    f"Expected server action chunk {expected_shape}, got {self.actions.shape}"
                )
            self.cursor = 0
        action = self.actions[self.cursor]
        self.cursor += 1
        if self.requires_history_actions:
            self.history_actions[:-1] = self.history_actions[1:]
            self.history_actions[-1] = action
        return action


def make_env(task, seed: int):
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=RESOLUTION,
        camera_widths=RESOLUTION,
    )
    env.seed(seed)
    return env


def max_steps_for_suite(suite: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }[suite]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9883)
    parser.add_argument("--benchmark-mode", choices=("plus", "original"), default="plus")
    parser.add_argument("--task-suite-name", choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"), required=True)
    parser.add_argument("--num-trials-per-task", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument("--start-task", type=int, default=0)
    parser.add_argument("--end-task", type=int, default=-1)
    parser.add_argument("--shard-output", action="store_true")
    parser.add_argument(
        "--video-mode",
        choices=("all", "failures", "none"),
        default="all",
        help="Save all videos, only failures, or no videos.",
    )
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="outputs/libero_plus")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_trials_per_task is None:
        args.num_trials_per_task = 1 if args.benchmark_mode == "plus" else 50
    return args


def empty_results(
    suite_name: str,
    benchmark_mode: str,
    start_task: int,
    end_task: int,
    task_count: int,
) -> dict:
    return {
        "suite": suite_name,
        "benchmark_mode": benchmark_mode,
        "selection": {
            "start_task": start_task,
            "end_task": end_task,
            "task_count": task_count,
        },
        "tasks": [],
        "categories": {},
        "total_successes": 0,
        "total_episodes": 0,
    }


def load_resume_results(
    results_path: Path,
    suite,
    suite_name: str,
    benchmark_mode: str,
    classification: dict,
    expected_task_ids: list[int],
    num_trials_per_task: int,
) -> tuple[dict, int]:
    with results_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)

    if results.get("suite") != suite_name:
        raise ValueError(
            f"Resume suite mismatch: file={results.get('suite')!r}, requested={suite_name!r}"
        )
    if results.get("benchmark_mode", "plus") != benchmark_mode:
        raise ValueError(
            "Resume benchmark mode mismatch: "
            f"file={results.get('benchmark_mode', 'plus')!r}, requested={benchmark_mode!r}"
        )
    tasks = results.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Resume results has no valid tasks list")
    expected_selection = {
        "start_task": expected_task_ids[0],
        "end_task": expected_task_ids[-1] + 1,
        "task_count": int(results.get("selection", {}).get("task_count", -1)),
    }
    selection = results.get("selection")
    if selection is not None and (
        int(selection.get("start_task", -1)) != expected_selection["start_task"]
        or int(selection.get("end_task", -1)) != expected_selection["end_task"]
    ):
        raise ValueError(
            f"Resume task selection mismatch: file={selection}, "
            f"requested=[{expected_selection['start_task']}, {expected_selection['end_task']})"
        )
    if len(tasks) > len(expected_task_ids):
        raise ValueError(
            f"Resume file has {len(tasks)} tasks, but this shard has {len(expected_task_ids)}"
        )

    expected_categories = {}
    total_successes = 0
    total_episodes = 0
    for expected_task_id, row in zip(expected_task_ids, tasks):
        if int(row.get("task_id", -1)) != expected_task_id:
            raise ValueError(
                "Resume tasks must be a contiguous prefix of the selected task shard"
            )
        if int(row.get("episodes", -1)) != num_trials_per_task:
            raise ValueError(
                f"Resume trial count mismatch at task {expected_task_id}: "
                f"file={row.get('episodes')}, requested={num_trials_per_task}"
            )

        expected_instruction = str(suite.get_task(expected_task_id).language)
        expected_classification = classification[expected_task_id + 1]
        if row.get("instruction") != expected_instruction:
            raise ValueError(f"Resume task order mismatch at task {expected_task_id}")
        if row.get("benchmark_name") != expected_classification["name"]:
            raise ValueError(f"Resume benchmark mismatch at task {expected_task_id}")
        if row.get("category") != expected_classification["category"]:
            raise ValueError(f"Resume category mismatch at task {expected_task_id}")

        successes = int(row.get("successes", -1))
        if not 0 <= successes <= num_trials_per_task:
            raise ValueError(f"Invalid success count at task {expected_task_id}: {successes}")
        category = str(row["category"])
        category_result = expected_categories.setdefault(
            category, {"successes": 0, "episodes": 0}
        )
        category_result["successes"] += successes
        category_result["episodes"] += num_trials_per_task
        total_successes += successes
        total_episodes += num_trials_per_task

    if results.get("categories") != expected_categories:
        raise ValueError("Resume category totals do not match completed task rows")
    if int(results.get("total_successes", -1)) != total_successes:
        raise ValueError("Resume total_successes does not match completed task rows")
    if int(results.get("total_episodes", -1)) != total_episodes:
        raise ValueError("Resume total_episodes does not match completed task rows")
    return results, len(tasks)


def save_results(results_path: Path, results: dict) -> None:
    temporary_path = results_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    temporary_path.replace(results_path)


def main() -> None:
    args = parse_args()
    logging.info("Arguments: %s", json.dumps(vars(args), indent=4))
    np.random.seed(args.seed)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if args.benchmark_mode == "original" and suite.n_tasks != 10:
        raise ValueError(
            "Original LIBERO mode requires the official 10-task suite registry, "
            f"but {args.task_suite_name} contains {suite.n_tasks} tasks. "
            "Check LIBERO_HOME/PYTHONPATH and use the official LIBERO repository."
        )
    task_count = suite.n_tasks if args.max_tasks <= 0 else min(args.max_tasks, suite.n_tasks)
    start_task = int(args.start_task)
    end_task = task_count if args.end_task < 0 else int(args.end_task)
    if not 0 <= start_task < end_task <= task_count:
        raise ValueError(
            f"Invalid task range [{start_task}, {end_task}) for {task_count} selected tasks"
        )
    selected_task_ids = list(range(start_task, end_task))

    output_dir = Path(args.output_dir) / args.task_suite_name
    if args.shard_output:
        output_dir = output_dir / "shards" / f"{start_task:06d}_{end_task:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    task_order = [str(suite.get_task(task_id).language) for task_id in selected_task_ids]
    logging.info("Task orders: %s", json.dumps(task_order, indent=4))
    logging.info("Task suite: %s", args.task_suite_name)
    logging.info(
        "Evaluating task range [%d, %d): %d of %d selected tasks (%d suite tasks)",
        start_task,
        end_task,
        len(selected_task_ids),
        task_count,
        suite.n_tasks,
    )
    if args.benchmark_mode == "plus":
        libero_home = Path(os.environ.get("LIBERO_HOME", ""))
        classification_path = (
            libero_home / "libero" / "libero" / "benchmark" / "task_classification.json"
        )
        with classification_path.open("r", encoding="utf-8") as handle:
            classification_rows = json.load(handle)[args.task_suite_name]
        classification = {int(row["id"]): row for row in classification_rows}
    else:
        classification = {
            task_id + 1: {
                "id": task_id + 1,
                "name": str(suite.get_task(task_id).name),
                "category": "original",
            }
            for task_id in range(suite.n_tasks)
        }
    results_path = output_dir / "results.json"
    if args.resume and results_path.exists():
        results, start_task_id = load_resume_results(
            results_path=results_path,
            suite=suite,
            suite_name=args.task_suite_name,
            benchmark_mode=args.benchmark_mode,
            classification=classification,
            expected_task_ids=selected_task_ids,
            num_trials_per_task=args.num_trials_per_task,
        )
        pending_task_ids = selected_task_ids[start_task_id:]
        logging.info(
            "Resuming shard after %d completed tasks; keeping %d episodes",
            start_task_id,
            len(results["tasks"]),
            results["total_episodes"],
        )
    else:
        results = empty_results(
            args.task_suite_name,
            benchmark_mode=args.benchmark_mode,
            start_task=start_task,
            end_task=end_task,
            task_count=task_count,
        )
        pending_task_ids = selected_task_ids
        if args.resume:
            logging.info("No existing results.json found; starting shard from task %d", start_task)

    policy = ChunkedPolicyClient(args.host, args.port, args.num_inference_steps)
    logging.info("Policy metadata: %s", json.dumps(policy.metadata, indent=4))
    if args.resume and results_path.exists():
        backup_path = output_dir / (
            f"results.before_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        shutil.copy2(results_path, backup_path)
        logging.info("Backed up existing results to %s", backup_path)

    completed_tasks = len(results["tasks"])
    for task_id in tqdm.tqdm(
        pending_task_ids,
        initial=completed_tasks,
        total=len(selected_task_ids),
    ):
        task = suite.get_task(task_id)
        instruction = str(task.language)
        logging.info("Task: %s", instruction)
        task_classification = classification[task_id + 1]
        category = str(task_classification["category"])
        category_result = results["categories"].setdefault(
            category, {"successes": 0, "episodes": 0}
        )
        initial_states = suite.get_task_init_states(task_id)
        env = make_env(task, args.seed)
        task_successes = 0
        try:
            for episode_index in tqdm.tqdm(range(args.num_trials_per_task)):
                logging.info("Starting episode %d...", episode_index + 1)
                policy.reset(
                    deterministic_episode_seed(args.seed, task_id, episode_index)
                )
                env.reset()
                observation = env.set_init_state(initial_states[episode_index])
                frames = []
                done = False
                for step in range(max_steps_for_suite(args.task_suite_name) + args.num_steps_wait):
                    if step < args.num_steps_wait:
                        observation, _, done, _ = env.step(DUMMY_ACTION)
                        continue

                    third = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
                    state = np.concatenate([
                        observation["robot0_eef_pos"],
                        quat_to_axis_angle(observation["robot0_eef_quat"]),
                        observation["robot0_gripper_qpos"],
                    ]).astype(np.float32)
                    raw_action = policy.step({"image": [third, wrist], "lang": instruction, "state": state[None]})
                    if args.video_mode != "none":
                        frames.append(third)
                    observation, _, done, _ = env.step(env_action(raw_action).tolist())
                    if done:
                        task_successes += 1
                        results["total_successes"] += 1
                        category_result["successes"] += 1
                        break

                save_video = args.video_mode == "all" or (
                    args.video_mode == "failures" and not done
                )
                if save_video:
                    suffix = "success" if done else "failure"
                    imageio.mimwrite(
                        output_dir / f"task_{task_id:04d}_episode_{episode_index:03d}_{suffix}.mp4",
                        frames,
                        fps=20,
                    )
                results["total_episodes"] += 1
                category_result["episodes"] += 1
                total_success_rate = results["total_successes"] / results["total_episodes"]
                logging.info("Success: %s", done)
                logging.info("# episodes completed so far: %d", results["total_episodes"])
                logging.info(
                    "# successes: %d (%.1f%%)",
                    results["total_successes"],
                    100.0 * total_success_rate,
                )
        finally:
            env.close()
        results["tasks"].append({
            "task_id": task_id,
            "benchmark_name": task_classification["name"],
            "category": category,
            "instruction": instruction,
            "successes": task_successes,
            "episodes": args.num_trials_per_task,
        })
        save_results(results_path, results)
        logging.info(
            "Current task success rate: %s",
            task_successes / args.num_trials_per_task,
        )
        logging.info(
            "Current total success rate: %s",
            results["total_successes"] / results["total_episodes"],
        )

    total_success_rate = (
        results["total_successes"] / results["total_episodes"]
        if results["total_episodes"]
        else 0.0
    )
    logging.info("Total success rate: %s", total_success_rate)
    logging.info("Total episodes: %d", results["total_episodes"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
