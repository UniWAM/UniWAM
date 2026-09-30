#!/usr/bin/env python3
"""Serve a LIBERO-trained Motus checkpoint to the Motus LIBERO-plus client."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoProcessor

from data.libero.action_normalization import RawOscActionNormalizer
from data.libero.image_utils import compose_libero_views
from examples.libero_plus.tcp_protocol import PolicyServer
from models.motus import Motus, MotusConfig

ROOT = Path(__file__).resolve().parents[2]
BAK_ROOT = str((ROOT / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)
from wan.modules.t5 import T5EncoderModel  # noqa: E402

logger = logging.getLogger(__name__)


class LiberoMotusPolicy:
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        wan_path: str,
        vlm_path: str,
        action_stats_path: str | None = None,
        device: str = "cuda",
        num_inference_steps: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.wan_path = str(Path(wan_path).resolve())
        self.vlm_path = str(Path(vlm_path).resolve())
        with open(config_path, "r", encoding="utf-8") as handle:
            self.config_dict = yaml.safe_load(handle)
        configured_stats_path = self.config_dict["dataset"].get("action_stats_path")
        self.action_normalizer = RawOscActionNormalizer.from_file(
            action_stats_path or configured_stats_path
        )

        self.model = Motus(self._model_config()).to(self.device)
        self.model.load_checkpoint(checkpoint_path, strict=True)
        self.model.eval()
        self.prediction_horizon = int(self.model.config.action_chunk_size)
        flow_source = self.config_dict["model"].get("flow_source", {})
        self.flow_source_mode = str(flow_source.get("mode", "gaussian"))
        self.history_action_length = int(
            flow_source.get("history_length", self.prediction_horizon)
        )
        if self.flow_source_mode == "history" and self.history_action_length != self.prediction_horizon:
            raise ValueError(
                "history_length must match the LIBERO prediction horizon "
                f"({self.prediction_horizon}), got {self.history_action_length}"
            )
        configured_execution_horizon = self.config_dict["model"]["inference"].get(
            "execution_horizon", self.prediction_horizon
        )
        self.execution_horizon = int(configured_execution_horizon)
        self.model_action_dim = self._validate_model_action_dim(self.model.config.action_dim)
        if not 0 < self.execution_horizon <= self.prediction_horizon:
            raise ValueError(
                "execution_horizon must be positive and no greater than the prediction horizon"
            )

        self.t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=self.dtype,
            device=str(self.device),
            checkpoint_path=os.path.join(self.wan_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(self.wan_path, "google", "umt5-xxl"),
        )
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)
        configured_steps = self.config_dict["model"]["inference"]["num_inference_timesteps"]
        self.num_inference_steps = int(num_inference_steps or configured_steps)
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        self._inference_lock = threading.Lock()
        self._request_count = 0
        self._queue_wait_total = 0.0
        self._inference_total = 0.0

    def _model_config(self) -> MotusConfig:
        common = self.config_dict["common"]
        model = self.config_dict["model"]
        action = model["action_expert"]
        understanding = model["und_expert"]
        flow_source = model.get("flow_source", {})
        flow_source_mode = str(flow_source.get("mode", "gaussian"))
        return MotusConfig(
            wan_checkpoint_path=self.wan_path,
            vae_path=os.path.join(self.wan_path, "Wan2.2_VAE.pth"),
            wan_config_path=self.wan_path,
            video_precision=model["wan"].get("precision", "bfloat16") if "wan" in model else "bfloat16",
            vlm_checkpoint_path=self.vlm_path,
            und_expert_hidden_size=int(understanding["hidden_size"]),
            und_expert_ffn_dim_multiplier=int(understanding["ffn_dim_multiplier"]),
            und_expert_norm_eps=float(understanding["norm_eps"]),
            vlm_adapter_input_dim=int(understanding["vlm"]["input_dim"]),
            vlm_adapter_projector_type=str(understanding["vlm"]["projector_type"]),
            num_layers=30,
            action_state_dim=int(common["state_dim"]),
            action_dim=int(common["action_dim"]),
            action_expert_dim=int(action["hidden_size"]),
            action_expert_ffn_dim_multiplier=int(action["ffn_dim_multiplier"]),
            action_expert_norm_eps=float(action["norm_eps"]),
            global_downsample_rate=int(common["global_downsample_rate"]),
            video_action_freq_ratio=int(common["video_action_freq_ratio"]),
            num_video_frames=int(common["num_video_frames"]),
            video_height=int(common["video_height"]),
            video_width=int(common["video_width"]),
            batch_size=1,
            video_loss_weight=float(model["loss_weights"]["video_loss_weight"]),
            action_loss_weight=float(model["loss_weights"]["action_loss_weight"]),
            flow_source_mode=flow_source_mode,
            flow_source_video_mode=str(flow_source.get("video_mode", flow_source_mode)),
            flow_source_action_noise_std=float(flow_source.get("action_noise_std", 0.0)),
            load_pretrained_backbones=False,
            vlm_frozen=False,
            training_mode="finetune",
        )

    def _prepare_observation(self, example: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, str]:
        images = example.get("image")
        if not isinstance(images, (list, tuple)) or len(images) != 2:
            raise ValueError("Expected [third_person, wrist] images from LIBERO-plus")
        third = np.asarray(images[0], dtype=np.uint8)
        wrist = np.asarray(images[1], dtype=np.uint8)
        target_size = (
            int(self.config_dict["common"]["video_height"]),
            int(self.config_dict["common"]["video_width"]),
        )
        # LIBERO-plus eval_libero.py rotates both views before sending them.
        composed = compose_libero_views(third, wrist, target_size=target_size, rotate_180=False)
        frame = torch.from_numpy(composed).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

        state = np.asarray(example.get("state"), dtype=np.float32).reshape(1, -1)
        expected_state_dim = int(self.config_dict["common"]["state_dim"])
        if state.shape != (1, expected_state_dim):
            raise ValueError(f"Expected LIBERO state [1,{expected_state_dim}], got {state.shape}")
        instruction = str(example.get("lang", "")).strip()
        if not instruction:
            raise ValueError("LIBERO task instruction is empty")
        return frame.to(self.device), torch.from_numpy(state).to(self.device), instruction

    def _prepare_vlm(self, instruction: str, frame: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_np = np.rint(
            frame[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0
        ).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image_np, mode="RGB")
        question = f"任务：{instruction}\n请给出下一步动作语言描述。"
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        text = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.vlm_processor(text=[text], images=[image], padding=True, return_tensors="pt")
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "action_chunk_size": self.prediction_horizon,
            "execution_horizon": self.execution_horizon,
            "action_representation": "raw_osc",
            "action_normalization": "denormalized",
            "include_state": True,
            "requires_history_actions": self.flow_source_mode == "history",
            "history_action_length": self.history_action_length,
        }

    def _prepare_action_source(self, example: Dict[str, Any]) -> torch.Tensor | None:
        if self.flow_source_mode != "history":
            return None
        history = np.asarray(example.get("history_actions"), dtype=np.float32)
        expected_shape = (self.history_action_length, 7)
        if history.shape != expected_shape:
            raise ValueError(
                f"Expected LIBERO history_actions {expected_shape}, got {history.shape}"
            )
        normalized = self.action_normalizer.normalize(history)
        action_source = np.zeros(
            (self.history_action_length, self.model_action_dim),
            dtype=np.float32,
        )
        action_source[:, :7] = normalized
        return torch.from_numpy(action_source).unsqueeze(0).to(
            device=self.device,
            dtype=self.dtype,
        )

    def predict_action(self, request: Dict[str, Any]) -> Dict[str, np.ndarray]:
        # A shared-GPU evaluation may connect several simulator workers to this
        # server. Keep model, processor, and CUDA access single-threaded while
        # the simulators continue stepping in parallel.
        queued_at = time.perf_counter()
        with self._inference_lock:
            started_at = time.perf_counter()
            result = self._predict_action_locked(request)
            finished_at = time.perf_counter()
            self._request_count += 1
            self._queue_wait_total += started_at - queued_at
            self._inference_total += finished_at - started_at
            if self._request_count % 100 == 0:
                logger.info(
                    "Inference requests=%d avg_queue_wait=%.3fs avg_inference=%.3fs",
                    self._request_count,
                    self._queue_wait_total / self._request_count,
                    self._inference_total / self._request_count,
                )
            return result

    def _predict_action_locked(self, request: Dict[str, Any]) -> Dict[str, np.ndarray]:
        request_seed = request.get("seed")
        if request_seed is None:
            return self._predict_action_with_current_rng(request)

        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(request_seed))
            return self._predict_action_with_current_rng(request)

    def _predict_action_with_current_rng(self, request: Dict[str, Any]) -> Dict[str, np.ndarray]:
        examples = request["examples"]
        num_ddim_steps = request.get("num_inference_steps")
        if len(examples) != 1:
            raise ValueError(f"Only batch size 1 is supported, got {len(examples)}")
        example = examples[0]
        frame, state, instruction = self._prepare_observation(example)
        action_source = self._prepare_action_source(example)
        with torch.no_grad():
            t5 = self.t5_encoder([instruction], str(self.device))
            if isinstance(t5, torch.Tensor):
                t5 = [t5.squeeze(0) if t5.ndim == 3 else t5]
            _, actions = self.model.inference_step(
                first_frame=frame,
                state=state,
                action_source=action_source,
                num_inference_steps=int(num_ddim_steps or self.num_inference_steps),
                language_embeddings=t5,
                vlm_inputs=[self._prepare_vlm(instruction, frame)],
            )
        raw_actions = self._denormalize_libero_plus_actions(actions)
        return {"actions": raw_actions}

    @staticmethod
    def _validate_model_action_dim(action_dim: int) -> int:
        action_dim = int(action_dim)
        if action_dim not in {7, 14}:
            raise ValueError(
                f"LIBERO-plus supports 7D or 14D Motus actions, got {action_dim}D"
            )
        return action_dim

    def _denormalize_libero_plus_actions(self, actions: torch.Tensor) -> np.ndarray:
        normalized_actions = actions.detach().float().cpu().numpy()
        expected_model_shape = (1, self.prediction_horizon, self.model_action_dim)
        if normalized_actions.shape != expected_model_shape:
            raise ValueError(
                f"Expected predicted actions {expected_model_shape}, got {normalized_actions.shape}"
            )
        normalized_actions = normalized_actions[..., :7]
        expected_shape = (1, self.prediction_horizon, 7)
        if normalized_actions.shape != expected_shape:
            raise ValueError(
                f"Expected predicted actions {expected_shape}, got {normalized_actions.shape}"
            )
        return self.action_normalizer.denormalize(normalized_actions, clip=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/libero_history_future_noise.yaml")
    parser.add_argument("--wan-path", default="pretrained_models/Wan2.2-TI2V-5B")
    parser.add_argument("--vlm-path", default="pretrained_models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--action-stats", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9883)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = LiberoMotusPolicy(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        wan_path=args.wan_path,
        vlm_path=args.vlm_path,
        action_stats_path=args.action_stats,
        num_inference_steps=args.num_inference_steps,
    )
    PolicyServer(policy, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
