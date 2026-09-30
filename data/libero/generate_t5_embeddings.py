#!/usr/bin/env python3
"""Generate one WAN UMT5 embedding per unique LIBERO task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from data.libero.libero_dataset import load_jsonl, suite_name_from_root


def _init_wan_t5_encoder(wan_path: str, device: str, text_len: int):
    bak_root = str(Path(__file__).resolve().parents[2] / "bak")
    if bak_root not in sys.path:
        sys.path.insert(0, bak_root)
    from wan.modules.t5 import T5EncoderModel

    model_dir = Path(wan_path) / "Wan2.2-TI2V-5B"
    return T5EncoderModel(
        text_len=text_len,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device=device,
        checkpoint_path=str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(model_dir / "google" / "umt5-xxl"),
    )


def _encode_t5(encoder, instruction: str, device: str) -> torch.Tensor:
    with torch.no_grad():
        result = encoder([instruction], device)
    embedding = result[0] if isinstance(result, list) else result
    if embedding.ndim == 3 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0)
    return embedding.detach().cpu()


def generate(
    dataset_dir: Path,
    cache_dir: Path,
    wan_path: str,
    device: str,
    text_len: int,
    overwrite: bool,
) -> None:
    suite_roots = sorted(
        path for path in dataset_dir.iterdir()
        if path.is_dir() and (path / "meta" / "info.json").exists()
    )
    pending = []
    for root in suite_roots:
        suite = suite_name_from_root(root)
        for row in load_jsonl(root / "meta" / "tasks.jsonl"):
            output = cache_dir / "t5" / suite / f"task_{int(row['task_index']):02d}.pt"
            if overwrite or not output.exists():
                pending.append((suite, int(row["task_index"]), str(row["task"]), output))
    if not pending:
        print("All LIBERO T5 embeddings already exist")
        return

    encoder = _init_wan_t5_encoder(wan_path=wan_path, device=device, text_len=text_len)
    for index, (suite, task_index, instruction, output) in enumerate(pending, start=1):
        output.parent.mkdir(parents=True, exist_ok=True)
        embedding = _encode_t5(encoder, instruction, device=device)
        temporary = output.with_suffix(".pt.tmp")
        torch.save(embedding, temporary)
        temporary.replace(output)
        print(f"[{index}/{len(pending)}] {suite} task {task_index}: {tuple(embedding.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--wan-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text-len", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate(
        args.dataset_dir.resolve(),
        args.cache_dir.resolve(),
        args.wan_path,
        args.device,
        args.text_len,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
