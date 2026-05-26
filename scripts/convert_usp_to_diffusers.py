#!/usr/bin/env python3
# Copyright 2025 USP Authors. SPDX-License-Identifier: MIT

"""Convert legacy USP DiT/SiT .pt checkpoints into a Diffusers pipeline directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from register_usp import register_usp_diffusers

register_usp_diffusers()

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except Exception:  # pragma: no cover
    safe_load_file = None
    safe_save_file = None

from diffusers.models.transformers.transformer_usp import USPTransformer2DModel, USP_MODEL_PRESETS
from diffusers.schedulers.scheduling_flow_match_usp import USPFlowMatchScheduler


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("Install safetensors to load .safetensors checkpoints.")
        state_dict = safe_load_file(checkpoint_path, device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model", "module", "ema"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
    return _clean_state_dict(state_dict)


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    prefixes = ("model.", "module.", "transformer.")
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def _save_config(output_dir: Path, config: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _save_weights(output_dir: Path, state_dict: Dict[str, torch.Tensor], safe_serialization: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if safe_serialization:
        if safe_save_file is None:
            raise ImportError("Install safetensors or pass --no-safe-serialization.")
        safe_save_file(
            state_dict,
            str(output_dir / "diffusion_pytorch_model.safetensors"),
            metadata={"format": "pt"},
        )
    else:
        torch.save(state_dict, output_dir / "diffusion_pytorch_model.bin")


def _write_model_index(output_dir: Path, pipeline_class: str, scheduler_class: str, vae: str | None) -> None:
    model_index = {
        "_class_name": pipeline_class,
        "_diffusers_version": "0.30.1",
        "scheduler": ["diffusers", scheduler_class],
        "transformer": ["diffusers", "USPTransformer2DModel"],
    }
    if vae is not None:
        model_index["vae"] = ["diffusers", "AutoencoderKL"]
    with open(output_dir / "model_index.json", "w", encoding="utf-8") as handle:
        json.dump(model_index, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert USP DiT/SiT checkpoints to Diffusers layout.")
    parser.add_argument("--checkpoint", required=True, help="Path to USP .pt/.bin/.safetensors weights.")
    parser.add_argument("--output", required=True, help="Output Diffusers model directory.")
    parser.add_argument("--backbone", choices=["dit", "sit"], required=True)
    parser.add_argument("--model-size", choices=sorted(USP_MODEL_PRESETS), default="dit-xl")
    parser.add_argument("--sample-size", type=int, default=32, help="Latent spatial size (image_size // 8).")
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-load", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"

    state_dict = _load_state_dict(args.checkpoint)
    preset = dict(USP_MODEL_PRESETS[args.model_size])
    preset["backbone"] = args.backbone

    config = {
        "sample_size": args.sample_size,
        "patch_size": args.patch_size,
        "in_channels": args.in_channels,
        "class_dropout_prob": args.class_dropout_prob,
        "num_classes": args.num_classes,
        "learn_sigma": args.learn_sigma,
        **preset,
    }

    if args.check_load:
        model = USPTransformer2DModel(**config)
        model.load_state_dict(state_dict, strict=True)
        state_dict = model.state_dict()

    _save_config(transformer_dir, config)
    _save_weights(transformer_dir, state_dict, args.safe_serialization)

    if args.backbone == "dit":
        from diffusers.schedulers import DDIMScheduler

        scheduler = DDIMScheduler(num_train_timesteps=1000)
        pipeline_class = "USPDiTPipeline"
        scheduler_class = "DDIMScheduler"
    else:
        scheduler = USPFlowMatchScheduler()
        pipeline_class = "USPSiTPipeline"
        scheduler_class = "USPFlowMatchScheduler"

    scheduler.save_pretrained(scheduler_dir)
    _write_model_index(output_dir, pipeline_class, scheduler_class, args.vae)

    with open(output_dir / "vae_pretrained_model_name_or_path.txt", "w", encoding="utf-8") as handle:
        handle.write(args.vae + "\n")

    print(f"Saved Diffusers pipeline to {output_dir}")


if __name__ == "__main__":
    main()
