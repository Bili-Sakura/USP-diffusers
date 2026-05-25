#!/usr/bin/env python3
# Copyright 2025 USP Authors. SPDX-License-Identifier: MIT

import argparse
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from register_usp import register_usp_diffusers

register_usp_diffusers()

from diffusers.models.transformers.transformer_usp import USPTransformer2DModel
from diffusers.pipelines.usp.pipeline_usp_sit import USPSiTPipeline
from diffusers.schedulers.scheduling_flow_match_usp import USPFlowMatchScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a USP SiT Diffusers checkpoint directory.")
    parser.add_argument("--model", required=True, help="Converted Diffusers directory or transformer subfolder.")
    parser.add_argument("--class-label", type=int, default=207)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--num-inference-steps", type=int, default=250)
    parser.add_argument("--scheduler-mode", choices=["ode", "sde"], default="ode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="sample_sit.png")
    parser.add_argument("--vae", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model)
    transformer_path = model_dir / "transformer" if (model_dir / "transformer").exists() else model_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    transformer = USPTransformer2DModel.from_pretrained(str(transformer_path), torch_dtype=dtype).to(device)

    vae_name = args.vae
    if vae_name is None:
        vae_file = model_dir / "vae_pretrained_model_name_or_path.txt"
        vae_name = vae_file.read_text(encoding="utf-8").strip() if vae_file.exists() else "stabilityai/sd-vae-ft-mse"

    vae = AutoencoderKL.from_pretrained(vae_name, torch_dtype=dtype).to(device)
    if (model_dir / "scheduler").exists():
        scheduler = USPFlowMatchScheduler.from_pretrained(model_dir / "scheduler")
    else:
        scheduler = USPFlowMatchScheduler(mode=args.scheduler_mode)

    pipe = USPSiTPipeline(transformer=transformer, vae=vae, scheduler=scheduler)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipe(
        class_labels=[args.class_label],
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    )
    result.images[0].save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
