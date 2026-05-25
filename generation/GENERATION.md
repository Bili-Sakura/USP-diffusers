# Image generation (Diffusers)

USP finetuned DiT/SiT weights are consumed through native **Diffusers** components in `src/diffusers`. See [README_DIFFUSERS.md](../README_DIFFUSERS.md) for full API and layout details.

Finetuned weights are published on [Hugging Face](https://huggingface.co/GD-ML/USP-Image_Generation/tree/main) as legacy `.pt` checkpoints. Convert them once, then sample with the Diffusers pipelines.

## Convert checkpoints

```bash
pip install -e ".[dev]"

# DiT (DDIM)
python scripts/convert_usp_to_diffusers.py \
  --checkpoint /path/to/DiT-XL-2-VAE-simple.pt \
  --output usp-dit-xl-diffusers \
  --backbone dit \
  --model-size dit-xl \
  --check-load

# SiT (flow matching)
python scripts/convert_usp_to_diffusers.py \
  --checkpoint /path/to/SiT-XL-2-VAE-simple.pt \
  --output usp-sit-xl-diffusers \
  --backbone sit \
  --model-size sit-xl \
  --check-load
```

Supported sizes: `dit-b`, `dit-l`, `dit-xl`, `sit-b`, `sit-xl` (maps to USP `DiT-*-VAE-simple` / `SiT-*-VAE-simple` configs).

## Inference

```bash
python scripts/sample_usp_dit.py \
  --model usp-dit-xl-diffusers \
  --class-label 207 \
  --guidance-scale 4.0 \
  --num-inference-steps 250 \
  --output sample.png

python scripts/sample_usp_sit.py \
  --model usp-sit-xl-diffusers \
  --class-label 207 \
  --guidance-scale 4.0 \
  --num-inference-steps 250 \
  --scheduler-mode ode \
  --output sample.png
```

## Evaluation (FID / IS)

Use the converted pipeline to export samples (e.g. NPZ batches for [ADM evaluation](https://github.com/openai/guided-diffusion/tree/main/evaluations)), or adapt `scripts/sample_usp_*.py` for distributed export.

Reference batch: [VIRTUAL_imagenet256_labeled.npz](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz).

## Training

Finetuning scripts previously under `generation/DiT` and `generation/SiT` were removed in favor of Diffusers-native modules. For new training, build on `USPTransformer2DModel` with Diffusers schedulers (`DDIMScheduler` for DiT, `USPFlowMatchScheduler` for SiT) and your preferred training loop (Accelerate / Trainer).

Pretrained initialization weights remain available on Hugging Face; load via `USPTransformer2DModel.from_pretrained` after conversion, or map keys with `convert_usp_to_diffusers.py --check-load`.
