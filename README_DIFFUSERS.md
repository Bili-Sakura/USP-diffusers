# USP Diffusers integration

Image generation for USP-finetuned **DiT** and **SiT** (VAE-simple blocks) is implemented as native [Diffusers](https://github.com/huggingface/diffusers) components under `src/diffusers`, following the layout of [NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers.git).

## Layout

| Path | Component |
|------|-----------|
| `src/diffusers/models/transformers/transformer_usp.py` | `USPTransformer2DModel` (`ModelMixin` / `ConfigMixin`) |
| `src/diffusers/schedulers/scheduling_flow_match_usp.py` | `USPFlowMatchScheduler` (SiT flow matching) |
| `src/diffusers/pipelines/usp/pipeline_usp_dit.py` | `USPDiTPipeline` + `DDIMScheduler` |
| `src/diffusers/pipelines/usp/pipeline_usp_sit.py` | `USPSiTPipeline` + flow scheduler |
| `src/register_usp.py` | Registers extensions into installed `diffusers` |
| `scripts/convert_usp_to_diffusers.py` | Legacy `.pt` → Diffusers directory |
| `scripts/sample_usp_dit.py` / `sample_usp_sit.py` | Inference |

Legacy `generation/DiT` and `generation/SiT` vendored code has been removed. Pretraining under `pretrain/` is unchanged.

## Install

```bash
pip install -e ".[dev]"
```

Requires `diffusers>=0.30.1`, `torch`, and `timm`.

## Convert Hugging Face / local checkpoints

```bash
python scripts/convert_usp_to_diffusers.py \
  --checkpoint /path/to/DiT-XL-2-VAE-simple.pt \
  --output usp-dit-xl-diffusers \
  --backbone dit \
  --model-size dit-xl \
  --check-load

python scripts/convert_usp_to_diffusers.py \
  --checkpoint /path/to/SiT-XL-2-VAE-simple.pt \
  --output usp-sit-xl-diffusers \
  --backbone sit \
  --model-size sit-xl \
  --check-load
```

Model sizes: `dit-b`, `dit-l`, `dit-xl`, `sit-b`, `sit-xl`.

Output layout:

```text
model_index.json
scheduler/
transformer/config.json
transformer/diffusion_pytorch_model.safetensors
vae_pretrained_model_name_or_path.txt
```

Weights use the same state-dict keys as the original USP `models.py` (e.g. `x_embedder.proj.weight`, `blocks.0.norm1.weight`).

## Sample

```bash
python scripts/sample_usp_dit.py \
  --model usp-dit-xl-diffusers \
  --class-label 207 \
  --guidance-scale 4.0 \
  --num-inference-steps 250

python scripts/sample_usp_sit.py \
  --model usp-sit-xl-diffusers \
  --class-label 207 \
  --scheduler-mode ode \
  --num-inference-steps 250
```

## Python API

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("./src").resolve()))
from register_usp import register_usp_diffusers

register_usp_diffusers()

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models.transformers.transformer_usp import USPTransformer2DModel
from diffusers.pipelines.usp.pipeline_usp_dit import USPDiTPipeline

transformer = USPTransformer2DModel.from_pretrained("./usp-dit-xl-diffusers/transformer")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
scheduler = DDIMScheduler.from_pretrained("./usp-dit-xl-diffusers/scheduler")
pipe = USPDiTPipeline(transformer=transformer, vae=vae, scheduler=scheduler).to("cuda")
images = pipe(class_labels=[207], num_inference_steps=250).images
```

## Upstreaming

To land in `huggingface/diffusers`, copy files from `src/diffusers` into the matching package paths and register classes in Diffusers lazy-import tables (same workflow as NiT-diffusers).

## Tests

```bash
pytest tests/test_usp_diffusers.py
```
