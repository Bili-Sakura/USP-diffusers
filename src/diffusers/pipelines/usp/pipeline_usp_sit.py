# Copyright 2025 USP Authors. SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from ...models.transformers.transformer_usp import USPTransformer2DModel
from ...schedulers.scheduling_flow_match_usp import USPFlowMatchScheduler


class USPSiTPipeline(DiffusionPipeline):
    """
    Class-conditional image generation with USP-finetuned SiT and flow-matching sampling.
    """

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer: USPTransformer2DModel,
        vae: AutoencoderKL,
        scheduler: USPFlowMatchScheduler,
        id2label: Optional[dict] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler)
        self.labels = {}
        if id2label:
            for key, value in id2label.items():
                for label in value.split(","):
                    self.labels[label.strip()] = int(key)
            self.labels = dict(sorted(self.labels.items()))

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        if isinstance(label, str):
            label = [label]
        missing = [item for item in label if item not in self.labels]
        if missing:
            raise ValueError(f"Unknown label(s): {missing}")
        return [self.labels[item] for item in label]

    @torch.no_grad()
    def __call__(
        self,
        class_labels: Union[int, List[int]],
        guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 250,
        output_type: str = "pil",
        return_dict: bool = True,
        vae_scale_factor: float = 0.18215,
    ) -> Union[ImagePipelineOutput, Tuple]:
        if isinstance(class_labels, int):
            class_labels = [class_labels]
        batch_size = len(class_labels)
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        latents = randn_tensor(
            (batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        )

        class_tensor = torch.tensor(class_labels, device=self._execution_device, dtype=torch.long)
        if guidance_scale > 1.0:
            latents = torch.cat([latents, latents], dim=0)
            null_labels = torch.full((batch_size,), self.transformer.config.num_classes, device=self._execution_device)
            class_input = torch.cat([class_tensor, null_labels], dim=0)
        else:
            class_input = class_tensor

        self.scheduler.set_timesteps(num_inference_steps, device=self._execution_device)
        timesteps = self.scheduler.timesteps

        for step_index, t in enumerate(self.progress_bar(timesteps[:-1])):
            prev_t = timesteps[step_index + 1]
            if guidance_scale > 1.0:
                half = latents[: latents.shape[0] // 2]
                latent_input = torch.cat([half, half], dim=0)
            else:
                latent_input = latents

            flow_t = t.expand(latent_input.shape[0]).to(device=latent_input.device, dtype=latent_input.dtype)

            if guidance_scale > 1.0:
                velocity = self.transformer.forward_with_cfg(
                    latent_input, flow_t, class_input, cfg_scale=guidance_scale
                )
                if self.transformer.config.learn_sigma:
                    velocity, _ = torch.split(velocity, latent_channels, dim=1)
            else:
                velocity = self.transformer(latent_input, flow_t, class_input, return_dict=False)[0]
                if self.transformer.config.learn_sigma:
                    velocity, _ = torch.split(velocity, latent_channels, dim=1)

            latents = self.scheduler.step(
                velocity,
                t,
                latents,
                prev_timestep=prev_t,
                final_step=step_index == len(timesteps) - 2,
            ).prev_sample

        if guidance_scale > 1.0:
            latents, _ = latents.chunk(2, dim=0)

        latents = latents / vae_scale_factor
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
