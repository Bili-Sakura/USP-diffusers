# Copyright 2025 USP Authors. SPDX-License-Identifier: MIT
"""Flow-matching schedulers for USP SiT inference (ODE / SDE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.schedulers.scheduling_utils import SchedulerMixin
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class _Config(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error

    class ConfigMixin:
        config_name = "scheduler_config.json"

    class SchedulerMixin:
        def __init__(self):
            self.config = _Config()

    def register_to_config(init):
        def wrapper(self, *args, **kwargs):
            import inspect

            signature = inspect.signature(init)
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self.config = _Config({key: value for key, value in bound.arguments.items() if key != "self"})
            init(self, *args, **kwargs)

        return wrapper


@dataclass
class USPFlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class USPFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler flow-matching integrator used by USP SiT sampling.
    """

    config_name = "scheduler_config.json"

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        mode: str = "ode",
        path_type: str = "linear",
        train_eps: float = 1e-5,
        sample_eps: float = 1e-3,
    ):
        super().__init__()
        if mode not in {"ode", "sde"}:
            raise ValueError("mode must be 'ode' or 'sde'")
        if path_type not in {"linear", "cosine"}:
            raise ValueError("path_type must be 'linear' or 'cosine'")
        self.timesteps: torch.Tensor = torch.tensor([])

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)

    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        prev_timestep: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        final_step: bool = False,
        **kwargs,
    ) -> Union[USPFlowMatchSchedulerOutput, Tuple[torch.Tensor]]:
        del kwargs, generator
        if prev_timestep is None:
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0]
            if index.numel() == 0:
                raise ValueError("timestep not found in scheduler.timesteps; call set_timesteps first.")
            step_index = int(index[0].item())
            prev_timestep = self.timesteps[step_index + 1] if step_index + 1 < len(self.timesteps) else torch.tensor(
                0.0, device=sample.device, dtype=sample.dtype
            )

        t = float(timestep.flatten()[0].item()) if torch.is_tensor(timestep) else float(timestep)
        t_prev = float(prev_timestep.flatten()[0].item()) if torch.is_tensor(prev_timestep) else float(prev_timestep)
        dt = t - t_prev

        if self.config.mode == "ode" or final_step:
            prev_sample = sample - dt * model_output
        else:
            noise = torch.randn_like(sample)
            diffusion = 2.0 * max(dt, self.config.sample_eps)
            prev_sample = sample - dt * model_output + torch.sqrt(diffusion) * noise

        if not return_dict:
            return (prev_sample,)
        return USPFlowMatchSchedulerOutput(prev_sample=prev_sample)
