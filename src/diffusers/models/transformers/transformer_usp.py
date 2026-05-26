# Copyright 2025 USP Authors. SPDX-License-Identifier: MIT
"""USP DiT/SiT VAE-simple transformer in Diffusers ModelMixin style."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
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
        config_name = "config.json"

    class ModelMixin(nn.Module):
        pass

    def register_to_config(init):
        def wrapper(self, *args, **kwargs):
            import inspect

            signature = inspect.signature(init)
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self.config = _Config({key: value for key, value in bound.arguments.items() if key != "self"})
            init(self, *args, **kwargs)

        return wrapper


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class USPPatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int, bias: bool = True):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class USPAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, **kwargs):
        super().__init__()
        del kwargs
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        x = F.scaled_dot_product_attention(query, key, value)
        x = x.transpose(1, 2).reshape(batch, seq_len, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class USPMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, act_layer=None, drop: float = 0.0, **kwargs):
        super().__init__()
        del kwargs, act_layer
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


@dataclass
class USPTransformer2DModelOutput(BaseOutput):
    sample: torch.FloatTensor


class USPTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(timesteps, self.frequency_embedding_size)
        return self.mlp(t_freq)


class USPLabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + int(use_cfg_embedding), hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class USPBlock(nn.Module):
    """adaLN-Zero block with elementwise-affine LayerNorm (USP VAE-simple)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        del block_kwargs
        self.attn = USPAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = USPMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(
            6, dim=1
        )
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * self.attn(
            _modulate(self.norm1(hidden_states), shift_msa, scale_msa)
        )
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)
        )
        return hidden_states


class USPFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=1)
        hidden_states = _modulate(self.norm_final(hidden_states), shift, scale)
        return self.linear(hidden_states)


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    return _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class USPTransformer2DModel(ModelMixin, ConfigMixin):
    """
    Class-conditional latent transformer for USP-finetuned DiT/SiT (VAE-simple blocks).
    Checkpoint keys match the original USP `models.py` layout for direct loading.
    """

    config_name = "config.json"
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        backbone: str = "dit",
    ):
        super().__init__()
        del backbone  # dit vs sit share architecture; kept for Hub metadata
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = USPPatchEmbed(sample_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = USPTimestepEmbedder(hidden_size)
        self.y_embedder = USPLabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList(
            [USPBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = USPFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        grid = int(self.x_embedder.num_patches**0.5)
        pos_embed = _get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, hidden_states: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], h, w, p, p, c)
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(hidden_states.shape[0], c, h * p, h * p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: torch.LongTensor,
        return_dict: bool = True,
    ) -> Union[USPTransformer2DModelOutput, Tuple[torch.Tensor]]:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=hidden_states.device, dtype=torch.float32)
        timestep = timestep.to(device=hidden_states.device).reshape(-1)
        if timestep.shape[0] == 1:
            timestep = timestep.expand(hidden_states.shape[0])

        class_labels = class_labels.to(hidden_states.device, dtype=torch.long).reshape(-1)
        if class_labels.shape[0] == 1:
            class_labels = class_labels.expand(hidden_states.shape[0])

        tokens = self.x_embedder(hidden_states) + self.pos_embed
        conditioning = self.t_embedder(timestep) + self.y_embedder(class_labels, self.training)
        for block in self.blocks:
            tokens = block(tokens, conditioning)
        tokens = self.final_layer(tokens, conditioning)
        sample = self.unpatchify(tokens)

        if not return_dict:
            return (sample,)
        return USPTransformer2DModelOutput(sample=sample)

    def forward_with_cfg(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[torch.Tensor, float],
        class_labels: torch.LongTensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        half = hidden_states[: hidden_states.shape[0] // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, class_labels, return_dict=False)[0]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


USP_MODEL_PRESETS = {
    "dit-b": dict(depth=12, hidden_size=768, num_heads=12, backbone="dit"),
    "dit-l": dict(depth=24, hidden_size=1024, num_heads=16, backbone="dit"),
    "dit-xl": dict(depth=28, hidden_size=1152, num_heads=16, backbone="dit"),
    "sit-b": dict(depth=12, hidden_size=768, num_heads=12, backbone="sit"),
    "sit-xl": dict(depth=28, hidden_size=1152, num_heads=16, backbone="sit"),
}


def create_usp_transformer(
    model_size: str,
    *,
    sample_size: int = 32,
    patch_size: int = 2,
    in_channels: int = 4,
    num_classes: int = 1000,
    class_dropout_prob: float = 0.1,
    learn_sigma: bool = True,
) -> USPTransformer2DModel:
    if model_size not in USP_MODEL_PRESETS:
        raise ValueError(f"Unknown model_size {model_size}. Choose from {sorted(USP_MODEL_PRESETS)}.")
    return USPTransformer2DModel(
        sample_size=sample_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=class_dropout_prob,
        learn_sigma=learn_sigma,
        **USP_MODEL_PRESETS[model_size],
    )
