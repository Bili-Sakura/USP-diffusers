import pytest

torch = pytest.importorskip("torch")

from register_usp import register_usp_diffusers

register_usp_diffusers()

from diffusers.models.transformers.transformer_usp import USPTransformer2DModel
from diffusers.schedulers.scheduling_flow_match_usp import USPFlowMatchScheduler


def test_usp_transformer_forward():
    model = USPTransformer2DModel(
        sample_size=4,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        learn_sigma=False,
        num_classes=10,
    )
    latents = torch.randn(2, 4, 4, 4)
    timesteps = torch.tensor([999, 500])
    class_labels = torch.tensor([1, 2])
    output = model(latents, timesteps, class_labels)
    assert output.sample.shape == (2, 4, 4, 4)


def test_flow_scheduler_ode_step():
    scheduler = USPFlowMatchScheduler(mode="ode")
    scheduler.set_timesteps(4)
    sample = torch.ones(1, 4, 2, 2)
    velocity = torch.full_like(sample, 2.0)
    output = scheduler.step(velocity, torch.tensor([1.0]), sample, prev_timestep=torch.tensor([0.75]))
    assert torch.allclose(output.prev_sample, torch.full_like(sample, 0.5))


def test_legacy_key_compatibility():
    model = USPTransformer2DModel(
        sample_size=4,
        patch_size=2,
        hidden_size=32,
        depth=1,
        num_heads=4,
        learn_sigma=True,
    )
    keys = set(model.state_dict().keys())
    assert "x_embedder.proj.weight" in keys
    assert "blocks.0.norm1.weight" in keys
    assert "final_layer.linear.weight" in keys
