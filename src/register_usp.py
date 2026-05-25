# Copyright 2025 USP Authors. SPDX-License-Identifier: MIT
"""Register USP components under the installed Hugging Face `diffusers` package."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parent
_USP_ROOT = _REPO_SRC / "diffusers"


def _import_hf_diffusers():
    """Import the Hugging Face diffusers distribution, not the local `src/diffusers` tree."""
    src_root = str(_REPO_SRC.resolve())
    filtered_path = [entry for entry in sys.path if Path(entry).resolve() != Path(src_root)]
    removed = len(filtered_path) != len(sys.path)
    if removed:
        sys.path[:] = filtered_path
    try:
        importlib.invalidate_caches()
        if "diffusers" in sys.modules:
            mod = sys.modules["diffusers"]
            mod_file = getattr(mod, "__file__", "") or ""
            if mod_file and src_root in mod_file:
                for name in list(sys.modules):
                    if name == "diffusers" or name.startswith("diffusers."):
                        del sys.modules[name]
        return importlib.import_module("diffusers")
    finally:
        if removed:
            sys.path.insert(0, src_root)


def register_usp_diffusers() -> None:
    """Load USP modules from `src/diffusers` into `sys.modules` under `diffusers.*`."""
    _import_hf_diffusers()

    for module_name in (
        "diffusers.models.autoencoders.autoencoder_kl",
        "diffusers.pipelines.pipeline_utils",
        "diffusers.image_processor",
        "diffusers.schedulers.scheduling_ddim",
        "diffusers.utils.torch_utils",
    ):
        importlib.import_module(module_name)

    _ensure_package("diffusers.models.transformers")
    _load_module(
        "diffusers.models.transformers.transformer_usp",
        _USP_ROOT / "models" / "transformers" / "transformer_usp.py",
    )
    _ensure_package("diffusers.schedulers")
    _load_module(
        "diffusers.schedulers.scheduling_flow_match_usp",
        _USP_ROOT / "schedulers" / "scheduling_flow_match_usp.py",
    )
    _ensure_package("diffusers.pipelines.usp")
    _load_module(
        "diffusers.pipelines.usp.pipeline_usp_dit",
        _USP_ROOT / "pipelines" / "usp" / "pipeline_usp_dit.py",
    )
    _load_module(
        "diffusers.pipelines.usp.pipeline_usp_sit",
        _USP_ROOT / "pipelines" / "usp" / "pipeline_usp_sit.py",
    )

    transformer_mod = sys.modules["diffusers.models.transformers.transformer_usp"]
    sys.modules["diffusers.models.transformers"].USPTransformer2DModel = transformer_mod.USPTransformer2DModel

    usp_sched = sys.modules["diffusers.schedulers.scheduling_flow_match_usp"]
    importlib.import_module("diffusers.schedulers")
    sys.modules["diffusers.schedulers"].USPFlowMatchScheduler = usp_sched.USPFlowMatchScheduler

    usp_pipelines = sys.modules["diffusers.pipelines.usp"]
    usp_pipelines.USPDiTPipeline = sys.modules["diffusers.pipelines.usp.pipeline_usp_dit"].USPDiTPipeline
    usp_pipelines.USPSiTPipeline = sys.modules["diffusers.pipelines.usp.pipeline_usp_sit"].USPSiTPipeline


def _ensure_package(name: str) -> None:
    """Create namespace packages without importing heavy Diffusers subpackage inits."""
    if name in sys.modules:
        return
    parent, _, child = name.rpartition(".")
    if parent:
        _ensure_package(parent)
    module = importlib.util.module_from_spec(importlib.machinery.ModuleSpec(name, loader=None))
    module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module
    if parent:
        setattr(sys.modules[parent], child, module)


def _load_module(module_name: str, path: Path) -> None:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
