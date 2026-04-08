from __future__ import annotations

import os
from pathlib import Path


def hf_repo_is_cached(repo_id: str) -> bool:
    """Return True when a Hugging Face repo already exists in the local cache."""
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_dir = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"
    return repo_dir.exists() and any(repo_dir.iterdir())


def torch_hub_repo_is_cached(repo_dir_name: str) -> bool:
    """Return True when a Torch Hub repo snapshot exists locally."""
    torch_home = Path(os.getenv("TORCH_HOME", Path.home() / ".cache" / "torch"))
    repo_dir = torch_home / "hub" / repo_dir_name
    return repo_dir.exists() and any(repo_dir.iterdir())
