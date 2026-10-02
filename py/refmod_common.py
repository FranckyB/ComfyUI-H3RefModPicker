"""Shared helpers for the RefMod pack: media loading and the refmods folder."""

from __future__ import annotations

import os
from typing import List, Optional

import torch

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def refmods_dir() -> str:
    """ComfyUI models/refmods — the mod storage folder, created on first use.

    Sits next to loras/, unet/ etc. instead of inside the pack folder, so mods
    live in the standard model tree.
    """
    import folder_paths
    d = os.path.join(folder_paths.models_dir, "refmods")
    os.makedirs(d, exist_ok=True)
    return d


def refmods_dirs() -> List[str]:
    """All RefMod folders ComfyUI knows about, write folder first.

    Includes folders declared under ``refmods:`` in extra_model_paths.yaml
    (or --extra-model-paths-config). Duplicates are removed by real path,
    since a root may be a symlink to another one.
    """
    import folder_paths

    candidates = [refmods_dir()]
    try:
        candidates += list(folder_paths.get_folder_paths("refmods"))
    except Exception:
        pass

    roots: List[str] = []
    seen = set()
    for directory in candidates:
        if not os.path.isdir(directory):
            continue
        real = os.path.realpath(directory)
        if real in seen:
            continue
        seen.add(real)
        roots.append(os.path.abspath(directory))
    return roots


def list_image_files(folder: str) -> List[str]:
    """Images directly under ``folder`` (top level only), sorted by name."""
    images = []
    if os.path.isdir(folder):
        for fn in sorted(os.listdir(folder)):
            ext = os.path.splitext(fn)[1].lower()
            p = os.path.join(folder, fn)
            if os.path.isfile(p):
                if ext in IMAGE_EXTS:
                    images.append(p)
    return images


def load_image_file(path: str, max_edge: Optional[int] = None) -> torch.Tensor:
    """Load one image file -> [1, H, W, 3] float32 in [0, 1]."""
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        w, h = img.size
        if max_edge is not None:
            scale = min(1.0, max_edge / max(w, h))
            if scale < 1.0:
                img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                 Image.LANCZOS)
        arr = torch.from_numpy(np.asarray(img).copy()).float() / 255.0
    return arr.unsqueeze(0)  # [1, H, W, 3]


def _prompt_hint(loads) -> str:
    """Merge loaded mods' concept_type + description into one prompt-ready string.

    e.g. "identity: ginger woman, tattooed neck, black lipstick; pose_motion:
    slow twirl into camera, hair whipping". Concat this onto your positive
    prompt (a string-concat node ahead of CLIP Text Encode) instead of
    retyping each mod's description by hand. Mods with no description are
    skipped — a bare concept_type with nothing to say isn't a useful clue.
    """
    parts = []
    for mod, _strength in loads:
        if mod.description:
            parts.append(f"{mod.concept_type}: {mod.description}")
    return "; ".join(parts)
