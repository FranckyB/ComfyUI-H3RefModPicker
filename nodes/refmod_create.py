"""
H3RefModCreateFromFolder — folder-driven extraction of reference media into a saved RefMod.
"""

from __future__ import annotations

import math
import os
import shutil
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import io

from ..py.refmod_common import (
    list_media_files,
    load_video_file,
    refmods_dir,
)
from ..py.refmod_core import (
    AUDIO_CONCEPT_TYPES,
    CONCEPT_TYPES,
    H3RefMod,
    _blur_latent,
    aspect_grid,
    fit_token_budget,
    normalize_mode,
    optimize_latent,
    optimize_latent_multi,
    pool_latent,
    save_refmod_bundle,
)
from ..py.refmod_vae_loader import load_h3_vaes_from_av_encoder
# reuse the encode helpers shared with the loader/mods-listing side of the pack
from . import refmod_loader as _nodes_mod  # loader-side cache/list refresh
from .refmod_loader import (
    _MOD_CACHE,
    _MOD_CACHE_MAX,
)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg", ".opus"}
THUMB_TARGET_ASPECT = 3.0 / 4.0
THUMB_ASPECT_TOLERANCE = 0.05
VISUAL_SUFFIX = "_visual"
AUDIO_SUFFIX = "_audio"


def _resize_ref(image, short_edge: int, canvas=None):
    """Aspect-preserving downscale (never upscale) to ``short_edge`` px; dims to /32."""
    h, w = image.shape[1], image.shape[2]
    if h <= 0 or w <= 0:
        raise ValueError(
            f"_resize_ref: source has an empty frame ({h}x{w}) before any "
            f"resize — the reference itself is invalid.")
    scale = min(1.0, short_edge / min(h, w))
    tw = max(32, round(w * scale / 32) * 32)
    th = max(32, round(h * scale / 32) * 32)
    crop = "disabled"
    if canvas is not None:
        tw, th = canvas
        crop = "center"
    if tw <= 0 or th <= 0:
        raise ValueError(
            f"_resize_ref: computed a zero-size resize target ({tw}x{th}) "
            f"for a {h}x{w} source (short_edge={short_edge}, canvas={canvas}). "
            f"This should be impossible — please report this shape.")
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, tw, th, "lanczos", crop)
    if samples.shape[2] <= 0 or samples.shape[3] <= 0:
        raise ValueError(
            f"_resize_ref: common_upscale produced an empty result "
            f"{tuple(samples.shape)} from a {h}x{w} source targeting "
            f"{tw}x{th} (crop={crop}, canvas={canvas}). This points to a bug "
            f"in comfy.utils.common_upscale for this input, not in RefMod's "
            f"own math.")
    return samples.movedim(1, -1)


def _snap_to_causal_grid(n_frames: int) -> int:
    """Round a video frame count down to the nearest valid ``4k + 1``."""
    if n_frames <= 1:
        return 1
    return ((n_frames - 1) // 4) * 4 + 1


def _ensure_min_size(image, floor: int = 320):
    """Upscale (never downscale) so both spatial dims are >= ``floor`` px."""
    h, w = image.shape[1], image.shape[2]
    if h >= floor and w >= floor:
        return image
    scale = floor / min(h, w)
    tw = max(floor, round(w * scale / 32) * 32)
    th = max(floor, round(h * scale / 32) * 32)
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, tw, th, "lanczos", "disabled")
    return samples.movedim(1, -1)


def _normalize_mask_batch(mask, label: str = "mask") -> torch.Tensor:
    """Canonicalize a MASK input to ``[N, H, W]`` float32 in [0, 1]."""
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"H3RefModExtract: {label} must be a MASK tensor, "
                         f"got {type(mask)}")
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError(f"H3RefModExtract: {label} has unexpected shape "
                         f"{tuple(mask.shape)} (expected [H,W] or [N,H,W])")
    return mask.float().clamp(0.0, 1.0)


def _resize_mask(mask: torch.Tensor, target_h: int, target_w: int, crop="disabled") -> torch.Tensor:
    """Resize a ``[T, H, W]`` mask to ``target_h x target_w`` (bilinear)."""
    samples = mask.unsqueeze(1)
    samples = comfy.utils.common_upscale(samples, target_w, target_h, "bilinear", crop)
    return samples.squeeze(1).clamp(0.0, 1.0)


def _mask_latent(z: torch.Tensor, mask_px: torch.Tensor, background_retention: float,
                 seed_key: str) -> torch.Tensor:
    """Suppress the latent outside ``mask_px`` toward a blurred copy of itself, per cell."""
    del seed_key
    t, h, w = z.shape[2], z.shape[3], z.shape[4]
    mp = mask_px.unsqueeze(1)
    if mp.shape[0] == 1 and t > 1:
        mp = mp.expand(t, -1, -1, -1)
    elif mp.shape[0] != t:
        idx = torch.linspace(0, mp.shape[0] - 1, t).round().long()
        mp = mp[idx]
    mp = F.adaptive_avg_pool2d(mp.float(), (h, w))
    mp = mp.permute(1, 0, 2, 3).unsqueeze(0).clamp(0.0, 1.0)
    weight = background_retention + (1.0 - background_retention) * mp
    blurred = _blur_latent(z)
    return (weight * z.float() + (1.0 - weight) * blurred).to(z.dtype)


def _normalize_ref(src, label: str = "reference") -> torch.Tensor:
    """Canonicalize any ref source to ``[T, H, W, C]`` (T=1 for stills)."""
    if not isinstance(src, torch.Tensor) or src.dim() not in (3, 4, 5):
        raise ValueError(
            f"H3RefModExtract: {label} must be a 3-5D tensor, "
            f"got {getattr(src, 'shape', src)}")
    if src.dim() == 5:
        if src.shape[0] == 0:
            raise ValueError(
                f"H3RefModExtract: {label} has no frames "
                f"(T=0) — check the source image/video.")
        src = src[0] if src.shape[0] == 1 else src.reshape(-1, *src.shape[2:])
    if src.dim() == 3:
        src = src.unsqueeze(0)
    if src.shape[-1] != 3 and src.shape[1] == 3:
        src = src.movedim(1, -1)
    if src.dim() != 4 or src.shape[-1] != 3:
        raise ValueError(
            f"H3RefModExtract: {label} has an unexpected layout "
            f"{tuple(src.shape)} (expected [T, H, W, 3])")
    if src.shape[0] <= 0:
        raise ValueError(
            f"H3RefModExtract: {label} has no frames (T={src.shape[0]}) "
            f"— check the source image/video.")
    if src.shape[1] <= 0 or src.shape[2] <= 0:
        raise ValueError(
            f"H3RefModExtract: {label} has an empty frame "
            f"({src.shape[1]}x{src.shape[2]}) — check the source image/video.")
    return src


def _sanitize_name(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    if not name:
        raise ValueError("mod name must not be empty")
    return name


def _resolve_folder(folder: str) -> str:
    """Resolve a folder input: absolute path, a name inside input/, or input/ itself."""
    folder = (folder or "").strip().strip('"')
    if not folder:
        return folder_paths.get_input_directory()
    if os.path.isabs(folder):
        resolved = os.path.normpath(folder)
    else:
        resolved = os.path.join(folder_paths.get_input_directory(), folder)
    if not os.path.isdir(resolved):
        raise ValueError(
            f"folder not found: {folder!r} (looked at '{resolved}'; use an "
            "absolute path or a folder name inside input/).")
    return resolved


def _summarize(mod: H3RefMod) -> str:
    if mod.kind == "audio":
        return f"{mod.name}: audio, {mod.latent_t / 40:.2f}s, {mod.token_count} tokens"
    mb = mod.latent.numel() * mod.latent.element_size() / 1024 / 1024
    return (f"'{mod.name}' {mod.mode} {mod.kind} {tuple(mod.latent.shape)} "
            f"({mod.token_count} tokens, {mb:.2f} MB)")


def _auto_folder_description(folder: str, concept_type: str) -> str:
    """Default metadata description for folder-generated RefMods."""
    folder_label = os.path.basename(os.path.normpath(folder)) or "refmod"
    if concept_type == "identity":
        return f"{folder_label} persona reference mod"
    if concept_type and concept_type != "generic":
        return f"{folder_label} {concept_type} reference mod"
    return f"{folder_label} reference mod"


def _folder_refmod_name(folder: str) -> str:
    """RefMod base name derived from a folder, matching subfolder mode."""
    folder_name = _sanitize_name(os.path.basename(os.path.normpath(folder)))
    if "refmod" not in folder_name.lower():
        folder_name = f"{folder_name}_refMod"
    return folder_name


# ═══════════════════════════════════════════════════════════════════════════
# Folder scanning (images / videos / audio)
# ═══════════════════════════════════════════════════════════════════════════

def _scan_folder(folder: str) -> "tuple[List[str], List[str], List[str]]":
    """(images, videos, audios) directly under ``folder``, sorted by name."""
    images, videos = list_media_files(folder)
    audios = []
    if os.path.isdir(folder):
        for fn in sorted(os.listdir(folder)):
            if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                p = os.path.join(folder, fn)
                if os.path.isfile(p):
                    audios.append(p)
    if not images and not videos and not audios:
        print(
            f"[H3RefModCreateFromFolder] scan: '{folder}' contains no supported image, video, or audio files "
            "at the top level."
        )
    return images, videos, audios


def _image_size(path: str) -> "tuple[int, int]":
    from PIL import Image

    with Image.open(path) as img:
        return img.size


def _load_image_and_alpha_mask_file(path: str, max_edge: Optional[int] = None) -> "tuple[torch.Tensor, Optional[torch.Tensor]]":
    """Load one image file and an optional embedded alpha mask.

    Returns ``(image, mask)`` where image is ``[1, H, W, 3]`` RGB float32 in
    ``[0, 1]`` and mask is ``[1, H, W]`` float32 in ``[0, 1]`` when the source
    contains a non-trivial alpha channel. Fully opaque images return ``None``
    for the mask.
    """
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        rgba = img.convert("RGBA")
        rgb = rgba.convert("RGB")
        alpha = rgba.getchannel("A")
        w, h = rgb.size
        if max_edge is not None:
            scale = min(1.0, max_edge / max(w, h))
            if scale < 1.0:
                size = (max(1, round(w * scale)), max(1, round(h * scale)))
                rgb = rgb.resize(size, Image.LANCZOS)
                alpha = alpha.resize(size, Image.BILINEAR)
        image = torch.from_numpy(np.asarray(rgb).copy()).float() / 255.0
        alpha_arr = np.asarray(alpha).copy()
    mask = None
    if alpha_arr.min() < 255:
        mask = torch.from_numpy(alpha_arr).float() / 255.0
        mask = mask.unsqueeze(0)
    return image.unsqueeze(0), mask


def _is_preferred_thumb_aspect(width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    return abs((width / float(height)) - THUMB_TARGET_ASPECT) <= THUMB_ASPECT_TOLERANCE


def _choose_thumb_source(images: List[str]) -> Optional[str]:
    """Prefer a 3:4 image; otherwise use the largest image. Stable on ties."""
    ranked = []
    for idx, path in enumerate(images):
        try:
            width, height = _image_size(path)
        except Exception:
            continue
        ranked.append((
            0 if _is_preferred_thumb_aspect(width, height) else 1,
            -(width * height),
            idx,
            path,
        ))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][3]


def _copy_refmod_thumbnail(images: List[str], path_no_ext: str) -> None:
    """Copy a representative source image beside the saved RefMod."""
    thumb_src = _choose_thumb_source(images)
    if not thumb_src:
        return
    ext = os.path.splitext(thumb_src)[1].lower()
    thumb_base = path_no_ext
    if thumb_base.lower().endswith(VISUAL_SUFFIX.lower()):
        thumb_base = thumb_base[:-len(VISUAL_SUFFIX)]
    thumb_dst = thumb_base + ext
    shutil.copy2(thumb_src, thumb_dst)
    print(f"[H3RefModCreateFromFolder] thumbnail {os.path.basename(thumb_src)} -> {thumb_dst}")


def _latent_token_count(latent: torch.Tensor) -> int:
    return int(latent.shape[2]) * (int(latent.shape[3]) // 2) * (int(latent.shape[4]) // 2)


def _resolve_output_dir(subfolder: str = "") -> str:
    base = refmods_dir()
    subfolder = str(subfolder or "").strip().strip('"').strip("/\\")
    return os.path.join(base, subfolder) if subfolder else base


def _base_refmod_name(name: str) -> str:
    base = _sanitize_name(name)
    lower_base = base.lower()
    for suffix in (VISUAL_SUFFIX, AUDIO_SUFFIX):
        if lower_base.endswith(suffix):
            base = base[:-len(suffix)]
            lower_base = base.lower()
    return base or _sanitize_name(name)


def _paired_refmod_names(base_name: str) -> Tuple[str, str]:
    base = _base_refmod_name(base_name)
    return f"{base}{VISUAL_SUFFIX}", f"{base}{AUDIO_SUFFIX}"


def _unique_split_mod_paths(out_dir: str, base_name: str, include_audio: bool) -> Tuple[str, str, str, str, str]:
    root = _base_refmod_name(base_name)
    candidate = root
    attempt = 1
    while True:
        visual_name, audio_name = _paired_refmod_names(candidate)
        visual_path = os.path.join(out_dir, visual_name)
        audio_path = os.path.join(out_dir, audio_name)
        visual_exists = os.path.isfile(visual_path + ".safetensors")
        audio_exists = os.path.isfile(audio_path + ".safetensors")
        if not visual_exists and not audio_exists:
            return candidate, visual_name, visual_path, audio_name, audio_path
        attempt += 1
        candidate = f"{root}_{attempt}"


def _unique_bundle_mod_path(out_dir: str, base_name: str) -> Tuple[str, str]:
    root = _base_refmod_name(base_name)
    candidate = root
    attempt = 1
    while True:
        bundle_path = os.path.join(out_dir, candidate)
        if not os.path.isfile(bundle_path + ".safetensors"):
            return candidate, bundle_path
        attempt += 1
        candidate = f"{root}_{attempt}"


def _make_audio_refmod(name: str, audio_latent: torch.Tensor, audio_concept_type: str,
                       description: str, sample_rate: int = 32000) -> H3RefMod:
    return H3RefMod(
        name=name,
        kind="audio",
        latent=audio_latent,
        mode="encode",
        source="audio",
        source_shape=f"audio:{int(audio_latent.shape[-1])}",
        pool=f"{int(audio_latent.shape[-1])} audio",
        tags=[f"{int(audio_latent.shape[-1])} audio", f"audio:{audio_concept_type}"],
        description=description,
        concept_type=audio_concept_type,
        audio_concept_type=audio_concept_type,
        sample_rate=sample_rate,
    )


def _apply_extraction_preset(mode, ref_resolution, pool_h, pool_w, identity,
                             merge, motion_only, concept_type,
                             audio_concept_type, extraction_preset, log_prefix):
    preset_replaced = ""
    if extraction_preset == "identity_encode":
        concept_type, audio_concept_type = "identity", "voice"
        mode, ref_resolution, identity, merge, motion_only = "encode", 1024, 0, False, False
        preset_replaced = "concept_type=identity, audio_concept_type=voice, mode=Full Reference, ref_resolution=1024, Refinement Steps=0, merge=False, motion_only=False"
    elif extraction_preset == "style_experimental":
        concept_type, audio_concept_type = "style", "voice"
        mode, pool_h, pool_w, identity, merge, motion_only = "training", 8, 8, 150, False, False
        preset_replaced = "concept_type=style, audio_concept_type=voice, mode=Compressed Reference, pool_h=8, pool_w=8, Refinement Steps=150, merge=False, motion_only=False"
    elif extraction_preset == "motion_sequence":
        concept_type, audio_concept_type = "pose_motion", "voice"
        mode, pool_h, pool_w, merge, motion_only = "training", 16, 16, False, False
        preset_replaced = "concept_type=pose_motion, audio_concept_type=voice, mode=Compressed Reference, pool_h=16, pool_w=16, merge=False, motion_only=False (frame limit and Refinement Steps preserved)"
    elif extraction_preset != "manual":
        raise ValueError("Unknown extraction preset.")
    mode = normalize_mode(mode)
    if preset_replaced:
        print(f"[{log_prefix}] preset={extraction_preset} replaces {preset_replaced}")
    return mode, ref_resolution, pool_h, pool_w, identity, merge, motion_only, concept_type, audio_concept_type


# ═══════════════════════════════════════════════════════════════════════════
# Audio loading + encoding
# ═══════════════════════════════════════════════════════════════════════════

def _load_audio_waveform(path: str) -> "tuple[torch.Tensor, int]":
    """Load an audio file -> (waveform [C, L] float32, sample_rate)."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform.float(), sr
    except Exception:
        pass
    try:
        import soundfile as sf
        data, sr = sf.read(path, always_2d=True)  # [L, C]
        return torch.from_numpy(data.T).float(), sr
    except Exception:
        pass
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    t = torch.from_numpy(data).float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    else:
        t = t.T
    if data.dtype.kind in ("i", "u"):
        t = t / float(2 ** (8 * data.dtype.itemsize - 1))
    return t, sr


def _resample(waveform: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    """Resample [C, L] -> [C, L'] via torchaudio, falling back to scipy."""
    try:
        import torchaudio
        return torchaudio.functional.resample(waveform, sr, target_sr)
    except Exception:
        import numpy as np
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, target_sr)
        out = resample_poly(waveform.numpy(), target_sr // g, sr // g, axis=-1)
        return torch.from_numpy(np.ascontiguousarray(out)).float()


def _encode_ref_audio(audio_vae, waveform: torch.Tensor, sr: int, device) -> torch.Tensor:
    """Encode a waveform to an H3 audio latent [1, 32, 2, T]."""
    import comfy.model_management
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = _resample(waveform, sr, vae_sr)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)      # mono -> stereo
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    batch = waveform.unsqueeze(0)             # [1, 2, L]
    model = audio_vae.first_stage_model
    last_err = None
    for dev in (device, torch.device("cpu")):
        try:
            if dev.type == "cuda":
                comfy.model_management.load_models_gpu([audio_vae.patcher])
            else:
                model.to(dev)
            with torch.no_grad():
                z = model.encode(batch.to(dev)).float().cpu()
            if dev.type == "cuda":
                try:
                    audio_vae.patcher.unpatch_model()
                except Exception:
                    pass
            return z
        except (torch.OutOfMemoryError, RuntimeError) as e:
            last_err = e
            if "out of memory" not in str(e).lower() and "OOM" not in str(e):
                raise
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to encode folder audio with the H3 audio VAE.")


def _encode_input_audio(audio_vae, audio, max_seconds: float = 30.0,
                        chunk_seconds: float = 10.0):
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2):
        raise ValueError("Audio must have one batch of mono or stereo samples [1,C,L].")
    if sample_rate <= 0 or waveform.shape[-1] < 1:
        raise ValueError("Audio is empty or has an invalid sample rate.")
    if (not math.isfinite(max_seconds) or not math.isfinite(chunk_seconds)
            or max_seconds <= 0 or chunk_seconds <= 0):
        raise ValueError("Audio duration and chunk length must be positive.")

    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    waveform = waveform[..., :max(1, round(max_seconds * sample_rate))]
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    if sample_rate != vae_sr:
        waveform = _resample(waveform[0], sample_rate, vae_sr).unsqueeze(0)

    model = audio_vae.first_stage_model
    chunk = max(800, round(chunk_seconds * 40) * 800)
    last_err = None
    for dev in (comfy.model_management.get_torch_device(), torch.device("cpu")):
        try:
            if dev.type == "cuda":
                comfy.model_management.load_models_gpu([audio_vae.patcher])
            else:
                model.to(dev)
            latents = []
            with torch.no_grad():
                for start in range(0, waveform.shape[-1], chunk):
                    piece = waveform[..., start:start + chunk].to(dev)
                    z = model.encode(piece).float().cpu()
                    if z.ndim != 4 or tuple(z.shape[:3]) != (1, 32, 2):
                        raise ValueError(f"H3 audio VAE returned an invalid latent: {tuple(z.shape)}")
                    latents.append(z)
            if dev.type == "cuda":
                try:
                    audio_vae.patcher.unpatch_model()
                except Exception:
                    pass
            return torch.cat(latents, dim=-1)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            last_err = e
            if "out of memory" not in str(e).lower() and "oom" not in str(e).lower():
                raise
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to encode input audio with the H3 audio VAE.")


def _make_audio_refmod_from_input(name: str, audio_vae, audio, max_seconds: float = 30.0,
                                  max_tokens: int = 5120, budget_policy: str = "error",
                                  description: str = "", concept_type: str = "voice") -> H3RefMod:
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
        raise ValueError("Audio token budget must be a non-negative integer.")
    if budget_policy not in ("error", "truncate"):
        raise ValueError("Audio budget policy must be error or truncate.")
    latent = _encode_input_audio(audio_vae, audio, max_seconds=max_seconds)
    tokens = latent.shape[-1] * 2
    if max_tokens and tokens > max_tokens:
        if max_tokens < 2 or budget_policy == "error":
            raise ValueError(
                f"Audio requires {tokens} tokens; budget is {max_tokens}. "
                "Lower audio_max_seconds or choose truncate."
            )
        latent = latent[..., :max_tokens // 2].clone()
    return _make_audio_refmod(
        name, latent, concept_type, description,
        sample_rate=int(getattr(audio_vae, "audio_sample_rate", 32000)),
    )


def _crop_audio_latent(latent: torch.Tensor, max_tokens: int, budget_policy: str, label: str) -> torch.Tensor:
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
        raise ValueError("Audio token budget must be a non-negative integer.")
    if budget_policy not in ("error", "truncate"):
        raise ValueError("Audio budget policy must be error or truncate.")
    tokens = int(latent.shape[-1]) * 2
    if max_tokens and tokens > max_tokens:
        if max_tokens < 2 or budget_policy == "error":
            raise ValueError(
                f"{label} requires {tokens} audio tokens; budget is {max_tokens}. "
                "Lower audio_max_seconds or choose truncate."
            )
        latent = latent[..., :max_tokens // 2].clone()
    return latent


def _check_total_token_budget(rows: List[Tuple[H3RefMod, float]], max_total_tokens: int = 0) -> int:
    if not isinstance(max_total_tokens, int) or isinstance(max_total_tokens, bool) or max_total_tokens < 0:
        raise ValueError("Combined token budget must be a non-negative integer.")
    total = sum(mod.token_count for mod, strength in rows if float(strength) > 0)
    if max_total_tokens and total > max_total_tokens:
        raise ValueError(
            f"Combined visual+audio references require {total} tokens; budget is {max_total_tokens}. "
            "Lower the visual/audio budgets or set max_total_tokens to 0 to disable the extra cap."
        )
    return total


def _create_mod_from_folder(
    folder: str,
    name: str,
    mode: str,
    concept_type: str,
    audio_concept_type: str,
    vae,
    audio_vae=None,
    ref_resolution: int = 1024,
    pool_h: int = 16,
    pool_w: int = 16,
    latent_frames: int = 16,
    max_tokens: int = 8192,
    identity: int = 500,
    multiplier: int = 1,
    max_frames: int = 240,
    background_retention: float = 0.0,
    audio_max_seconds: float = 30.0,
    audio_max_tokens: int = 5120,
    audio_budget_policy: str = "error",
    max_total_tokens: int = 0,
    description: str = "",
    subfolder: str = "",
    save: bool = True,
    save_layout: str = "bundle",
    budget_policy: str = "truncate",
    merge: bool = False,
    motion_only: bool = False,
    extraction_preset: str = "manual",
) -> List[Tuple[H3RefMod, float]]:
    """Create refs from one folder and optionally save them as split files or one bundle."""
    if budget_policy not in ("truncate", "error"):
        raise ValueError("Unknown visual token budget policy.")
    if save_layout not in ("bundle", "separate_files"):
        raise ValueError("Unknown save layout.")
    if audio_budget_policy not in ("error", "truncate"):
        raise ValueError("Unknown audio token budget policy.")
    if not math.isfinite(audio_max_seconds) or audio_max_seconds <= 0:
        raise ValueError("audio_max_seconds must be a positive finite number.")
    name = _sanitize_name(name)
    mode, ref_resolution, pool_h, pool_w, identity, merge, motion_only, concept_type, audio_concept_type = _apply_extraction_preset(
        mode, ref_resolution, pool_h, pool_w, identity, merge, motion_only,
        concept_type, audio_concept_type, extraction_preset,
        "H3RefModCreateFromFolder"
    )
    description = (description or "").strip() or _auto_folder_description(folder, concept_type)

    images, videos, audios = _scan_folder(folder)
    has_visual = bool(images or videos)
    has_audio_files = bool(audios)
    if not has_visual and not has_audio_files:
        raise ValueError(
            f"H3RefModCreateFromFolder: no images, videos, or audio found in '{folder}'. "
            "Extraction needs at least one usable reference file.")
    print(f"[H3RefModCreateFromFolder] {folder}: {len(images)} image(s), "
          f"{len(videos)} video(s), {len(audios)} audio file(s)")

    ignored = []
    if has_visual and mode == "encode":
        ignored.append("pool_h, pool_w, Refinement Steps, merge, motion_only (Full Reference)")
    if has_visual and max_tokens == 0:
        ignored.append("budget_policy (max_tokens=0)")
    if ignored:
        print("[H3RefModCreateFromFolder] ignored: " + "; ".join(ignored))
    if has_visual and concept_type == "identity" and mode == "training" and max(pool_h, pool_w) < 16:
        print(
            f"[H3RefModCreateFromFolder] warning: concept_type='identity' with "
            f"mode='training' at a {pool_h}x{pool_w} grid — pooling averages away "
            f"exactly the detail that carries a face. For a person, either switch "
            f"mode='encode' or raise pool_h/pool_w toward 32x32+."
        )

    device = comfy.model_management.get_torch_device()

    # ── audio identity (optional) ────────────────────────────────────
    audio_latent = None
    ref_audio_t = 0
    if audios:
        if audio_vae is not None:
            aframes = []
            remaining_seconds = float(audio_max_seconds)
            for ap in audios:
                if remaining_seconds <= 1e-9:
                    print("[H3RefModCreateFromFolder] audio_max_seconds reached; skipping remaining audio files.")
                    break
                waveform, sr = _load_audio_waveform(ap)
                max_samples = max(1, int(round(remaining_seconds * sr)))
                if waveform.shape[-1] > max_samples:
                    waveform = waveform[..., :max_samples].contiguous()
                z = _encode_ref_audio(audio_vae, waveform, sr, device)
                print(f"[H3RefModCreateFromFolder] audio {os.path.basename(ap)}: "
                      f"latent {tuple(z.shape)}")
                aframes.append(z.to(torch.float16))
                remaining_seconds -= waveform.shape[-1] / float(sr)
            audio_latent = torch.cat(aframes, dim=-1) if len(aframes) > 1 else aframes[0]
            audio_latent = _crop_audio_latent(
                audio_latent, audio_max_tokens, audio_budget_policy,
                "H3RefModCreateFromFolder"
            ) if audio_latent is not None else None
            ref_audio_t = 0 if audio_latent is None else audio_latent.shape[-1]
        else:
            print(f"[H3RefModCreateFromFolder] {len(audios)} audio file(s) found but no "
                  "audio_vae connected — audio NOT embedded.")

    include_audio = ref_audio_t > 0
    if not has_visual and not include_audio:
        if has_audio_files and audio_vae is None:
            raise ValueError(
                f"H3RefModCreateFromFolder: '{folder}' contains only audio files, but no audio_vae is connected. "
                "Connect the MiniMax H3 audio VAE or add image/video references."
            )
        raise ValueError(
            f"H3RefModCreateFromFolder: '{folder}' did not produce any usable visual or audio references."
        )

    # ── load visual refs as tensors ──────────────────────────────────
    sources = []  # (tensor [T,H,W,3], is_video, mask [1,H,W] | None)
    for p in images:
        image, mask = _load_image_and_alpha_mask_file(p, max_edge=ref_resolution * 2)
        sources.append((image, False, mask))
    for p in videos:
        sources.append((load_video_file(p, max_frames=max_frames,
                                        max_edge=ref_resolution * 2), True, None))
    if merge and mode != "training":
        print("[H3RefModCreateFromFolder] warning: 'merge' only applies to "
              "training mode — stacking the refs as usual for mode='encode'.")
    if motion_only and mode != "training":
        print("[H3RefModCreateFromFolder] warning: 'motion_only' only applies to "
              "training mode — extracting the full appearance for mode='encode'.")
        motion_only = False

    # shared spatial canvas (encode mode) / pool grid (training mode)
    canvas = None
    if mode == "encode" and len(sources) > 1:
        h, w = sources[0][0].shape[1], sources[0][0].shape[2]
        scale = min(1.0, ref_resolution / min(h, w))
        canvas = (max(32, round(w * scale / 32) * 32),
                  max(32, round(h * scale / 32) * 32))
    pool_grid = None
    if mode == "training" and sources:
        h0, w0 = sources[0][0].shape[1], sources[0][0].shape[2]
        pool_grid = aspect_grid(pool_h, pool_w, h0 / w0)
        if pool_grid != (pool_h, pool_w):
            print(f"[H3RefModCreateFromFolder] pooled grid {pool_h}x{pool_w} -> "
                  f"{pool_grid[0]}x{pool_grid[1]} to match source aspect "
                  f"{w0}x{h0} (avoids squishing the subject wide)")
    gh, gw = pool_grid if pool_grid is not None else (pool_h, pool_w)

    # ── encode each source ───────────────────────────────────────────
    frames = []
    source_shapes = []
    n_img = n_vid = 0
    n_refs = len(sources)
    motion_applied = False
    motion_warned = False
    mask_applied = False
    merge_refs = [] if (merge and mode == "training" and n_refs > 1) else None
    pbar = comfy.utils.ProgressBar(n_refs)
    for idx, (src, is_video, source_mask) in enumerate(sources):
        label = f"ref {idx + 1}/{n_refs} ({'video' if is_video else 'image'})"
        if not is_video:
            src = src[:1]  # pin stills to a single frame
        if mode == "encode":
            if is_video and latent_frames < src.shape[0]:
                sample_idx = torch.linspace(0, src.shape[0] - 1, latent_frames).round().long()
                src = src[sample_idx]
            src = _resize_ref(src, ref_resolution, canvas)
        else:
            src = _resize_ref(src, ref_resolution, None)
        if motion_only and is_video and src.shape[0] > 1:
            diffs = (src[1:] - src[:-1]).abs()
            peak = diffs.max()
            if peak > 1e-6:
                diffs = diffs / peak
            src = diffs
            motion_applied = True
            print(f"[H3RefModCreateFromFolder] {label}: motion_only — encoded temporal differences instead of the frames")
        elif motion_only and not is_video and not motion_warned:
            print("[H3RefModCreateFromFolder] warning: motion_only needs video refs — a still has no motion, keeping its appearance.")
            motion_warned = True
        src = _ensure_min_size(src)
        if is_video and src.shape[0] > 1:
            valid_t = _snap_to_causal_grid(src.shape[0])
            if valid_t != src.shape[0]:
                src = src[:valid_t]
        mask_px = None
        if source_mask is not None:
            mask_px = _resize_mask(source_mask, src.shape[1], src.shape[2])
        with torch.no_grad():
            z = vae.encode(src)
        if z.dim() != 5 or z.shape[1] != 24:
            raise ValueError(f"Expected a MiniMax H3 video VAE latent [1,24,T,H,W], "
                             f"got {tuple(z.shape)}. The connected VAE is not the H3 VAE.")
        source_shapes.append(f"{z.shape[2]}x{z.shape[3]}x{z.shape[4]}")
        if mask_px is not None:
            z = _mask_latent(z, mask_px, background_retention, seed_key=f"{name}:{idx}")
            mask_applied = True
            print(f"[H3RefModCreateFromFolder] {label}: applied embedded alpha mask "
                  f"(background_retention={background_retention})")
        if mode == "encode":
            pooled = z.to(torch.float16)
        else:
            pool_t = min(latent_frames, z.shape[2]) if is_video else 1
            pooled = pool_latent(z, pool_t, gh, gw).to(torch.float16)
            if merge_refs is not None:
                merge_refs.append((pooled.cpu(), z.float().cpu()))
            elif identity > 0:
                pooled = optimize_latent(pooled, z.float(), steps=int(identity),
                                         progress_every=100)
        if merge_refs is None:
            frames.append(pooled)
        n_vid += 1 if is_video else 0
        n_img += 0 if is_video else 1
        print(f"[H3RefModCreateFromFolder] {label}: encoded {tuple(pooled.shape)}")
        pbar.update_absolute(idx + 1)

    merged_n = 0
    latent = None
    if merge_refs is not None:
        common_t = max(p.shape[2] for p, _ in merge_refs)
        init = torch.stack([pool_latent(p, common_t, gh, gw) for p, _ in merge_refs]).mean(0)
        print(f"[H3RefModCreateFromFolder] merging {len(merge_refs)} references into one shared {common_t}x{gh}x{gw} latent"
              + (f", {int(identity)} joint gradient steps" if identity > 0 else " (pure pooling mean — identity=0)"))
        if identity > 0:
            init = optimize_latent_multi(init, [f for _, f in merge_refs], steps=int(identity), progress_every=100)
            print("[H3RefModCreateFromFolder] merge refinement done")
        latent = init.to(torch.float16)
        merged_n = len(merge_refs)
    elif frames:
        latent = torch.cat(frames, dim=2)
    if latent is not None and multiplier > 1:
        latent = latent.repeat(1, 1, multiplier, 1, 1)
    if latent is not None and max_tokens > 0 and budget_policy == "error":
        tokens = _latent_token_count(latent)
        if tokens > max_tokens:
            raise ValueError(
                f"H3RefModCreateFromFolder: extracted {tokens} visual tokens, which exceeds "
                f"max_tokens={max_tokens}. Increase the budget, reduce frames/resolution, "
                "or switch budget_policy to 'truncate'."
            )
    if latent is not None and max_tokens > 0:
        latent = fit_token_budget(latent, max_tokens, name)
    if latent is not None:
        total_t = latent.shape[2]
        kind = "video" if total_t > 1 else "image"
        px_w, px_h = latent.shape[4] * 16, latent.shape[3] * 16
        if merged_n:
            source = f"merge {merged_n} refs"
        elif len(frames) > 1:
            source = "stack"
        else:
            source = "video" if n_vid else "image"

    out_dir = _resolve_output_dir(subfolder)
    requested_base = _base_refmod_name(name)
    bundle_name = resolved_base = requested_base
    bundle_path_no_ext = os.path.join(out_dir, requested_base)
    if save and save_layout == "separate_files":
        resolved_base, visual_name, visual_path_no_ext, audio_name, audio_path_no_ext = _unique_split_mod_paths(
            out_dir, requested_base, include_audio)
        if resolved_base != requested_base:
            print(f"[H3RefModCreateFromFolder] '{requested_base}' already exists in {out_dir} — "
                  f"saving as '{resolved_base}{VISUAL_SUFFIX}'"
                  + (f" and '{resolved_base}{AUDIO_SUFFIX}'" if include_audio else "")
                  + " instead (existing mods are never overwritten).")
    elif save:
        bundle_name, bundle_path_no_ext = _unique_bundle_mod_path(out_dir, requested_base)
        resolved_base = bundle_name
        if bundle_name != requested_base:
            print(f"[H3RefModCreateFromFolder] '{requested_base}' already exists in {out_dir} — "
                  f"saving as '{bundle_name}' instead (existing mods are never overwritten).")
        visual_name, audio_name = _paired_refmod_names(bundle_name)
        visual_path_no_ext = os.path.join(out_dir, visual_name)
        audio_path_no_ext = os.path.join(out_dir, audio_name)
    else:
        resolved_base = requested_base
        visual_name, audio_name = _paired_refmod_names(resolved_base)
        visual_path_no_ext = os.path.join(out_dir, visual_name)
        audio_path_no_ext = os.path.join(out_dir, audio_name)

    rows: List[Tuple[H3RefMod, float]] = []
    visual_mod = None
    if latent is not None:
        visual_mod = H3RefMod(
            name=visual_name,
            kind=kind,
            latent=latent,
            latent_h=latent.shape[3],
            latent_w=latent.shape[4],
            latent_t=total_t,
            mode=mode,
            source=source,
            source_shape=" +".join(source_shapes),
            pool=(f"full-res {px_w}x{px_h}px (short-edge cap {ref_resolution}px)"
                  if mode == "encode" else f"{total_t}x{gh}x{gw}"),
            optimize_steps=int(identity) if mode == "training" else 0,
            tags=[f"{n_img} img, {n_vid} vid"]
                   + ([f"merged {merged_n} refs"] if merged_n else [])
                   + (["motion_only"] if motion_applied else [])
                   + ([f"masked (bg_retention={background_retention})"] if mask_applied else [])
                   + ([f"x{multiplier} repeat"] if multiplier > 1 else [])
                   + ([f"paired {audio_name}"] if include_audio else []),
            description=description,
            concept_type=concept_type,
            audio_concept_type="",
        )
        rows.append((visual_mod, 1.0))
    if include_audio:
        audio_mod = _make_audio_refmod(
            audio_name, audio_latent, audio_concept_type, description,
            sample_rate=getattr(audio_vae, "audio_sample_rate", 32000),
        )
        rows.append((audio_mod, 1.0))
    total_tokens = _check_total_token_budget(rows, max_total_tokens)
    if save:
        if save_layout == "bundle":
            destination = save_refmod_bundle(bundle_path_no_ext, bundle_name, [mod for mod, _ in rows])
            _copy_refmod_thumbnail(images, bundle_path_no_ext)
            for mod, _ in rows:
                _MOD_CACHE[mod.name] = mod
            print(f"[CreateH3RefMod] saved bundle '{bundle_name}' with {len(rows)} member(s) -> {destination}")
        else:
            if visual_mod is not None:
                visual_path = visual_mod.save(visual_path_no_ext)
                _copy_refmod_thumbnail(images, visual_path_no_ext)
                _MOD_CACHE[visual_mod.name] = visual_mod
                print(f"[CreateH3RefMod] saved {_summarize(visual_mod)} -> {visual_path}")
            if include_audio:
                audio_mod = rows[-1][0]
                audio_path = audio_mod.save(audio_path_no_ext)
                _MOD_CACHE[audio_mod.name] = audio_mod
                print(f"[CreateH3RefMod] saved {_summarize(audio_mod)} -> {audio_path}")
        if len(_MOD_CACHE) > _MOD_CACHE_MAX:
            _MOD_CACHE.pop(next(iter(_MOD_CACHE)))
        _nodes_mod._MOD_LIST_CACHE_KEY = None  # refresh the loader dropdown
    else:
        if visual_mod is not None:
            print(f"[CreateH3RefMod] {_summarize(visual_mod)} (not saved)")
        if include_audio:
            print(f"[CreateH3RefMod] {_summarize(rows[-1][0])} (not saved)")
    print(f"[CreateH3RefMod] complete: {len(rows)} reference(s), {total_tokens} tokens")
    return rows


class H3RefModCreateFromFolder(io.ComfyNode):
    """Create a RefMod from every image/video/audio in a folder.

    Scans a dataset folder, encodes the media with the connected H3 VAEs,
    and saves a visual RefMod.
    When audio is present and an audio VAE is connected, it also saves a
    paired audio RefMod. Defaults to an ``identity`` concept in ``encode``
    mode — the right choice for a person/character.

    Flagged as an output node (``is_output_node=True``) so it actually runs
    even when nothing is wired to its ``mods`` output — it still saves to
    disk either way.  ``mods`` carries the freshly created mod at strength
    1.0 (same ``H3_REF_MODS`` shape ``H3RefModExtract`` outputs), so you can
    chain it straight into Apply H3 RefMod / MiniMax H3 RefMods to Video
    without a separate reload step — or just ignore it and reload later with
    Load H3 RefMods.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefModCreateFromFolder",
            display_name="Create H3 RefMod From Folder",
            category="H3RefModPicker",
            description="Scan a dataset folder for reference images/videos/audio and "
                        "create a RefMod (.safetensors). Defaults to an 'identity' "
                        "concept in 'Full Reference' mode — the right choice for a person/"
                        "character. Audio files in the folder are embedded as the mod's "
                        "voice (needs the audio VAE). Saves to models/refmods/ by default, "
                        "and copies a thumbnail from the best source image.",
            inputs=[
                io.String.Input("folder", default="",
                    tooltip="Dataset folder with reference images/videos/audio. REQUIRED — "
                            "an absolute path, or a folder name inside ComfyUI's input/ "
                            "directory. The node will NOT run if left empty."),
                io.Boolean.Input("use_subfolders", default=False,
                    label_on="subfolders", label_off="single folder",
                    tooltip="When ON, the folder input is treated as a parent directory: "
                            "every immediate subfolder is scanned and turned into its own "
                            "RefMod, named after the subfolder. The 'name' input is ignored "
                            "in this mode, and the description is auto-generated from each "
                            "subfolder name. Great for batch-processing a dataset of concepts."),
                io.Boolean.Input("use_folder_as_name", default=True,
                    label_on="folder name", label_off="manual name",
                    tooltip="When ON, single-folder mode names the RefMod from the folder, using "
                        "the same naming scheme as subfolder mode. The 'name' input is ignored. "
                        "Turn this OFF to type a manual name instead."),
                io.String.Input("name", default="name_refMod",
                    tooltip="Saved mod name when 'use folder as name' is OFF. Appears in the "
                        "Load H3 RefMods dropdown after a reload."),
                io.Combo.Input("extraction_preset", options=["manual", "identity_encode", "style_experimental", "motion_sequence"], default="identity_encode",
                    tooltip="manual preserves controls. identity_encode: identity + voice, Full Reference, resolution=1024, steps=0. "
                            "style_experimental: style + voice, Compressed Reference, 8x8 pool, 150 steps. "
                            "motion_sequence: pose_motion + voice, Compressed Reference, 16x16 pool; preserves frame limit and Refinement Steps."),
                io.Boolean.Input("advanced", default=False,
                    label_on="advanced", label_off="simple",
                    tooltip="Preset display mode. simple hides most preset-managed controls; advanced shows the fuller control set while preserving the same preset values."),
                io.Combo.Input("concept_type", options=list(CONCEPT_TYPES), default="identity",
                    tooltip="What this mod represents. 'identity' (default) = a specific "
                            "person/character; also pose_motion, clothing, background, style, "
                            "generic. Audio-style labels are also accepted for compatibility with "
                            "upstream RefMods, but visual concepts should usually stay on identity/"
                            "pose_motion/clothing/background/style."),
                io.Combo.Input("audio_concept_type", options=list(AUDIO_CONCEPT_TYPES), default="voice",
                    tooltip="How to label embedded audio in this RefMod. Matches upstream audio "
                            "concept labels: voice (default), singing, music_style, sound_fx, "
                            "ambience."),
                io.Combo.Input("mode", options=["Full Reference", "Compressed Reference"], default="Full Reference",
                    tooltip="'Full Reference' (default) = full-res VAE encode, max identity (~1K "
                            "tokens/img) — recommended for people. 'Compressed Reference' = pooled grid "
                            "refined by 'Refinement Steps' — cheaper, concept/motion only. Legacy "
                            "'encode'/'training' values remain accepted."),
                io.Vae.Input("vae",
                    tooltip="The MiniMax H3 video VAE."),
                io.Vae.Input("audio_vae", optional=True,
                    tooltip="The MiniMax H3 audio VAE — needed to embed audio files found in "
                            "the folder. If not connected, audio is skipped with a note."),
                io.Int.Input("ref_resolution", default=1024, min=256, max=2048, step=64,
                    tooltip="Target short edge in px (downscale only). 1024 default; 2048 = "
                            "official max fidelity, 4x the tokens."),
                io.Int.Input("pool_h", default=16, min=2, max=64, step=2,
                    tooltip="Compressed Reference mode: pooled latent height. The grid is auto-fit "
                            "to the first source's aspect ratio so a portrait subject isn't squished."),
                io.Int.Input("pool_w", default=16, min=2, max=64, step=2,
                    tooltip="Compressed Reference mode: pooled latent width (long edge if the source is wider than tall)."),
                io.Int.Input("latent_frames", default=16, min=1, max=4800, step=1,
                    tooltip="Per-video temporal limit. Full Reference samples up to this many frames "
                            "before VAE encode; Compressed Reference pools up to this many latent frames. "
                            "Images always use 1."),
                io.Int.Input("max_tokens", default=8192, min=0, max=65536, step=512,
                    tooltip="Hard cap on total injected tokens (0 = no cap). Near-duplicate "
                            "frames dropped first, then resampled to fit."),
                io.Int.Input("identity", display_name="Refinement Steps", default=500, min=0, max=2000, step=50,
                    tooltip="Compressed Reference only: gradient refinement steps (0 = pure pooling)."),
                io.Boolean.Input("merge", default=False,
                    label_on="merge", label_off="stack",
                    tooltip="Compressed Reference only: optimize one shared consensus latent against "
                            "all refs instead of stacking each ref separately."),
                io.Boolean.Input("motion_only", default=False,
                    label_on="motion", label_off="full",
                    tooltip="Compressed Reference only: video refs are converted to temporal differences "
                            "before encoding, so the mod carries where/how things move instead of appearance."),
                io.Int.Input("multiplier", default=1, min=1, max=10, step=1,
                    tooltip="Repeat the extracted latent along time so a short clip is not drowned out by "
                            "the main video's token budget. 1 = no repeat."),
                io.Int.Input("max_frames", default=240, min=2, max=4800, step=1,
                    tooltip="Video decode cap while scanning folder clips before later frame sampling/pooling."),
                io.Float.Input("background_retention", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Only used for still images in the folder that carry an embedded alpha mask. "
                            "Outside the masked subject, 0 collapses the latent toward a blurred copy of "
                            "itself, 1 keeps the full background. Videos and opaque images ignore this."),
                io.Float.Input("audio_max_seconds", default=30.0, min=0.025, max=600.0,
                    tooltip="Combined audio reference length cap across folder audio files before encoding. "
                            "Longer audio creates more tokens; 30 seconds matches the upstream master default."),
                io.Int.Input("audio_max_tokens", default=5120, min=0, max=2147483647, step=512,
                    tooltip="Hard cap on the paired audio RefMod token count (0 = no cap)."),
                io.Combo.Input("audio_budget_policy", options=["error", "truncate"], default="error",
                    tooltip="On audio_max_tokens overflow: error stops without saving, truncate crops "
                            "the combined folder audio latent to fit the token budget."),
                io.Int.Input("max_total_tokens", default=0, min=0, max=1048576, step=512,
                    tooltip="Extra combined cap across visual and audio tokens. 0 disables this shared limit."),
                io.Combo.Input("budget_policy", options=["truncate", "error"], default="truncate",
                    tooltip="On max_tokens overflow: truncate uses the existing frame reduction; "
                            "error stops without saving. 0 max_tokens disables the cap."),
                io.String.Input("description", default="", multiline=True,
                    tooltip="Optional text describing the concept, stored in the mod and "
                        "emitted by the loaders' prompt_hint output. If left empty, "
                        "Create From Folder auto-generates one from the folder name. "
                        "In subfolder mode this field is ignored and each subfolder gets "
                        "its own auto-generated description. A thumbnail is also copied "
                        "from the source images: 3:4 images first, otherwise the largest one."),
                io.String.Input("subfolder", default="", optional=True,
                    tooltip="Optional folder inside the RefMods save directory, for example celebs or voices."),
                io.Boolean.Input("save", default=True, label_on="save", label_off="don't save",
                    tooltip="Save the mod to disk so Load H3 RefMods can pick it up later."),
                io.Combo.Input("save_layout", options=["bundle", "separate_files"], default="bundle",
                    tooltip="bundle (default) saves one single-file RefMod that can contain visual and audio members together. separate_files keeps the older *_Video and *_Audio files."),
            ],
            outputs=[
                io.Custom("H3_REF_MODS").Output("mods",
                    tooltip="Bundle with the freshly created mod(s) at strength 1.0 — feed "
                            "straight into Apply H3 RefMod / MiniMax H3 RefMods to Video, no "
                            "reload needed. In subfolder mode this "
                            "contains one mod per subfolder."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, folder, use_folder_as_name, name, mode, concept_type, audio_concept_type, vae, audio_vae=None,
                ref_resolution=1024, pool_h=16, pool_w=16, latent_frames=16,
                max_tokens=8192, identity=500, merge=False, motion_only=False,
            multiplier=1, max_frames=240, background_retention=0.0, extraction_preset="identity_encode",
            audio_max_seconds=30.0, audio_max_tokens=5120,
            audio_budget_policy="error", max_total_tokens=0,
            description="", subfolder="", budget_policy="truncate",
            save=True, save_layout="bundle", advanced=False, use_subfolders=False, save_dir="") -> io.NodeOutput:
        del advanced
        del save_dir
        raw_folder = (folder or "").strip().strip('"')
        if not (folder or "").strip().strip('"'):
            print("[H3RefModCreateFromFolder] folder input is empty; refusing to scan.")
            raise ValueError(
                "Create H3 RefMod: 'folder' is empty. Point it at a dataset folder "
                "(absolute path, or a name inside input/) — the node refuses to run "
                "without one so it can't accidentally scan your whole input/ directory.")
        folder = _resolve_folder(folder)
        mode = normalize_mode(mode)
        print(
            f"[H3RefModCreateFromFolder] start: folder='{raw_folder}' -> '{folder}', "
            f"use_subfolders={bool(use_subfolders)}, preset='{extraction_preset}', mode='{mode}'"
        )

        if use_subfolders:
            subfolders = [
                os.path.join(folder, d)
                for d in sorted(os.listdir(folder))
                if os.path.isdir(os.path.join(folder, d))
            ]
            if not subfolders:
                raise ValueError(
                    f"H3RefModCreateFromFolder: use_subfolders=True but no subfolders "
                    f"found in '{folder}'. In subfolder mode, files placed directly in "
                    "that folder are ignored.")
            print(f"[H3RefModCreateFromFolder] subfolder mode: {len(subfolders)} folder(s)")
            mods = []
            for sub in subfolders:
                sub_name = _folder_refmod_name(sub)
                created = _create_mod_from_folder(
                    sub, sub_name, mode, concept_type, audio_concept_type, vae, audio_vae,
                    ref_resolution, pool_h, pool_w, latent_frames, max_tokens,
                    identity, multiplier, max_frames, background_retention,
                    audio_max_seconds, audio_max_tokens,
                    audio_budget_policy, max_total_tokens,
                    _auto_folder_description(sub, concept_type), subfolder,
                    save, save_layout, budget_policy, merge, motion_only, extraction_preset,
                )
                mods.extend(created)
            return io.NodeOutput(mods)

        name = _folder_refmod_name(folder) if use_folder_as_name else _sanitize_name(name)
        rows = _create_mod_from_folder(
            folder, name, mode, concept_type, audio_concept_type, vae, audio_vae,
            ref_resolution, pool_h, pool_w, latent_frames, max_tokens,
            identity, multiplier, max_frames, background_retention,
            audio_max_seconds, audio_max_tokens,
            audio_budget_policy, max_total_tokens, description, subfolder,
            save, save_layout, budget_policy, merge, motion_only, extraction_preset,
        )
        return io.NodeOutput(rows)


NODE_CLASS_MAPPINGS = {
    "H3RefModCreateFromFolder": H3RefModCreateFromFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModCreateFromFolder": "Create H3 RefMod From Folder",
}
