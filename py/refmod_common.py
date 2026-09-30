"""Shared helpers for the RefMod pack: media loading and the refmods folder."""

from __future__ import annotations

import math
import os
import random
from typing import List, Optional, Tuple

import torch

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


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


def list_media_files(folder: str) -> Tuple[List[str], List[str]]:
    """(images, videos) directly under ``folder`` (top level only), sorted by name."""
    images, videos = [], []
    if os.path.isdir(folder):
        for fn in sorted(os.listdir(folder)):
            ext = os.path.splitext(fn)[1].lower()
            p = os.path.join(folder, fn)
            if os.path.isfile(p):
                if ext in IMAGE_EXTS:
                    images.append(p)
                elif ext in VIDEO_EXTS:
                    videos.append(p)
    return images, videos


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


def _sample_video_frames(frames, total, max_frames, prepare):
    """Keep at most max_frames; convert only retained frames, before buffering."""
    import numpy as np
    known = math.isfinite(total) and total > 0
    total = int(total) if known else 0
    targets = None
    if total > max_frames:
        targets = {round(i * (total - 1) / max(1, max_frames - 1)) for i in range(max_frames)}
    kept = []
    rng = random.Random(0)
    for index, frame in enumerate(frames):
        if targets is not None and index not in targets:
            continue
        slot = len(kept) if len(kept) < max_frames else rng.randrange(index + 1)
        if slot >= max_frames:
            continue
        item = (index, np.array(prepare(frame), copy=True))
        if slot == len(kept):
            kept.append(item)
        else:
            kept[slot] = item
        if targets is not None and len(kept) == len(targets):
            break
    if not kept:
        raise ValueError("Video contains no decodable frames.")
    kept.sort(key=lambda item: item[0])
    shape = kept[0][1].shape
    result = torch.empty((len(kept), *shape), dtype=torch.float32, device="cpu")
    for index, (_, frame) in enumerate(kept):
        if frame.shape != shape:
            raise ValueError("Video frame dimensions change during the clip.")
        result[index].copy_(torch.from_numpy(frame))
    return result.div_(255.0)


def _prepare_av_frame(frame, max_edge):
    import numpy as np
    from PIL import Image
    rotation = frame.rotation
    scale = min(1.0, max_edge / max(frame.width, frame.height)) if max_edge else 1.0
    rgb = frame.reformat(width=max(1, round(frame.width * scale)),
                         height=max(1, round(frame.height * scale)), format="rgb24").to_ndarray()
    if rotation and rotation % 90 == 0:
        return np.rot90(rgb, int(rotation // 90))
    if rotation:
        image = Image.fromarray(rgb).rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
        if max_edge:
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        return np.asarray(image)
    return rgb


def load_video_file(path: str, max_frames: int = 240,
                    max_edge: Optional[int] = None) -> torch.Tensor:
    """Load bounded RGB frames using ComfyUI's PyAV, then ImageIO/FFmpeg.

    Known-length clips are uniformly sampled; unknown lengths use a bounded,
    deterministic reservoir. Every retained frame is resized before buffering.
    """
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1.")
    if max_edge is not None and max_edge < 1:
        raise ValueError("max_edge must be positive when provided.")
    errors = []
    try:
        import av
    except ImportError as exc:
        errors.append(f"PyAV: {exc}")
    else:
        try:
            with av.open(path) as container:
                if not container.streams.video:
                    raise ValueError("File contains no video stream.")
                stream = container.streams.video[0]
                return _sample_video_frames(
                    container.decode(stream), stream.frames or 0, max_frames,
                    lambda frame: _prepare_av_frame(frame, max_edge))
        except (av.error.FFmpegError, OSError, ValueError) as exc:
            errors.append(f"PyAV: {exc}")
    try:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image
        reader = imageio.get_reader(path, format="FFMPEG")
        try:
            def prepare(frame):
                image = Image.fromarray(np.asarray(frame)).convert("RGB")
                if max_edge:
                    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                return np.asarray(image)
            total = reader.get_meta_data().get("nframes", 0) or 0
            return _sample_video_frames(iter(reader), total, max_frames, prepare)
        finally:
            reader.close()
    except (ImportError, OSError, ValueError) as exc:
        errors.append(f"ImageIO/FFmpeg: {exc}")
    raise RuntimeError(f"Cannot decode video '{path}': " + "; ".join(errors))


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
