from __future__ import annotations

import os
from typing import Dict, List, Optional

from .refmod_common import refmods_dir, refmods_dirs
from .refmod_core import read_refmod_meta, refmod_capabilities


PREVIEW_EXTS = (".png", ".jpg", ".jpeg", ".webp")
HIDDEN_BROWSER_DIRS = {"graph_presets"}
VISUAL_SUFFIXES = ("_visual", "_video")
AUDIO_SUFFIXES = ("_audio",)
VISUAL_FILE_SUFFIXES = ("_visual", "_Visual", "_video", "_Video")
AUDIO_FILE_SUFFIXES = ("_audio", "_Audio")


def browser_root() -> str:
    root = os.path.abspath(refmods_dir())
    os.makedirs(root, exist_ok=True)
    return root


def browser_roots() -> List[str]:
    """Every declared RefMod folder; the first one is browser_root()."""
    return refmods_dirs()


def _browser_root_labels() -> Dict[str, str]:
    labels: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for index, root in enumerate(browser_roots(), start=1):
        base = os.path.basename(os.path.normpath(root)) or f"root{index}"
        count = counts.get(base, 0) + 1
        counts[base] = count
        labels[os.path.realpath(root)] = base if count == 1 else f"{base}#{count}"
    return labels


def _is_root(path: str) -> bool:
    real = os.path.realpath(_safe_abspath(path))
    return any(real == os.path.realpath(root) for root in browser_roots())


def _root_label_for_path(path: str) -> str:
    current = os.path.realpath(_safe_abspath(path))
    for root_real, label in _browser_root_labels().items():
        if current == root_real or current.startswith(root_real + os.sep):
            return label
    return "refmods"


def _safe_abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _resolve_browser_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return browser_root()
    expanded = os.path.expanduser(raw)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    normalized = raw.replace("\\", os.sep).replace("/", os.sep)
    for root in browser_roots():
        candidate = os.path.abspath(os.path.join(root, normalized))
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(os.path.join(browser_root(), normalized))


def _is_under_root(path: str) -> bool:
    current = os.path.realpath(_safe_abspath(path))
    for root_dir in browser_roots():
        root = os.path.realpath(root_dir)
        if current == root or current.startswith(root + os.sep):
            return True
    return False


def safe_dir_path(path: str = "") -> str:
    if not path:
        return browser_root()
    current = _resolve_browser_path(path)
    if not _is_under_root(current):
        raise ValueError("Path is outside models/refmods")
    if not os.path.isdir(current):
        raise ValueError("Folder not found")
    return current


def safe_file_path(path: str) -> str:
    if not path:
        raise ValueError("Missing path")
    current = _resolve_browser_path(path)
    if not _is_under_root(current):
        raise ValueError("Path is outside models/refmods")
    if not os.path.isfile(current):
        raise ValueError("File not found")
    return current


def _strip_known_suffix(path_no_ext: str) -> tuple[str, str | None, str | None]:
    name = os.path.basename(path_no_ext)
    lower_name = name.lower()
    for suffix in VISUAL_SUFFIXES:
        if lower_name.endswith(suffix):
            return (path_no_ext[:-len(suffix)], "visual", suffix)
    for suffix in AUDIO_SUFFIXES:
        if lower_name.endswith(suffix):
            return (path_no_ext[:-len(suffix)], "audio", suffix)
    return (path_no_ext, None, None)


def paired_mod_path(path_no_ext: str, target_kind: str) -> Optional[str]:
    base, current_kind, _suffix = _strip_known_suffix(path_no_ext)
    if current_kind is None:
        return None
    current_path = path_no_ext + ".safetensors"
    suffixes = VISUAL_FILE_SUFFIXES if target_kind == "visual" else AUDIO_FILE_SUFFIXES
    for suffix in suffixes:
        candidate = base + suffix + ".safetensors"
        if candidate != current_path and os.path.isfile(candidate):
            return candidate
    return None


def _find_preview(path_no_ext: str) -> Optional[str]:
    candidates = [path_no_ext]
    base, kind, _suffix = _strip_known_suffix(path_no_ext)
    if kind is not None:
        candidates.insert(0, base)
    for stem in candidates:
        for ext in PREVIEW_EXTS:
            candidate = stem + ext
            if os.path.isfile(candidate):
                return candidate
    return None


def _display_mod_name(path_no_ext: str) -> str:
    base, kind, _suffix = _strip_known_suffix(path_no_ext)
    if kind is not None:
        name = os.path.basename(base)
        return name or os.path.basename(path_no_ext)
    return os.path.basename(path_no_ext)


def _mod_entry(path: str) -> Optional[Dict]:
    if not path.lower().endswith(".safetensors"):
        return None
    path_no_ext = path[:-len(".safetensors")]
    meta = read_refmod_meta(path_no_ext)
    if meta is None:
        return None
    caps = refmod_capabilities(meta)
    meta_kind = str(caps.get("kind", "") or "")
    if meta_kind not in ("image", "video", "audio", "bundle"):
        return None
    _base, paired_kind, _suffix = _strip_known_suffix(path_no_ext)
    paired_visual = paired_mod_path(path_no_ext, "visual")
    paired_audio = paired_mod_path(path_no_ext, "audio")
    paired_split = bool(paired_visual or paired_audio)
    if paired_kind == "audio" and paired_visual:
        return None
    has_visual = bool(caps.get("has_visual", False))
    has_audio = bool(caps.get("has_audio", False))
    if paired_split:
        has_visual = True
        has_audio = True
    preview = _find_preview(path_no_ext) if has_visual else None
    return {
        "name": _display_mod_name(path_no_ext) + ".safetensors",
        "path": path,
        "preview_path": preview,
        "kind": meta_kind or "unknown",
        "has_visual": has_visual,
        "has_audio": has_audio,
        "paired_split": paired_split,
        "concept_type": str(meta.get("concept_type", "generic") or "generic"),
        "description": str(meta.get("description", "") or ""),
    }


def _scan_dir(current: str) -> tuple[List[Dict], List[Dict]]:
    dirs: List[Dict] = []
    mods: List[Dict] = []

    try:
        entries = list(os.scandir(current))
    except PermissionError as exc:
        raise ValueError("Access denied") from exc

    for entry in entries:
        name = entry.name
        if name.startswith(".") or name in HIDDEN_BROWSER_DIRS:
            continue
        try:
            if entry.is_dir(follow_symlinks=False):
                dirs.append({"name": name, "path": os.path.abspath(entry.path)})
            elif entry.is_file(follow_symlinks=False) and name.lower().endswith(".safetensors"):
                item = _mod_entry(os.path.abspath(entry.path))
                if item is not None:
                    mods.append(item)
        except OSError:
            continue
    return dirs, mods


def _annotate_merged_duplicates(items: List[Dict]) -> None:
    counts: Dict[str, int] = {}
    for item in items:
        key = str(item.get("name", "")).lower()
        counts[key] = counts.get(key, 0) + 1
    for item in items:
        key = str(item.get("name", "")).lower()
        if counts.get(key, 0) > 1:
            item["name"] = f"{item['name']} [{_root_label_for_path(str(item.get('path', '')))}]"


def list_browser_dir(path: str = "") -> Dict:
    current = safe_dir_path(path)

    if _is_root(current):
        dirs: List[Dict] = []
        mods: List[Dict] = []
        for root in browser_roots():
            root_dirs, root_mods = _scan_dir(root)
            dirs.extend(root_dirs)
            mods.extend(root_mods)
        _annotate_merged_duplicates(dirs)
        _annotate_merged_duplicates(mods)
        current = browser_root()
        parent_path = None
    else:
        dirs, mods = _scan_dir(current)
        parent = os.path.dirname(current.rstrip("\\/"))
        if parent and _is_root(parent):
            parent_path = browser_root()
        elif parent and _is_under_root(parent) and parent != current:
            parent_path = parent
        else:
            parent_path = None

    dirs.sort(key=lambda item: item["name"].lower())
    mods.sort(key=lambda item: item["name"].lower())
    return {
        "root": browser_root(),
        "current_path": current,
        "parent_path": parent_path,
        "dirs": dirs,
        "mods": mods,
    }