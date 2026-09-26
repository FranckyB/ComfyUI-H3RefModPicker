"""
decode_refmod.py — H3RefModDecode node.

Decode a saved RefMod's latent back into images, so you can see what the mod
actually captured.  Purely a debugging / inspection tool — it plays no part in
generation.

A RefMod stores a video-VAE latent [1, 24, T, h, w] (the reference).  This node
runs it back through the H3 video VAE decoder and returns the frames as an IMAGE
batch [T, H, W, 3], so you can preview a mod, sanity-check an extraction, or
eyeball what a pooled (training-mode) mod actually kept vs an encode-mode one.

It decodes the full latent timeline exactly as stored in the RefMod; it does
not resample or guess a frame subset.

Note: a training-mode (pooled) mod decodes to a tiny, blurry grid — that's the
point of the mode, it only stores a concept thumbnail.  An encode-mode mod
decodes to recognizable full-res frames.  Audio (if the mod has it) is not
decoded here — this is the visual reference only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.utils
from comfy_api.latest import io

from .refmod_loader import _list_mod_names, _load_mod


def _reshape_decoded_frames(frames: torch.Tensor) -> torch.Tensor:
    if frames.dim() == 5:  # [B, T, H, W, C] -> [B*T, H, W, C]
        frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
    return frames


def _upscale_for_inspection(frames: torch.Tensor, min_edge: int, mode: str) -> torch.Tensor:
    if min_edge <= 0 or frames.dim() != 4 or frames.shape[0] <= 0:
        return frames
    height, width = int(frames.shape[1]), int(frames.shape[2])
    shortest = min(height, width)
    if shortest <= 0 or shortest >= min_edge:
        return frames
    scale = float(min_edge) / float(shortest)
    target_h = max(1, int(round(height * scale)))
    target_w = max(1, int(round(width * scale)))
    nchw = frames.movedim(-1, 1)
    if mode == "training":
        upscaled = F.interpolate(nchw, size=(target_h, target_w), mode="nearest")
    else:
        upscaled = comfy.utils.common_upscale(nchw, target_w, target_h, "lanczos", "disabled")
    return upscaled.movedim(1, -1)


class H3RefModDecode(io.ComfyNode):
    """Decode a RefMod's latent back to images for inspection."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefModDecode",
            display_name="Decode H3 RefMod",
            category="H3RefMod",
            description="Decode a saved RefMod's latent back into images so you can see "
                        "what it captured. Inspection only — not part of generation. "
                        "The full stored latent timeline is decoded as-is; no frame "
                        "sampling is applied. "
                        "An encode-mode mod decodes to recognizable frames; a "
                        "training-mode (pooled) mod decodes to a tiny blurry grid "
                        "(that's all it stores). Audio is not decoded.",
            inputs=[
                io.Combo.Input("mod", options=_list_mod_names(),
                    tooltip="The RefMod to decode (from models/refmods/)."),
                io.Vae.Input("vae",
                    tooltip="The MiniMax H3 video VAE."),
                io.Int.Input("inspect_min_edge", default=320, min=0, max=2048, step=32,
                    tooltip="Upscale small decoded previews so pooled/training mods are easier "
                            "to inspect. 0 disables inspection upscaling. Training-mode uses "
                            "nearest-neighbor to keep the stored cell structure visible."),
            ],
            outputs=[
                io.Image.Output("images", display_name="images",
                    tooltip="The decoded reference frames [T, H, W, 3]."),
            ],
        )

    @classmethod
    def execute(cls, mod, vae, inspect_min_edge=320) -> io.NodeOutput:
        m = _load_mod(mod)
        if m.kind == "audio":
            raise ValueError("H3RefModDecode only decodes visual RefMods; audio-only mods are not supported.")
        z = m.latent  # [1, 24, T, h, w]
        total_t = z.shape[2]

        with torch.no_grad():
            frames = vae.decode(z)
        # video latent decodes channel-last; frame count rides the batch dim, so a
        # 5-dim result is reshaped like the core VAEDecode node does
        frames = _reshape_decoded_frames(frames)
        frames = frames.clamp(0.0, 1.0).float().cpu()
        frames = _upscale_for_inspection(frames, int(inspect_min_edge), m.mode)

        print(f"[H3RefModDecode] '{m.name}' ({m.mode}, {m.kind}, "
              f"latent_t={total_t}, inspect_min_edge={inspect_min_edge}) -> {tuple(frames.shape)}")
        return io.NodeOutput(frames)


NODE_CLASS_MAPPINGS = {
    "H3RefModDecode": H3RefModDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModDecode": "Decode H3 RefMod",
}
