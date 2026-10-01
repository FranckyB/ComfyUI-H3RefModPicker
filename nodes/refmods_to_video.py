"""Native H3 reference conditioning with grounded RefMod stacks."""

from __future__ import annotations

import io as bytes_io
import math

import torch
import torch.nn.functional as functional
from comfy_api.latest import io

REFERENCE_TYPE = "H3_REFMOD_REFERENCES"


def _mod_rows(mods):
    if mods is None:
        return []
    if hasattr(mods, "ref_block"):
        return [(mods, 1.0)]
    if isinstance(mods, (tuple, list)) and mods and hasattr(mods[0], "ref_block"):
        mods = [mods]
    rows = []
    for row in mods:
        if not isinstance(row, (tuple, list)) or len(row) not in (2, 3):
            raise ValueError("mods must contain (RefMod, strength) rows.")
        mod, strength = row[:2]
        if mod is None:
            continue
        if not callable(getattr(mod, "ref_block", None)):
            raise ValueError("mods contains an unsupported RefMod object.")
        strength = float(strength)
        if not math.isfinite(strength):
            raise ValueError("RefMod strengths must be finite.")
        if strength > 0:
            rows.append((mod, min(1.0, strength)))
    return rows


def _decoded_frames(video_vae, latent):
    with torch.no_grad():
        frames = video_vae.decode(latent)
    if frames.ndim == 5 and frames.shape[0] == 1:
        frames = frames[0]
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] < 1:
        raise ValueError(f"H3 video VAE returned invalid images: {tuple(frames.shape)}")
    return frames.detach().float().clamp(0, 1).cpu()


def _encoder_frames(mod, video_vae, reference_fps):
    times = getattr(mod, "enc_times", None)
    path = getattr(mod, "path", "")
    packed_frames = getattr(mod, "enc_frames", None)
    is_stack = getattr(mod, "source", "") == "stack"
    if times and (packed_frames or path) and (is_stack or abs(getattr(mod, "enc_fps", 0) - reference_fps) < 1e-6):
        import numpy as np
        from PIL import Image
        def unpack(get_tensor):
            frames = []
            for index in range(len(times)):
                packed = get_tensor(f"enc_{index}").numpy().tobytes()
                with Image.open(bytes_io.BytesIO(packed)) as image:
                    frames.append(torch.from_numpy(np.array(image.convert("RGB"))).float() / 255)
            return torch.stack(frames), list(times), "stored pictures"

        if packed_frames:
            return unpack(packed_frames.__getitem__)
        from safetensors import safe_open
        filename = path if path.endswith(".safetensors") else path + ".safetensors"
        with safe_open(filename, framework="pt", device="cpu") as handle:
            return unpack(handle.get_tensor)
    if video_vae is None:
        raise ValueError(f"Connect video_vae to reconstruct visual RefMod '{mod.name}'.")
    if is_stack:
        frames = [_decoded_frames(video_vae, mod.latent[:, :, index:index + 1])[:1]
                  for index in range(mod.latent.shape[2])]
        return torch.cat(frames), [float(index) for index in range(len(frames))], "decoded stack"
    frames = _decoded_frames(video_vae, mod.latent)
    if mod.kind == "image":
        return frames[:1], [0.0], "decoded image"
    times = [index / 2 for index in range(math.ceil(frames.shape[0] * 2 / reference_fps))]
    indices = [min(round(time * reference_fps), frames.shape[0] - 1) for time in times]
    return frames[indices], times, "decoded video"


def _visual_item(mod, frames, times, strength, stack_pictures, visual_kind=None):
    if strength < 1:
        samples = frames.movedim(-1, 1)
        blurred = functional.adaptive_avg_pool2d(samples, (max(1, mod.latent_h // 8),
                                                         max(1, mod.latent_w // 8)))
        blurred = functional.interpolate(blurred, size=samples.shape[-2:], mode="bilinear", align_corners=False)
        frames = (strength * samples + (1 - strength) * blurred).movedim(1, -1)
    if (visual_kind or mod.kind) == "image":
        return {"type": "image", "data": frames[:1]}, 1
    if getattr(mod, "source", "") == "stack":
        count = frames.shape[0]
        if stack_pictures == "every 4th":
            indices = list(range(0, count, 4))
            frames = frames[indices]
            times = [index / 2 for index in range(len(indices))]
        else:
            indices = ([round(index * (count - 1) / 7) for index in range(8)]
                       if stack_pictures == "up to 8" and count > 8 else list(range(count)))
            frames = frames[indices].repeat_interleave(2, dim=0)
            times = [index + offset for index in range(len(indices)) for offset in (0.0, 0.5)]
        shown = len(indices)
    else:
        shown = frames.shape[0]
    return {"type": "video", "data": frames, "timestamps": times}, shown


class H3RefModReferences(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefModReferences",
            display_name="Collect H3 References",
            category="H3RefModPicker",
            inputs=[
                io.Custom(REFERENCE_TYPE).Input("references", optional=True,
                    tooltip="Optional references from another collector."),
                io.Autogrow.Input("ref_images", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("ref_video", tooltip="Video frame batch at 24 fps."),
                    prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("ref_video_audio", tooltip="Soundtrack for the same-numbered video input."),
                    prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[io.Custom(REFERENCE_TYPE).Output("references")],
        )

    @classmethod
    def execute(cls, references=None, ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None):
        entries = list(references or [])
        for name, image in (ref_images or {}).items():
            if image is not None:
                entries.append({"kind": "image", "data": image, "name": name})
        videos = ref_videos or {}
        soundtracks = ref_video_audios or {}
        for name, soundtrack in soundtracks.items():
            if soundtrack is not None and videos.get("ref_video_" + name.rsplit("_", 1)[-1]) is None:
                raise ValueError(f"{name} needs its same-numbered ref_video input; use ref_audio for standalone audio.")
        for name, frames in videos.items():
            if frames is not None:
                entries.append({"kind": "video", "data": frames, "name": name,
                                "audio": soundtracks.get("ref_video_audio_" + name.rsplit("_", 1)[-1])})
        for name, audio in (ref_audios or {}).items():
            if audio is not None:
                entries.append({"kind": "audio", "data": audio, "name": name})
        return io.NodeOutput(entries)


class _ReferencePresentation:
    def __init__(self):
        self.items = []

    def tokenize(self, prompt, minimax_ref_items):
        self.items = minimax_ref_items
        return None

    def encode_from_tokens_scheduled(self, tokens):
        return [[None, {}]]


def _native_references(references, video_vae, audio_vae, width, height, length, ref_image_size):
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

    images, videos, soundtracks, audios = {}, {}, {}, {}
    names = []
    for kind in ("image", "video", "audio"):
        for index, entry in enumerate(references or []):
            if entry.get("kind") != kind:
                continue
            name = entry["name"]
            if kind == "image":
                images[f"ref_image_{index}"] = entry["data"]
            elif kind == "video":
                videos[f"ref_video_{index}"] = entry["data"]
                if entry.get("audio") is not None:
                    soundtracks[f"ref_video_audio_{index}"] = entry["audio"]
                    names.append(name + " soundtrack")
            else:
                audios[f"ref_audio_{index}"] = entry["data"]
            names.append(name)
    presentation = _ReferencePresentation()
    output = MiniMaxH3ReferenceToVideo.execute(
        presentation, "", width, height, length, ref_image_size=ref_image_size,
        vae=video_vae, audio_vae=audio_vae, ref_images=images, ref_videos=videos,
        ref_video_audios=soundtracks, ref_audios=audios)
    conditioning, latent = output.result
    return presentation.items, list(conditioning[0][1].get("minimax_refs", [])), names, latent


class H3RefModsToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefModsToVideo",
            display_name="MiniMax H3 RefMods to Video",
            category="model/conditioning/minimax",
            description="Reference-to-Video with grounded RefMods and optional collected media references.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae", optional=True,
                    tooltip="Encodes raw pictures/videos and reconstructs RefMods without stored encoder pictures."),
                io.Vae.Input("audio_vae", optional=True,
                    tooltip="Encodes raw reference audio. Audio RefMods are already encoded."),
                io.Custom(REFERENCE_TYPE).Input("references", optional=True),
                io.MultiType.Input("mods", types=[io.Custom("H3_REFMOD"), io.Custom("H3_REF_MODS")], optional=True),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=8192, step=32),
                io.Int.Input("height", default=768, min=32, max=8192, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Combo.Input("stack_pictures", options=["every 4th", "up to 8", "all"], default="every 4th",
                    tooltip="Pictures shown to the text encoder for a stacked RefMod. All latent frames remain applied."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output(),
                     io.String.Output("reference_map")],
        )

    @classmethod
    def execute(cls, clip, prompt, width, height, length, video_vae=None, audio_vae=None,
                references=None, mods=None, ref_image_size="match", stack_pictures="every 4th"):
        if stack_pictures not in ("every 4th", "up to 8", "all"):
            raise ValueError("Unknown stack_pictures mode.")
        items, blocks, names, latent = _native_references(
            references, video_vae, audio_vae, width, height, length, ref_image_size)
        counters = {"image": 0, "video": 0, "audio": 0}
        labels = {"image": "Picture", "video": "Video", "audio": "Audio"}
        mapping = []

        def append_label(item, name):
            kind = item["type"]
            counters[kind] += 1
            mapping.append(f"<{labels[kind]} {counters[kind]}> = {name}")

        for item, name in zip(items, names):
            append_label(item, name)
        decoded = {}
        for mod, strength in _mod_rows(mods):
            block = mod.ref_block(strength)
            if block is None:
                continue
            block = dict(block)
            kind = block["kind"]
            name = mod.name
            if kind not in ("image", "video", "video_audio", "audio"):
                raise ValueError(f"Unsupported RefMod kind: {kind}")
            if kind == "audio" or (block.get("audio_latent") is not None and block.get("ref_audio_t", 0) > 0):
                item = {"type": "audio"}
                items.append(item)
                append_label(item, name + (" soundtrack" if kind != "audio" else ""))
            if kind != "audio":
                if id(mod) not in decoded:
                    decoded[id(mod)] = _encoder_frames(mod, video_vae, 24.0)
                frames, times, _source = decoded[id(mod)]
                visual_kind = "video" if kind == "video_audio" else kind
                item, _shown = _visual_item(mod, frames, times, strength, stack_pictures, visual_kind)
                items.append(item)
                append_label(item, name)
            block["refmod"] = True
            blocks.append(block)
        if video_vae is not None or audio_vae is not None:
            import comfy.model_management
            comfy.model_management.soft_empty_cache()
        tokens = clip.tokenize(prompt, minimax_ref_items=items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        output = []
        for embedding, metadata in conditioning:
            if items and "minimax_token_tags" not in metadata:
                raise ValueError("Connect a MiniMax H3 CLIP that supports reference token tags.")
            metadata = dict(metadata)
            if blocks:
                metadata["minimax_refs"] = list(metadata.get("minimax_refs", [])) + blocks
            output.append([embedding, metadata])
        reference_map = "\n".join(mapping) or "No references."
        return io.NodeOutput(output, latent, reference_map)


NODE_CLASS_MAPPINGS = {
    "H3RefModsToVideo": H3RefModsToVideo,
    "H3RefModReferences": H3RefModReferences,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RefModsToVideo": "MiniMax H3 RefMods to Video",
    "H3RefModReferences": "Collect H3 References",
}

