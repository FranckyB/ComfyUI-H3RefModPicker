# ComfyUI-H3RefModPicker

Companion add-on for [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)

This add-on is meant to work alongside **MiniMaxH3Mod**. It provides a small, focused subset of RefMod features: a Visual Picker, a simple loader, an apply axis, a batch Create From Folder node, and grounded RefMods-to-Video generation with an optional reference collector. It keeps compatibility with both the original RefMod format and the newer bundle format.

If you want the fuller MiniMax H3 RefMod toolset and more creation/apply options, the recommended install is [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod).

Please note, the `Create from Input` node was removed from this add-on. The same functionality is available in the official MiniMaxH3Mod add-on, using the `Create H3 RefMod Master` node.

## Latest Updates

Video-file support has been removed from `Create H3 RefMod From Folder`. Video clips are not ideal for the simple, lightweight RefMods this add-on aims to create: around **six to ten well-chosen reference images** are usually sufficient. Keep their aspect ratios consistent within each dataset folder. This is a recommendation, not an image limit; optional audio remains supported, and video files in the folder are ignored.

## Preview image naming

To have the Visual Picker pick up a preview image, place the image beside the RefMod using the same base name as the RefMod file.

- For the newer bundle format, the image and the RefMod just need to share the exact same name. Example: `character_refMod.safetensors` with `character_refMod.png`.
- For older split RefMods, do not name the preview image `_audio` or `_visual`. Use only the shared base name. Example: `character_refMod_visual.safetensors` and `character_refMod_audio.safetensors` should use `character_refMod.png`.
- When a matching `_visual` and `_audio` pair exists, plus that shared preview image, the picker sees them as one logical item.


<p align="center">
  <img src="docs/examples/browser_example.png" alt="Extract H3 RefMod in use" />
  <br />
  <em>Example using refMods from <a href="https://huggingface.co/spaces/malcolmrey/browser">Malcolm Reynolds</a>.</em>
</p>

<p align="center">
  <img src="docs/examples/workflow_example.png" alt="Workflow example" />
  <br />
  <em>Use Left and Right arrows when over Picker to switch quickly.</em>
</p>

<p align="center">
  <img src="docs/examples/create_from_folder_example.png" alt="Create from folder example" />
  <br />
  <em>Create refMods from Folder example.</em>
</p>

## Main additions

- `Visual RefMod Picker` lets you browse using a **RefMods** browser. It supports legacy split RefMods and the newer single-file bundle format.
- For legacy split RefMods, found pairs are grouped as one item in the picker. They are shown as one entry and loaded together, with separate video/audio weight controls. This uses a weight behavior: `0..1` is regular strength, values above 1 expand into repeated copies. For example, a weight of 2.7 would be the same as strength: 1.0, copies: 2.7.
- `Create H3 RefMod From Folder` scans a folder of reference images and optional audio, and saves in the bundle format by default. Around six to ten images can be sufficient. It can also batch-create RefMods for all subfolders found. The optional `include_images` switch adds reconstructed encoder pictures to visual RefMods while keeping the existing latent format compatible; it is off by default.
- `Load RefMod Simple` is a singular loader. It supports standalone visual/audio RefMods and the new bundle format, while still using the same weight behavior as the Picker.
- `MiniMax H3 RefMods to Video` grounds prompt labels with RefMod pictures and applies their visual/audio latents, returning positive conditioning, an empty AV latent, and a reference map. Optional ordinary media can be added through `Collect H3 References`. It replaced minimax's Reference to Video node. It takes in mods as an input, bypassing the need to apply the RefMods using an apply node.
- `Apply H3 RefMod Simple` is a streamlined version of the Apply H3 RefMod, without the extra controls.

## RefMods to Video Node

The grounded RefMods-to-Video node builds on work by [Adudeguyman's ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder). This also revisits earlier visual-grounding experiments this add-on had initially, which were perhaps set aside a little too quickly :)

Unlike plain prompt encoding followed by RefMod Apply, this node gives the text encoder visual content behind labels such as `<Picture 1>` and `<Video 1>`, while supplying the corresponding latents to diffusion. This lets you refer to specific references in your prompt, but does not guarantee exclusive character or voice assignment.

Connect an H3 CLIP and the `mods` output from the Picker, Prompt Manager's `Prompt Composer`, a RefMod loader, or MiniMaxH3Mod's standard stack to `MiniMax H3 RefMods to Video`. Prompt Composer preserves embedded encoder images and returns weights above 1 as repeated stack rows, matching the Picker. For RefMod-only generation, leave `references` disconnected. To add ordinary media, connect `Collect H3 References`; connect video frame batches to `ref_video_N` and their soundtracks to the same-numbered `ref_video_audio_N`. The reference inputs grow as connections are added.

`video_vae` reconstructs visual RefMods without stored encoder pictures, and encodes raw images/videos. Encoder pictures stored by this creator or Fantastic can be read without decoding. `audio_vae` is needed to encode raw audio, not audio RefMods. As in the native reference node, raw media without its VAE only enters the encoder presentation. Latent-only visual RefMods require `video_vae`. Video playback is assumed to be 24 fps.

The `reference_map` string output contains only prompt labels and reference names, such as `<Video 1> = Alice_visual`. Connect it to a text display to see the mapping. Raw references are presented first (pictures, videos with soundtracks, standalone audio), followed by active RefMods in stack order. Counters are separate per media type. For example, if the map lists Alice as `<Video 1>` and Bob as `<Video 2>`:

```text
<Subject 1> is based on <Video 1>.
<Subject 2> is based on <Video 2>.
```

`stack_pictures` controls which reconstructed pictures from an image-stack RefMod the encoder sees. It never removes latent frames from the diffusion reference. Grounding helps the model associate labels with visual content; it does not enforce exclusive character assignments. Legacy combined visual/audio RefMods are supported. Do not Apply the same RefMods again after this node, as their latent references are already included.

### Include Encoder Images

`Create H3 RefMod From Folder` has an `include_images` switch, off by default. When enabled, it stores JPEG pictures reconstructed from the final visual latent, like Fantastic's creator, retaining every stored latent frame for image stacks. These are reconstructions, not copies of the original source images. Latents and audio are unchanged.

With the switch off, creation retains the existing latent-only behavior. Older RefMods without pictures remain supported by RefMods-to-Video through VAE reconstruction; adding pictures does not require converting existing files or change how their latents are applied.

The pictures are retained in both bundle and separate-file layouts and can be used by this add-on's RefMods-to-Video node without decoding again. Both layouts use Fantastic's `enc_N` tensor keys and `enc_times` / `enc_fps` metadata. In bundles, the picture metadata stays on the visual member; the optional audio member has no pictures. The version-5 bundle container is unchanged, so other loaders can read its standard latents, but reuse of its pictures depends on their bundle support. Enabling the switch adds a VAE decode during creation and increases file size. It also applies in subfolder mode and to unsaved mods passed directly to another node.

## Notes

- Supported RefMod files include older split `_visual` / `_audio` saves, single-file MiniMax bundles, and RefMods with stored encoder images, including the image-inclusive bundles added by this add-on. Existing latent-only files remain supported.
- For more options and the complete MiniMax H3 RefMod feature set, use [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod).

## Installation

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/FranckyB/ComfyUI-H3RefModPicker
```