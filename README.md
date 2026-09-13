# ComfyUI-H3RefModPicker

Companion add-on for [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)

This add-on is meant to work alongside **MiniMaxH3Mod**. It provides only a small, focused subset of RefMod features: a Visual Picker, a simple loader, an apply axis, and a batch Create From Folder node, while keeping compatibility with both the original RefMod format and the newer bundle format.

If you want the fuller MiniMax H3 RefMod toolset and more creation/apply options, the recommended install is [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod).

Please note, the `Create from Input` node was removed from this add-on. The same functionality is available in the official MiniMaxH3Mod add-on, using the `Create H3 RefMod Master` node.


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
- For legacy split RefMods, found pairs are grouped as one item in the picker. They are shown as one entry and loaded together, with separate video/audio weight controls. This uses a weight behavior: `0..1` is regular strength, values above `1` expand into repeated copies. For example, a weight of 2.7 would be the same as strength: 1.0, copies: 2.7.
- `Create H3 RefMod From Folder` scans a folder of images, video, and audio, and now saves in the new bundle format by default. It can also batch-create RefMods for all subfolders found.
- `Load RefMod Simple` is a singular loader. It supports standalone visual/audio RefMods and the new bundle format, while still using the same weight behavior as the Picker.
- `Apply H3 RefMod Simple` is a streamlined version of the Apply H3 RefMod, without the extra controls.
- `Apply H3 RefMod Axis` provides a streamlined option to slide between two RefMods. It is a smaller variant of `Load H3 RefMod Axis`.
- A `generate_video_thumbnails.py` script is provided to help generate RefMod thumbnails from `.mp4` files. It can be found in `tools`. It grabs a random frame between 25% and 75% for each clip found in a folder. It requires `ffmpeg` and `ffprobe`.

## Notes

- This repo is a minimal companion add-on, not the full RefMod toolkit.
- Supported RefMod files now include both older split `_visual` / `_audio` style saves and newer single-file MiniMax bundle saves.
- For more options and the complete MiniMax H3 RefMod feature set, [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod) is the recommended install.

## Installation

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/FranckyB/ComfyUI-H3RefModPicker
```