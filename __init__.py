"""
ComfyUI-H3RefModPicker — Addons to MiniMaxH3Mod, adds a visual picker and a batch creation tool.
"""

__author__      = "FranckB"
__version__         = "0.1.6"

from .nodes import refmod_apply
from .nodes import refmod_loader
from .nodes import refmod_visual_picker
from .nodes import refmod_create
from .nodes import refmod_axis
#  from .nodes import refmod_decode

NODE_CLASS_MAPPINGS = {
    **refmod_apply.NODE_CLASS_MAPPINGS,
    **refmod_loader.NODE_CLASS_MAPPINGS,
    **refmod_visual_picker.NODE_CLASS_MAPPINGS,
    **refmod_create.NODE_CLASS_MAPPINGS,
    **refmod_axis.NODE_CLASS_MAPPINGS,
    # **refmod_decode.NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **refmod_apply.NODE_DISPLAY_NAME_MAPPINGS,
    **refmod_loader.NODE_DISPLAY_NAME_MAPPINGS,
    **refmod_visual_picker.NODE_DISPLAY_NAME_MAPPINGS,
    **refmod_create.NODE_DISPLAY_NAME_MAPPINGS,
    **refmod_axis.NODE_DISPLAY_NAME_MAPPINGS,
    # **refmod_decode.NODE_DISPLAY_NAME_MAPPINGS,
}
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
