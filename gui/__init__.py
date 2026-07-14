"""GUI package with compatibility-preserving lazy exports."""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "MedSAM2NapariGUI": ("gui.auto_gui", "MedSAM2NapariGUI"),
    "ManualPromptNapariGUI": ("gui.manual_gui", "ManualPromptNapariGUI"),
    "PatientNavigationManager": ("gui.navigation", "PatientNavigationManager"),
    "MedSAM2NapariGUIWithNavigation": (
        "gui.navigation",
        "MedSAM2NapariGUIWithNavigation",
    ),
    "ManualPromptNapariGUIWithNavigation": (
        "gui.navigation",
        "ManualPromptNapariGUIWithNavigation",
    ),
    "run_napari_gui_with_navigation": (
        "gui.navigation",
        "run_napari_gui_with_navigation",
    ),
    "render_auto_volume": ("gui.rendering", "render_auto_volume"),
    "render_manual_volume": ("gui.rendering", "render_manual_volume"),
    "load_masks_manual": ("gui.io", "load_masks_manual"),
    "save_masks_auto": ("gui.io", "save_masks_auto"),
    "save_masks_manual": ("gui.io", "save_masks_manual"),
    "auto_segmentation": ("gui.segmentation", "auto_segmentation"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
