"""Public package interface for SidescanTools.

Imports are resolved lazily so lightweight domain modules (for example contact
models and exporters) do not require the optional GUI and sonar-reader stack.
The existing public names remain available through ``from sidescantools import``.
"""

from importlib import import_module
from typing import Any


_PUBLIC_IMPORTS = {
    "SidescanPreprocessor": ("sidescantools.sidescan_preproc", "SidescanPreprocessor"),
    "SidescanFile": ("sidescantools.sidescan_file", "SidescanFile"),
    "Georeferencer": ("sidescantools.georef_thread", "Georeferencer"),
    "generate_egn_info": ("sidescantools.egn_table_build", "generate_egn_info"),
    "generate_egn_table_from_infos": (
        "sidescantools.egn_table_build",
        "generate_egn_table_from_infos",
    ),
    "convert_to_dB": ("sidescantools.aux_functions", "convert_to_dB"),
    "hist_equalization": ("sidescantools.aux_functions", "hist_equalization"),
    "CFG": ("sidescantools.cfg_parser", "CFG"),
    "GAINSTRAT": ("sidescantools.cfg_parser", "GAINSTRAT"),
    "FileImportManager": ("sidescantools.custom_threading", "FileImportManager"),
    "EGNTableBuilder": ("sidescantools.custom_threading", "EGNTableBuilder"),
    "PreProcManager": ("sidescantools.custom_threading", "PreProcManager"),
    "NavPlotter": ("sidescantools.custom_threading", "NavPlotter"),
    "GeoreferencerManager": (
        "sidescantools.custom_threading",
        "GeoreferencerManager",
    ),
    "QHLine": ("sidescantools.custom_widgets", "QHLine"),
    "Labeled2Buttons": ("sidescantools.custom_widgets", "Labeled2Buttons"),
    "LabeledLineEdit": ("sidescantools.custom_widgets", "LabeledLineEdit"),
    "OverwriteWarnDialog": (
        "sidescantools.custom_widgets",
        "OverwriteWarnDialog",
    ),
    "ErrorWarnDialog": ("sidescantools.custom_widgets", "ErrorWarnDialog"),
    "FilePicker": ("sidescantools.custom_widgets", "FilePicker"),
    "run_napari_btm_line": (
        "sidescantools.bottom_detection_napari_ui",
        "run_napari_btm_line",
    ),
    "XTFWrapper": ("sidescantools.xtf_wrapper", "XTFWrapper"),
    "JSFFile": ("sidescantools.jsf", "JSFFile"),
    "JSFSystemInformation": ("sidescantools.jsf", "JSFSystemInformation"),
    "JSFSonarDataPacket": ("sidescantools.jsf", "JSFSonarDataPacket"),
    "Channel": ("sidescantools.contact_model", "Channel"),
    "ContactAnchor": ("sidescantools.contact_model", "ContactAnchor"),
    "ContactCoordinate": ("sidescantools.contact_model", "ContactCoordinate"),
    "ContactDraft": ("sidescantools.contact_model", "ContactDraft"),
    "ContactRecord": ("sidescantools.contact_model", "ContactRecord"),
    "ContactThumbnail": ("sidescantools.contact_model", "ContactThumbnail"),
    "CoordinateStatus": ("sidescantools.contact_model", "CoordinateStatus"),
    "ContactValidationError": (
        "sidescantools.contact_model",
        "ContactValidationError",
    ),
    "InvalidContactPixel": (
        "sidescantools.contact_picker",
        "InvalidContactPixel",
    ),
    "anchor_from_display_pixel": (
        "sidescantools.contact_picker",
        "anchor_from_display_pixel",
    ),
    "display_position_for_anchor": (
        "sidescantools.contact_picker",
        "display_position_for_anchor",
    ),
    "source_array_sample_for_anchor": (
        "sidescantools.contact_picker",
        "source_array_sample_for_anchor",
    ),
    "ContactPickerService": (
        "sidescantools.contact_picker",
        "ContactPickerService",
    ),
    "PickContactResult": (
        "sidescantools.contact_picker",
        "PickContactResult",
    ),
    "GeometrySettings": ("sidescantools.swath_geometry", "GeometrySettings"),
    "SwathGeometry": ("sidescantools.swath_geometry", "SwathGeometry"),
    "GeometryUnavailable": (
        "sidescantools.swath_geometry",
        "GeometryUnavailable",
    ),
    "ContactStore": ("sidescantools.contact_store", "ContactStore"),
    "ContactStoreError": ("sidescantools.contact_store", "ContactStoreError"),
    "DuplicateContactAnchor": (
        "sidescantools.contact_store",
        "DuplicateContactAnchor",
    ),
    "SourceFileRecord": ("sidescantools.contact_store", "SourceFileRecord"),
    "ContactThumbnailExtractor": (
        "sidescantools.contact_thumbnail",
        "ContactThumbnailExtractor",
    ),
    "ThumbnailExtractionError": (
        "sidescantools.contact_thumbnail",
        "ThumbnailExtractionError",
    ),
    "GPXExporter": ("sidescantools.contact_export", "GPXExporter"),
    "ExportResult": ("sidescantools.contact_export", "ExportResult"),
    "InteractionMode": ("sidescantools.interaction_mode", "InteractionMode"),
    "InteractionModeController": (
        "sidescantools.interaction_mode",
        "InteractionModeController",
    ),
    "ContactDock": ("sidescantools.contact_ui", "ContactDock"),
    "ContactTableModel": ("sidescantools.contact_ui", "ContactTableModel"),
}

__all__ = tuple(_PUBLIC_IMPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public symbol on first use and cache it on the package."""

    try:
        module_name, attribute_name = _PUBLIC_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
