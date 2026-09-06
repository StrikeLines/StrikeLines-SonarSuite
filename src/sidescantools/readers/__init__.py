"""Built-in sonar readers and the public reader-registry API."""

from __future__ import annotations

from pathlib import Path

from sidescantools.readers.base import (
    SonarDataset,
    SonarReader,
    SonarReaderRegistry,
    UnsupportedSonarFormatError,
)
from sidescantools.readers.jsf import JSFReader, jsf_tow_data
from sidescantools.readers.rsd import RSDReader
from sidescantools.readers.xtf import XTFReader, xtf_tow_data


sonar_reader_registry = SonarReaderRegistry()
sonar_reader_registry.register(JSFReader())
sonar_reader_registry.register(XTFReader())
sonar_reader_registry.register(RSDReader())


def register_sonar_reader(reader: SonarReader, *, replace: bool = False) -> None:
    """Register a reader, allowing extensions to be added without UI edits."""

    sonar_reader_registry.register(reader, replace=replace)


def unregister_sonar_reader(reader_or_format_id: SonarReader | str) -> None:
    sonar_reader_registry.unregister(reader_or_format_id)


def load_sonar_dataset(
    filepath: str | Path, *, choose_subsys: int = 0
) -> SonarDataset:
    return sonar_reader_registry.read(filepath, choose_subsys=choose_subsys)


def supported_sonar_suffixes() -> tuple[str, ...]:
    return sonar_reader_registry.supported_suffixes()


def supported_sonar_format_ids() -> tuple[str, ...]:
    return sonar_reader_registry.format_ids()


def is_supported_sonar_path(filepath: str | Path) -> bool:
    return sonar_reader_registry.supports(filepath)


def sonar_file_dialog_filter() -> str:
    patterns = " ".join(f"*{suffix}" for suffix in supported_sonar_suffixes())
    return f"Sidescan files ({patterns});;All files (*)"


__all__ = [
    "JSFReader",
    "RSDReader",
    "SonarDataset",
    "SonarReader",
    "SonarReaderRegistry",
    "UnsupportedSonarFormatError",
    "XTFReader",
    "is_supported_sonar_path",
    "jsf_tow_data",
    "load_sonar_dataset",
    "register_sonar_reader",
    "sonar_file_dialog_filter",
    "sonar_reader_registry",
    "supported_sonar_format_ids",
    "supported_sonar_suffixes",
    "unregister_sonar_reader",
    "xtf_tow_data",
]
