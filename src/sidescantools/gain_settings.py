"""Portable per-sonar gain settings shared by the UI and batch workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile


SCHEMA_VERSION = 1
SIDECAR_SUFFIX = ".tvg_gain.cfg"


@dataclass(frozen=True, slots=True)
class SonarGainSettings:
    """Versioned settings required to reproduce one sonar file's display."""

    source_file: str
    overall_gain_db: float
    tvg_spreading_db_per_decade: float
    tvg_absorption_db_per_m: float
    auto_tvg_brightness_target_percent: int
    auto_tvg_active: bool
    auto_tvg_gain_db: tuple[float, ...]
    speed_correction_px_per_ping: float
    processing_mode: str
    egn_table_path: str | None
    destripe_active: bool = False
    slant_range_correction_active: bool = False

    def __post_init__(self) -> None:
        if self.processing_mode not in {"raw", "egn"}:
            raise ValueError("processing_mode must be 'raw' or 'egn'")
        for name in (
            "overall_gain_db",
            "tvg_spreading_db_per_decade",
            "tvg_absorption_db_per_m",
            "speed_correction_px_per_ping",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.speed_correction_px_per_ping <= 0:
            raise ValueError("speed_correction_px_per_ping must be positive")
        if not 1 <= self.auto_tvg_brightness_target_percent <= 100:
            raise ValueError(
                "auto_tvg_brightness_target_percent must be between 1 and 100"
            )
        if any(not math.isfinite(float(value)) for value in self.auto_tvg_gain_db):
            raise ValueError("auto_tvg_gain_db must contain only finite values")

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_file": self.source_file,
            "display": {
                "overall_gain_db": self.overall_gain_db,
                "tvg_spreading_db_per_decade": self.tvg_spreading_db_per_decade,
                "tvg_absorption_db_per_m": self.tvg_absorption_db_per_m,
                "auto_tvg_brightness_target_percent": (
                    self.auto_tvg_brightness_target_percent
                ),
                "auto_tvg_active": self.auto_tvg_active,
                "auto_tvg_gain_db": list(self.auto_tvg_gain_db),
                "speed_correction_px_per_ping": self.speed_correction_px_per_ping,
            },
            "processing": {
                "mode": self.processing_mode,
                "egn_table_path": self.egn_table_path,
                "destripe_active": self.destripe_active,
                "slant_range_correction_active": (
                    self.slant_range_correction_active
                ),
            },
        }

    @classmethod
    def from_dict(cls, values: dict) -> "SonarGainSettings":
        if values.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported TVG gain settings schema: {values.get('schema_version')}"
            )
        display = values["display"]
        processing = values["processing"]
        return cls(
            source_file=str(values["source_file"]),
            overall_gain_db=float(display["overall_gain_db"]),
            tvg_spreading_db_per_decade=float(
                display["tvg_spreading_db_per_decade"]
            ),
            tvg_absorption_db_per_m=float(display["tvg_absorption_db_per_m"]),
            auto_tvg_brightness_target_percent=int(
                display["auto_tvg_brightness_target_percent"]
            ),
            auto_tvg_active=bool(display["auto_tvg_active"]),
            auto_tvg_gain_db=tuple(
                float(value) for value in display.get("auto_tvg_gain_db", ())
            ),
            speed_correction_px_per_ping=float(
                display["speed_correction_px_per_ping"]
            ),
            processing_mode=str(processing["mode"]),
            egn_table_path=(
                str(processing["egn_table_path"])
                if processing.get("egn_table_path")
                else None
            ),
            destripe_active=bool(processing.get("destripe_active", False)),
            slant_range_correction_active=bool(
                processing.get("slant_range_correction_active", False)
            ),
        )


def gain_settings_path(sonar_path: str | os.PathLike) -> Path:
    """Return the unambiguous sidecar path for a JSF/XTF source file."""

    sonar_path = Path(sonar_path)
    return sonar_path.with_name(sonar_path.name + SIDECAR_SUFFIX)


def save_gain_settings(
    sonar_path: str | os.PathLike, settings: SonarGainSettings
) -> Path:
    """Atomically write settings next to their source sonar file."""

    sonar_path = Path(sonar_path)
    if settings.source_file != sonar_path.name:
        raise ValueError("gain settings source_file does not match the sonar file")
    output_path = gain_settings_path(sonar_path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=output_path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(settings.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return output_path


def load_gain_settings(
    sonar_path: str | os.PathLike,
) -> SonarGainSettings | None:
    """Load one sidecar, or return ``None`` when it has not been created."""

    input_path = gain_settings_path(sonar_path)
    if not input_path.is_file():
        return None
    with input_path.open("r", encoding="utf-8") as stream:
        settings = SonarGainSettings.from_dict(json.load(stream))
    if settings.source_file != Path(sonar_path).name:
        raise ValueError("gain settings source_file does not match the sonar file")
    return settings


def portable_egn_table_path(
    table_path: str | os.PathLike | None, sonar_path: str | os.PathLike
) -> str | None:
    """Store an EGN path relative to the sonar directory when possible."""

    if not table_path:
        return None
    table_path = Path(table_path).expanduser().resolve()
    sonar_directory = Path(sonar_path).resolve().parent
    try:
        return os.path.relpath(table_path, sonar_directory)
    except ValueError:
        # Windows paths on separate drives cannot be expressed relatively.
        return str(table_path)


def resolve_egn_table_path(
    settings: SonarGainSettings, sonar_path: str | os.PathLike
) -> Path | None:
    """Resolve a stored EGN path for UI or future batch processing use."""

    if settings.egn_table_path is None:
        return None
    path = Path(settings.egn_table_path)
    if not path.is_absolute():
        path = Path(sonar_path).resolve().parent / path
    return path.resolve()
