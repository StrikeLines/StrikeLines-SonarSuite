"""Deterministic chunk-spanning waterfall thumbnail extraction."""

from __future__ import annotations

from io import BytesIO
from operator import index
from typing import Callable

import numpy as np
from PIL import Image

from sidescantools.contact_model import ContactAnchor, ContactThumbnail
from sidescantools.contact_picker import display_position_for_anchor


class ThumbnailExtractionError(ValueError):
    pass


class ContactThumbnailExtractor:
    """Extract an annotated PNG chip from the logical two-dimensional waterfall."""

    display_pipeline = "napari-waterfall-v1|minmax-uint8|red-crosshair-v1"

    def __init__(
        self,
        *,
        preprocessor,
        sidescan_file,
        ping_radius: int = 100,
        sample_radius: int = 150,
        crosshair_radius: int = 4,
        logical_waterfall_provider: Callable[[], np.ndarray] | None = None,
        display_pipeline_provider: Callable[[], str] | None = None,
    ):
        self.preprocessor = preprocessor
        self.sidescan_file = sidescan_file
        self.ping_radius = self._nonnegative_integer("ping_radius", ping_radius)
        self.sample_radius = self._nonnegative_integer(
            "sample_radius", sample_radius
        )
        self.crosshair_radius = self._nonnegative_integer(
            "crosshair_radius", crosshair_radius
        )
        self.logical_waterfall_provider = logical_waterfall_provider
        self.display_pipeline_provider = display_pipeline_provider

    def __call__(self, anchor: ContactAnchor) -> ContactThumbnail:
        expected_width = 2 * self.preprocessor.ping_len
        if self.logical_waterfall_provider is not None:
            logical_waterfall = np.asarray(self.logical_waterfall_provider())
            if logical_waterfall.ndim != 2:
                raise ThumbnailExtractionError(
                    "logical waterfall must be two-dimensional"
                )
            if logical_waterfall.shape != (
                self.sidescan_file.num_ping,
                expected_width,
            ):
                raise ThumbnailExtractionError(
                    "logical waterfall dimensions do not match the source"
                )
        else:
            waterfall = np.asarray(self.preprocessor.napari_fullmat)
            if waterfall.ndim != 3:
                raise ThumbnailExtractionError(
                    "Napari waterfall must be three-dimensional"
                )
            if waterfall.shape[1] != self.preprocessor.chunk_size:
                raise ThumbnailExtractionError(
                    "waterfall chunk size does not match preprocessor"
                )
            if waterfall.shape[2] != expected_width:
                raise ThumbnailExtractionError(
                    "waterfall width does not match preprocessor"
                )
            available_rows = waterfall.shape[0] * waterfall.shape[1]
            if self.sidescan_file.num_ping > available_rows:
                raise ThumbnailExtractionError(
                    "waterfall does not contain every source ping"
                )
            logical_waterfall = waterfall.reshape(available_rows, expected_width)[
                : self.sidescan_file.num_ping
            ]
        if not 0 <= anchor.global_ping_index < self.sidescan_file.num_ping:
            raise ThumbnailExtractionError("contact ping is outside the source file")

        _, _, display_x = display_position_for_anchor(
            anchor,
            chunk_size=self.preprocessor.chunk_size,
            display_channel_width=self.preprocessor.ping_len,
        )
        ping_start = max(0, anchor.global_ping_index - self.ping_radius)
        ping_stop = min(
            self.sidescan_file.num_ping,
            anchor.global_ping_index + self.ping_radius + 1,
        )
        sample_start = max(0, display_x - self.sample_radius)
        sample_stop = min(expected_width, display_x + self.sample_radius + 1)
        crop = logical_waterfall[ping_start:ping_stop, sample_start:sample_stop]
        if crop.size == 0:
            raise ThumbnailExtractionError("contact crop is empty")

        image_data = self._normalize(crop)
        rgb = np.repeat(image_data[:, :, None], 3, axis=2)
        target_y = anchor.global_ping_index - ping_start
        target_x = display_x - sample_start
        self._draw_crosshair(rgb, target_y, target_x)

        buffer = BytesIO()
        Image.fromarray(rgb, mode="RGB").save(
            buffer,
            format="PNG",
            optimize=False,
            compress_level=9,
        )
        return ContactThumbnail(
            image_bytes=buffer.getvalue(),
            width_px=rgb.shape[1],
            height_px=rgb.shape[0],
            ping_radius=self.ping_radius,
            sample_radius=self.sample_radius,
            display_pipeline=(
                self.display_pipeline_provider()
                if self.display_pipeline_provider is not None
                else self.display_pipeline
            ),
        )

    @staticmethod
    def _normalize(crop: np.ndarray) -> np.ndarray:
        values = np.asarray(crop, dtype=float)
        finite = np.isfinite(values)
        output = np.zeros(values.shape, dtype=np.uint8)
        if not np.any(finite):
            return output
        minimum = float(np.min(values[finite]))
        maximum = float(np.max(values[finite]))
        if maximum <= minimum:
            return output
        scaled = (values[finite] - minimum) * (255.0 / (maximum - minimum))
        output[finite] = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
        return output

    def _draw_crosshair(self, rgb: np.ndarray, target_y: int, target_x: int) -> None:
        radius = self.crosshair_radius
        x_start = max(0, target_x - radius)
        x_stop = min(rgb.shape[1], target_x + radius + 1)
        y_start = max(0, target_y - radius)
        y_stop = min(rgb.shape[0], target_y + radius + 1)
        rgb[target_y, x_start:x_stop] = (255, 0, 0)
        rgb[y_start:y_stop, target_x] = (255, 0, 0)

    @staticmethod
    def _nonnegative_integer(name: str, value: int) -> int:
        try:
            result = index(value)
        except TypeError as exc:
            raise ThumbnailExtractionError(f"{name} must be an integer") from exc
        if result < 0:
            raise ThumbnailExtractionError(f"{name} must be non-negative")
        return result
