"""Mutually exclusive interaction state shared by the Napari controls."""

from __future__ import annotations

from enum import Enum
from typing import Callable


class InteractionMode(str, Enum):
    PAN_ZOOM = "pan_zoom"
    BOTTOM_EDIT = "bottom_edit"
    TARGET_PICK = "target_pick"

    @property
    def label(self) -> str:
        return {
            InteractionMode.PAN_ZOOM: "Pan/Zoom",
            InteractionMode.BOTTOM_EDIT: "Bottom Edit",
            InteractionMode.TARGET_PICK: "Target Pick",
        }[self]


class InteractionModeController:
    """Small observable state machine that cannot activate two edit modes."""

    def __init__(self, mode: InteractionMode = InteractionMode.PAN_ZOOM):
        self._mode = InteractionMode(mode)
        self._listeners: list[Callable[[InteractionMode], None]] = []

    @property
    def mode(self) -> InteractionMode:
        return self._mode

    def set_mode(self, mode: InteractionMode) -> InteractionMode:
        mode = InteractionMode(mode)
        if mode is self._mode:
            return self._mode
        self._mode = mode
        for listener in tuple(self._listeners):
            listener(mode)
        return mode

    def toggle(self, mode: InteractionMode) -> InteractionMode:
        mode = InteractionMode(mode)
        if mode is InteractionMode.PAN_ZOOM:
            return self.set_mode(mode)
        return self.set_mode(
            InteractionMode.PAN_ZOOM if self._mode is mode else mode
        )

    def add_listener(
        self, listener: Callable[[InteractionMode], None]
    ) -> Callable[[], None]:
        self._listeners.append(listener)

        def remove() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return remove
