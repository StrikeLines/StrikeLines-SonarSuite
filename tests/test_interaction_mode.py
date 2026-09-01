import unittest

from sidescantools.interaction_mode import InteractionMode, InteractionModeController


class InteractionModeControllerTests(unittest.TestCase):
    def test_edit_modes_are_mutually_exclusive(self):
        controller = InteractionModeController()

        controller.set_mode(InteractionMode.BOTTOM_EDIT)
        controller.set_mode(InteractionMode.TARGET_PICK)

        self.assertIs(controller.mode, InteractionMode.TARGET_PICK)

    def test_toggling_active_edit_mode_returns_to_pan(self):
        controller = InteractionModeController(InteractionMode.TARGET_PICK)

        result = controller.toggle(InteractionMode.TARGET_PICK)

        self.assertIs(result, InteractionMode.PAN_ZOOM)

    def test_toggling_other_edit_mode_switches_directly(self):
        controller = InteractionModeController(InteractionMode.BOTTOM_EDIT)

        result = controller.toggle(InteractionMode.TARGET_PICK)

        self.assertIs(result, InteractionMode.TARGET_PICK)

    def test_listeners_receive_changes_once_and_can_unsubscribe(self):
        controller = InteractionModeController()
        changes = []
        remove = controller.add_listener(changes.append)

        controller.set_mode(InteractionMode.BOTTOM_EDIT)
        controller.set_mode(InteractionMode.BOTTOM_EDIT)
        remove()
        controller.set_mode(InteractionMode.PAN_ZOOM)

        self.assertEqual(changes, [InteractionMode.BOTTOM_EDIT])


if __name__ == "__main__":
    unittest.main()
