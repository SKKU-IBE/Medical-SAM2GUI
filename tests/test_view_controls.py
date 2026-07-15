import numpy as np
import pytest

from gui.view_controls import ViewRotationController, rotate_view_90


class _FakeDims:
    def __init__(self):
        self.order = (0, 1, 2)
        self.ndisplay = 2

    def transpose(self):
        order = list(self.order)
        order[-2], order[-1] = order[-1], order[-2]
        self.order = tuple(order)


class _FakeCamera:
    def __init__(self):
        self.orientation2d = ("down", "right")
        self.center = (4.0, 5.0, 6.0)
        self.zoom = 2.5


class _FakeViewer:
    def __init__(self):
        self.dims = _FakeDims()
        self.camera = _FakeCamera()
        self.layers = [np.arange(12).reshape(3, 4)]


def test_four_clockwise_rotations_restore_initial_view_without_touching_data():
    viewer = _FakeViewer()
    original = viewer.layers[0].copy()
    original_center = viewer.camera.center
    original_zoom = viewer.camera.zoom

    expected = (
        ((0, 2, 1), ("down", "left")),
        ((0, 1, 2), ("up", "left")),
        ((0, 2, 1), ("up", "right")),
        ((0, 1, 2), ("down", "right")),
    )
    for order, orientation in expected:
        rotate_view_90(viewer, clockwise=True)
        assert viewer.dims.order == order
        assert viewer.camera.orientation2d == orientation

    np.testing.assert_array_equal(viewer.layers[0], original)
    assert viewer.camera.center == original_center
    assert viewer.camera.zoom == original_zoom


def test_left_and_right_rotations_are_inverses():
    viewer = _FakeViewer()
    rotate_view_90(viewer, clockwise=False)
    assert viewer.dims.order == (0, 2, 1)
    assert viewer.camera.orientation2d == ("up", "right")

    rotate_view_90(viewer, clockwise=True)
    assert viewer.dims.order == (0, 1, 2)
    assert viewer.camera.orientation2d == ("down", "right")


def test_rotation_controller_resets_captured_orientation():
    viewer = _FakeViewer()
    controller = ViewRotationController(viewer)
    controller.rotate_right()
    controller.rotate_right()
    controller.reset()

    assert viewer.dims.order == (0, 1, 2)
    assert viewer.camera.orientation2d == ("down", "right")


def test_rotation_rejects_3d_display():
    viewer = _FakeViewer()
    viewer.dims.ndisplay = 3
    with pytest.raises(ValueError, match="only available in 2D"):
        rotate_view_90(viewer)
