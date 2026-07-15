from dataclasses import dataclass

from qtpy.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QWidget


@dataclass(frozen=True)
class ViewOrientationState:
    order: tuple[int, ...]
    orientation2d: tuple[str, str]


def _orientation_value(value):
    return getattr(value, "value", str(value))


def capture_view_orientation(viewer):
    orientation = tuple(
        _orientation_value(value) for value in viewer.camera.orientation2d
    )
    return ViewOrientationState(tuple(viewer.dims.order), orientation)


def rotate_view_90(viewer, clockwise=True):
    """Rotate the current 2D view without transforming layer data."""
    if int(viewer.dims.ndisplay) != 2:
        raise ValueError("View rotation is only available in 2D display mode.")

    vertical, horizontal = (
        _orientation_value(value) for value in viewer.camera.orientation2d
    )
    if vertical not in {"up", "down"} or horizontal not in {"left", "right"}:
        raise ValueError(
            f"Unsupported camera orientation: {(vertical, horizontal)}"
        )

    viewer.dims.transpose()
    if clockwise:
        new_vertical = "down" if horizontal == "right" else "up"
        new_horizontal = "left" if vertical == "down" else "right"
    else:
        new_vertical = "up" if horizontal == "right" else "down"
        new_horizontal = "right" if vertical == "down" else "left"
    viewer.camera.orientation2d = (new_vertical, new_horizontal)


def restore_view_orientation(viewer, state):
    viewer.dims.order = tuple(state.order)
    viewer.camera.orientation2d = tuple(state.orientation2d)


class ViewRotationController:
    def __init__(self, viewer):
        self.viewer = viewer
        self.initial_state = capture_view_orientation(viewer)

    def rotate_left(self):
        rotate_view_90(self.viewer, clockwise=False)

    def rotate_right(self):
        rotate_view_90(self.viewer, clockwise=True)

    def reset(self):
        restore_view_orientation(self.viewer, self.initial_state)


def create_view_rotation_controls(parent, viewer):
    """Create compact rotation controls and retain their controller on the widget."""
    widget = QWidget(parent)
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    controller = ViewRotationController(viewer)

    controls = (
        ("Left 90", controller.rotate_left, "Rotate view 90 degrees counter-clockwise"),
        ("Reset", controller.reset, "Restore the initial view orientation"),
        ("Right 90", controller.rotate_right, "Rotate view 90 degrees clockwise"),
    )

    def run(action):
        try:
            action()
        except ValueError as exc:
            QMessageBox.information(parent, "View Rotation", str(exc))

    for label, action, tooltip in controls:
        button = QPushButton(label, widget)
        button.setToolTip(tooltip)
        button.setMinimumWidth(54)
        button.clicked.connect(lambda checked=False, fn=action: run(fn))
        layout.addWidget(button)

    widget.rotation_controller = controller
    return widget
