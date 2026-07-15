from qtpy.QtCore import QObject, QTimer, Qt
from qtpy.QtWidgets import QApplication


def left_mouse_button_pressed():
    return bool(QApplication.mouseButtons() & Qt.LeftButton)


def set_shapes_mode(layer, mode_name):
    """Restore Napari's native callbacks when reactivating a Shapes mode."""
    current_mode = getattr(layer, "mode", None)
    current_mode = getattr(current_mode, "value", current_mode)
    mode_name = str(mode_name)
    if str(current_mode).lower() == mode_name.lower():
        layer.mode = "pan_zoom"
    layer.mode = mode_name


class DeferredBoxCommit(QObject):
    """Run Shapes data synchronization only after the mouse button is released."""

    def __init__(
        self,
        parent,
        callback,
        delay_ms=30,
        mouse_button_pressed=None,
    ):
        super().__init__(parent)
        self._callback = callback
        self._mouse_button_pressed = (
            mouse_button_pressed or left_mouse_button_pressed
        )
        self._pending = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(int(delay_ms))
        self.timer.timeout.connect(self._on_timeout)

    @property
    def pending(self):
        return self._pending

    def schedule(self):
        self._pending = True
        self.timer.start()

    def flush(self):
        if not self._pending:
            return False
        if self._mouse_button_pressed():
            self.timer.start()
            return False

        self.timer.stop()
        self._pending = False
        self._callback()
        return True

    def cancel(self):
        self.timer.stop()
        self._pending = False

    def _on_timeout(self):
        self.flush()
