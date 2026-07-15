import numpy as np

from qtpy.QtCore import QEvent, Qt, QTimer
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


_FALLBACK_LABEL_COLOR = "#808080"


def label_color_hex(labels_layer, label_id):
    """Return the displayed Napari label color as a CSS hex string."""
    try:
        color = labels_layer.get_color(int(label_id))
        if color is None:
            return _FALLBACK_LABEL_COLOR
        values = np.asarray(color, dtype=float).reshape(-1)
        if values.size < 3 or not np.all(np.isfinite(values[:3])):
            return _FALLBACK_LABEL_COLOR
        rgb = values[:3]
        if float(np.max(rgb)) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.rint(np.clip(rgb, 0.0, 255.0)).astype(int)
        return "#{:02x}{:02x}{:02x}".format(*rgb)
    except Exception:
        return _FALLBACK_LABEL_COLOR


def format_volume_row(object_id, volume_mm3):
    return (
        f"Obj {int(object_id)}: {float(volume_mm3):,.2f} mm3 "
        f"({float(volume_mm3) / 1000.0:,.3f} mL)"
    )


def get_viewer_canvas_widget(viewer):
    window = getattr(viewer, "window", None)
    for attr in ("_qt_viewer", "qt_viewer"):
        qt_viewer = getattr(window, attr, None)
        canvas = getattr(qt_viewer, "canvas", None)
        native = getattr(canvas, "native", None)
        if isinstance(native, QWidget):
            return native
        if isinstance(canvas, QWidget):
            return canvas
    return None


class _VolumeRow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.swatch = QFrame(self)
        self.swatch.setFixedSize(11, 11)
        self.text = QLabel(self)
        self.text.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(self.swatch)
        layout.addWidget(self.text)

    def update_content(self, color, text):
        self.swatch.setStyleSheet(
            f"background-color: {color}; border: 1px solid #d0d0d0;"
        )
        self.text.setText(text)


class VolumeOverlayWidget(QFrame):
    """Mouse-transparent volume legend anchored to a Napari canvas."""

    _MARGIN = 10

    def __init__(self, canvas_widget):
        if canvas_widget is None:
            raise ValueError("A canvas widget is required for the volume overlay.")
        super().__init__(canvas_widget)
        self.setObjectName("medicalSam2VolumeOverlay")
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet(
            "QFrame#medicalSam2VolumeOverlay {"
            "background-color: rgba(20, 20, 20, 185);"
            "border: 1px solid rgba(255, 255, 255, 80);"
            "border-radius: 4px;"
            "}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(3)
        self.title = QLabel("Volumes (source grid)", self)
        self.title.setStyleSheet(
            "color: white; background: transparent; font-weight: 600;"
        )
        layout.addWidget(self.title)

        self.rows_layout = QVBoxLayout()
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(2)
        layout.addLayout(self.rows_layout)

        self.empty_label = QLabel("No mask", self)
        self.empty_label.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(self.empty_label)

        self.total_label = QLabel(self)
        self.total_label.setStyleSheet(
            "color: white; background: transparent; font-weight: 600;"
        )
        self.total_label.hide()
        layout.addWidget(self.total_label)

        self._rows = {}
        canvas_widget.installEventFilter(self)
        self.show()
        self.raise_()
        QTimer.singleShot(0, self._resize_and_reposition)

    def set_entries(self, entries, labels_layer):
        entries = sorted(entries, key=lambda item: int(item[0]))
        active_ids = {int(item[0]) for item in entries}

        for object_id in list(self._rows):
            if object_id not in active_ids:
                row = self._rows.pop(object_id)
                self.rows_layout.removeWidget(row)
                row.deleteLater()

        for index, (object_id, volume_mm3) in enumerate(entries):
            object_id = int(object_id)
            row = self._rows.get(object_id)
            if row is None:
                row = _VolumeRow(self)
                self._rows[object_id] = row
            self.rows_layout.removeWidget(row)
            self.rows_layout.insertWidget(index, row)
            row.update_content(
                label_color_hex(labels_layer, object_id),
                format_volume_row(object_id, volume_mm3),
            )
            row.show()

        if entries:
            total = sum(float(item[1]) for item in entries)
            self.empty_label.hide()
            self.total_label.setText(
                f"Total: {total:,.2f} mm3 ({total / 1000.0:,.3f} mL)"
            )
            self.total_label.show()
        else:
            self.empty_label.show()
            self.total_label.hide()

        self.show()
        self.raise_()
        self._resize_and_reposition()

    def _resize_and_reposition(self):
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        x_pos = max(self._MARGIN, parent.width() - self.width() - self._MARGIN)
        self.move(x_pos, self._MARGIN)
        self.raise_()

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() in (
            QEvent.Resize,
            QEvent.Show,
        ):
            QTimer.singleShot(0, self._resize_and_reposition)
        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        parent = self.parentWidget()
        if parent is not None:
            parent.removeEventFilter(self)
        super().closeEvent(event)
