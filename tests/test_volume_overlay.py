import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QWidget

from gui.volume_overlay import (
    VolumeOverlayWidget,
    format_volume_row,
    label_color_hex,
)


class _FakeLabelsLayer:
    _colors = {
        1: np.array([1.0, 0.5, 0.0, 1.0]),
        2: np.array([12.0, 34.0, 56.0, 255.0]),
    }

    def get_color(self, label_id):
        return self._colors[label_id]


def _application():
    return QApplication.instance() or QApplication([])


def test_label_color_hex_supports_normalized_and_byte_rgb():
    layer = _FakeLabelsLayer()
    assert label_color_hex(layer, 1) == "#ff8000"
    assert label_color_hex(layer, 2) == "#0c2238"


def test_volume_row_format_preserves_existing_units():
    assert format_volume_row(2, 1905.15) == "Obj 2: 1,905.15 mm3 (1.905 mL)"


def test_overlay_rows_match_label_colors_and_follow_canvas_resize():
    app = _application()
    canvas = QWidget()
    canvas.resize(640, 480)
    canvas.show()
    overlay = VolumeOverlayWidget(canvas)
    overlay.set_entries([(2, 2000.0), (1, 1000.0)], _FakeLabelsLayer())
    app.processEvents()

    assert list(overlay._rows) == [1, 2]
    assert "#ff8000" in overlay._rows[1].swatch.styleSheet()
    assert "#0c2238" in overlay._rows[2].swatch.styleSheet()
    assert overlay.testAttribute(Qt.WA_TransparentForMouseEvents)
    assert overlay.x() == canvas.width() - overlay.width() - overlay._MARGIN
    assert overlay.y() == overlay._MARGIN

    canvas.resize(800, 600)
    app.processEvents()
    assert overlay.x() == canvas.width() - overlay.width() - overlay._MARGIN
    assert overlay.y() == overlay._MARGIN

    overlay.close()
    canvas.close()
