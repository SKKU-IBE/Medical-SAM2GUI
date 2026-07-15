import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from gui.box_interactions import DeferredBoxCommit, set_shapes_mode


def _application():
    return QApplication.instance() or QApplication([])


class _FakeShapesLayer:
    def __init__(self, mode="add_rectangle"):
        self._mode = mode
        self.assignments = []

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self.assignments.append(value)
        self._mode = value


def test_box_commit_waits_for_mouse_release_and_runs_once():
    _application()
    mouse_state = {"pressed": True}
    commits = []
    committer = DeferredBoxCommit(
        None,
        lambda: commits.append("committed"),
        delay_ms=1,
        mouse_button_pressed=lambda: mouse_state["pressed"],
    )

    committer.schedule()
    committer._on_timeout()
    committer._on_timeout()

    assert committer.pending is True
    assert commits == []

    mouse_state["pressed"] = False
    committer._on_timeout()
    committer._on_timeout()

    assert committer.pending is False
    assert commits == ["committed"]


def test_repeated_box_updates_remain_pending_during_long_drag():
    _application()
    mouse_state = {"pressed": True}
    commits = []
    committer = DeferredBoxCommit(
        None,
        lambda: commits.append(1),
        delay_ms=1,
        mouse_button_pressed=lambda: mouse_state["pressed"],
    )

    for _ in range(20):
        committer.schedule()
        committer._on_timeout()

    assert committer.pending is True
    assert commits == []

    mouse_state["pressed"] = False
    assert committer.flush() is True
    assert commits == [1]


def test_reactivating_same_shapes_mode_bounces_through_pan_zoom():
    layer = _FakeShapesLayer(mode="add_rectangle")

    set_shapes_mode(layer, "add_rectangle")

    assert layer.assignments == ["pan_zoom", "add_rectangle"]


def test_changing_shapes_mode_does_not_add_an_unnecessary_bounce():
    layer = _FakeShapesLayer(mode="select")

    set_shapes_mode(layer, "add_rectangle")

    assert layer.assignments == ["add_rectangle"]
