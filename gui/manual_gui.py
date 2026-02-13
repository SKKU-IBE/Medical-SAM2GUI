import os
import numpy as np
import torch
import napari
import time
from collections import deque, defaultdict
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QPushButton,
    QMessageBox,
    QShortcut,
)
from qtpy.QtGui import QKeySequence
from qtpy.QtCore import Qt
import threading
from PIL import Image

from gui.rendering import render_manual_volume
from gui.io import save_masks_manual


class ManualPromptNapariGUI(QWidget):
    """Napari GUI for manual prompt input (points/boxes) with propagate."""
    def __init__(self, imgs, net, device, patient_id, meta, metrics=None):
        super().__init__()
        if imgs.dim() == 5 and imgs.size(0) == 1:
            imgs = imgs.squeeze(0)
        self.imgs = imgs
        self.net = net
        self.device = device
        self.patient_id = patient_id
        self.meta = meta
        self.metrics = metrics
        self.session_started_at = time.time()
        self.first_prompt_ts = None
        self.manual_edit_started_at = None
        self.manual_edit_total_sec = 0.0

        self.n_frames = imgs.shape[0]
        self.frame_idx = 0
        self.current_obj_id = 1

        self.prompt_history = deque()
        self.redo_history = deque()
        self.pos_points = []  # (t, obj_id, y, x)
        self.neg_points = []
        self.box_prompts = []  # (t, obj_id, x1, y1, x2, y2)
        self._box_edit_timer = None
        self._updating_layers = False

        self.viewer = napari.Viewer(title=self._get_patient_display_name())
        self.viewer.bind_key('Escape', self.cancel_prompt_mode)
        self.img_layer = self.viewer.add_image(
            self.imgs.permute(0,2,3,1).cpu().numpy(), name='Image', rgb=True
        )
        self.mask_layer = self.viewer.add_labels(
            np.zeros((self.n_frames,) + self.imgs.shape[2:], dtype=np.uint8), name='Mask'
        )
        try:
            # Track last non-zero opacity so toggle can restore it
            self._last_nonzero_mask_opacity = float(self.mask_layer.opacity)
        except Exception:
            self._last_nonzero_mask_opacity = 1.0
        self.user_pts_layer = self.viewer.add_points(
            np.empty((0,3)), name='Points', size=5
        )
        # Initialize empty RGBA to avoid napari illegal color warnings
        empty_rgba = np.zeros((0, 4), dtype=float)
        try:
            self.user_pts_layer.face_color = empty_rgba
            self.user_pts_layer.edge_color = empty_rgba
        except Exception:
            pass
        self.box_layer = self.viewer.add_shapes(
            np.empty((0,4,3)), name='Boxes', shape_type='rectangle',
            edge_color='red', face_color=[0,0,0,0], edge_width=2, ndim=3
        )
        self._attach_mode_guard(self.user_pts_layer, allowed_modes=('select', 'pan_zoom'))
        # allow select/pan and add_rectangle so drag-to-add works
        self._attach_mode_guard(self.box_layer, allowed_modes=('select', 'pan_zoom', 'add_rectangle'))
        self.manual_edit_enabled = False  # allow napari painting/box drawing when toggled
        self.active_tool = None  # track current prompt/edit tool for toggling
        self.mask_history = deque()
        self.mask_redo_stack = deque()
        self._manual_stroke_active = False
        self._stroke_start_state = None
        self._suppress_box_history = False
        try:
            self._last_mask_state = self.mask_layer.data.copy()
            # Seed history so the first undo returns to the initial mask state
            self._push_mask_history(self._last_mask_state.copy())
        except Exception:
            self._last_mask_state = None
        # Button references for visual states
        self.btn_add_pos = None
        self.btn_add_neg = None
        self.btn_add_box = None
        self.manual_edit_button = None
        self.btn_edit_points = None
        self.btn_edit_boxes = None

        self.show_object_ids = True
        try:
            self.text_layer = self.viewer.add_points(
                np.empty((0,3)), name='Object IDs', size=0,
                face_color='transparent'
            )
            self.text_supported = True
        except Exception:
            self.text_layer = None
            self.text_supported = False

        self._build_controls()
        self.viewer.window.add_dock_widget(self, area='right')
        self._setup_viewer_callbacks()
        self._setup_layer_callbacks()
        self.update_layers()
        self._bind_shortcuts()
        self._bind_qshortcuts()
        if self.metrics and self.metrics.is_active():
            self.metrics.add_event('manual_gui_initialized', n_frames=self.n_frames, patient_id=str(self.patient_id))
            self.metrics.set_info('n_frames', int(self.n_frames))

    def _format_patient_id(self, patient_id):
        try:
            pid = patient_id[0] if isinstance(patient_id, (list, tuple)) else patient_id
        except Exception:
            pid = patient_id
        pid_str = '' if pid is None else str(pid)
        if not pid_str:
            return "Unknown"
        norm = os.path.normpath(pid_str)
        parts = norm.replace('\\', '/').split('/')
        if len(parts) >= 2:
            basename = parts[-1]
            parent = '/'.join(parts[:-1])
            return f"{basename} ({parent})"
        return pid_str

    def _get_patient_display_name(self):
        base = self._format_patient_id(self.patient_id)
        try:
            nav = getattr(self, 'navigation_manager', None)
            if nav and getattr(nav, 'patient_index', None) is not None:
                return f"Patient {nav.patient_index}: {base}"
        except Exception:
            pass
        return base

    def _setup_viewer_callbacks(self):
        @self.viewer.dims.events.current_step.connect
        def on_step_change(event):
            current_frame = self.viewer.dims.current_step[0]
            if current_frame != self.frame_idx:
                self.frame_idx = int(current_frame)
                self.frame_spin.blockSignals(True)
                self.frame_spin.setValue(self.frame_idx)
                self.frame_spin.blockSignals(False)
                self.current_frame_label.setText(str(self.frame_idx))

    def _setup_layer_callbacks(self):
        @self.user_pts_layer.events.data.connect
        def on_points_data_change():
            if not hasattr(self, 'user_pts_layer') or not self.user_pts_layer.editable:
                return
            if getattr(self, '_updating_layers', False):
                return
            self._sync_points_from_layer()
        self._setup_manual_box_editing_events()

        def _on_mask_change(event=None):
            if not getattr(self, 'manual_edit_enabled', False):
                return
            if getattr(self, '_manual_stroke_active', False):
                return
            if getattr(self, '_updating_layers', False):
                return
            try:
                current = self.mask_layer.data.copy()
                if self._last_mask_state is not None:
                    # Only push when data actually changed
                    if not np.array_equal(current, self._last_mask_state):
                        self._push_mask_history(self._last_mask_state.copy())
                        self.mask_redo_stack.clear()
                self._last_mask_state = current
            except Exception:
                pass
            if self.metrics and self.metrics.is_active():
                self.metrics.inc_counter('manual_edit_strokes', 1)
                self.metrics.add_event('manual_edit_stroke', frame=int(self.frame_idx))

        for _evt in ('data', 'set_data'):
            try:
                getattr(self.mask_layer.events, _evt).connect(_on_mask_change)
            except Exception:
                pass

    def _manual_edit_stroke_callback(self, layer, event):
        if not getattr(self, 'manual_edit_enabled', False):
            return
        if getattr(self, '_updating_layers', False):
            return
        self._manual_stroke_active = True
        try:
            self._stroke_start_state = self.mask_layer.data.copy()
        except Exception:
            self._stroke_start_state = None
        yield
        try:
            new_state = self.mask_layer.data.copy()
            if self._stroke_start_state is not None and not np.array_equal(new_state, self._stroke_start_state):
                self._push_mask_history(self._stroke_start_state.copy())
                self.mask_redo_stack.clear()
            self._last_mask_state = new_state
        except Exception:
            pass
        finally:
            self._manual_stroke_active = False
            self._stroke_start_state = None

    def _select_layer(self, layer):
        try:
            # ensure napari marks this as the sole active layer
            try:
                self.viewer.layers.selection.select_only(layer)
            except Exception:
                self.viewer.layers.selection = [layer]
                self.viewer.layers.selection.active = layer
            self._focus_viewer_canvas()
        except Exception:
            pass

    def _focus_viewer_canvas(self):
        try:
            if hasattr(self.viewer, 'window') and hasattr(self.viewer.window, 'qt_viewer'):
                canvas = getattr(self.viewer.window.qt_viewer, 'canvas', None)
                if canvas and hasattr(canvas, 'native'):  # napari uses vispy canvas
                    canvas.native.setFocus()
        except Exception:
            pass

    def _focus_mask_and_box_layers(self):
        # Make sure the mask/box layer is selected and visible while adding boxes
        try:
            self.mask_layer.visible = True
            self.box_layer.visible = True
        except Exception:
            pass
        self._select_layer(self.mask_layer)

    def _bind_shortcuts(self):
        keymap = [
            ('h', self.enable_add_positive),
            ('j', self.enable_add_negative),
            ('r', self.enable_add_box),
            ('q', self.toggle_manual_annotation),
            ('k', self.enable_edit_points),
            ('t', self.enable_edit_boxes),
            ('c', self.clear_all_prompts),
            ('y', self._toggle_mask_opacity),
            ('o', lambda: self._set_mask_opacity(1.0)),
            ('u', lambda: self._bump_mask_opacity(-0.1)),
            ('i', lambda: self._bump_mask_opacity(0.1)),
        ]
        def _bind_global(seq, fn, overwrite=True):
            try:
                self.viewer.bind_key(seq, lambda v: fn(), overwrite=overwrite, bind_global=True)
            except TypeError:
                self.viewer.bind_key(seq, lambda v: fn(), overwrite=overwrite)

        for key, handler in keymap:
            _bind_global(key, handler, overwrite=True)
        _bind_global('s', self.save_masks, overwrite=True)
        _bind_global('Control-Z', self.prompt_undo, overwrite=True)
        _bind_global('Control-Y', self.prompt_redo, overwrite=True)
        _bind_global('Control-X', self.mask_undo, overwrite=True)
        _bind_global('Control-U', self.mask_redo, overwrite=True)
        _bind_global('Control-S', self.save_masks, overwrite=True)
        _bind_global('Ctrl+Z', self.prompt_undo, overwrite=True)
        _bind_global('Ctrl+Y', self.prompt_redo, overwrite=True)
        _bind_global('Ctrl+X', self.mask_undo, overwrite=True)
        _bind_global('Ctrl+U', self.mask_redo, overwrite=True)
        _bind_global('Ctrl+S', self.save_masks, overwrite=True)
        _bind_global('Alt+Z', self.mask_undo, overwrite=True)
        _bind_global('Alt+Y', self.mask_redo, overwrite=True)
        if getattr(self, 'navigation_manager', None) is not None and hasattr(self, 'next_patient'):
            _bind_global('n', self.next_patient, overwrite=True)
            _bind_global('b', self.prev_patient, overwrite=True)

    def _bind_qshortcuts(self):
        # Qt-level shortcuts to ensure undo/redo work even when dock widget has focus
        self._qt_shortcuts = []
        for seq, handler in [
            ('Ctrl+Z', self.prompt_undo),
            ('Ctrl+Y', self.prompt_redo),
            ('Ctrl+X', self.mask_undo),
            ('Ctrl+U', self.mask_redo),
            ('Ctrl+S', self.save_masks),
            ('Alt+Z', self.mask_undo),
            ('Alt+Y', self.mask_redo),
            ('R', self.enable_add_box),
            ('T', self.enable_edit_boxes),
            ('N', self.next_patient if getattr(self, 'navigation_manager', None) is not None else None),
            ('B', self.prev_patient if getattr(self, 'navigation_manager', None) is not None else None),
        ]:
            if handler:
                sc = QShortcut(QKeySequence(seq), self)
                sc.setContext(Qt.ApplicationShortcut)
                sc.activated.connect(handler)
                self._qt_shortcuts.append(sc)

    def _attach_mode_guard(self, layer, allowed_modes=('select',)):
        def _on_mode_change(event=None):
            try:
                if layer.mode not in allowed_modes:
                    layer.mode = allowed_modes[0]
            except Exception:
                pass
        try:
            layer.events.mode.connect(_on_mode_change)
        except Exception:
            pass

    def _set_button_active(self, btn, active):
        if not btn:
            return
        if active:
            btn.setStyleSheet('background-color: lightgreen; font-weight: bold;')
        else:
            btn.setStyleSheet('')

    def _reset_prompt_buttons(self):
        for b in (self.btn_add_pos, self.btn_add_neg, self.btn_add_box):
            self._set_button_active(b, False)
        self._set_button_active(self.btn_edit_points, False)
        self._set_button_active(self.btn_edit_boxes, False)

    def _set_mask_opacity(self, value):
        try:
            value = float(np.clip(value, 0.0, 1.0))
            self.mask_layer.opacity = value
            if value > 0.0:
                self._last_nonzero_mask_opacity = value
                self.viewer.status = f"Mask opacity: {value:.2f}"
            else:
                self.viewer.status = "Mask opacity: off"
        except Exception:
            pass

    def _push_mask_history(self, snapshot):
        MAX_LEN = 20
        self.mask_history.append(snapshot)
        while len(self.mask_history) > MAX_LEN:
            self.mask_history.popleft()

    def _set_mask_data(self, new_data, record_history=True):
        try:
            if record_history and self._last_mask_state is not None:
                self._push_mask_history(self._last_mask_state.copy())
                self.mask_redo_stack.clear()
            self._updating_layers = True
            self.mask_layer.data = new_data
            self._last_mask_state = self.mask_layer.data.copy()
        finally:
            self._updating_layers = False

    def _bump_mask_opacity(self, delta):
        try:
            new_opacity = float(np.clip(self.mask_layer.opacity + delta, 0.0, 1.0))
            self.mask_layer.opacity = new_opacity
            if new_opacity > 0.0:
                self._last_nonzero_mask_opacity = new_opacity
            self.viewer.status = f"Mask opacity: {new_opacity:.2f}"
        except Exception:
            pass

    def _toggle_mask_opacity(self):
        try:
            current = float(self.mask_layer.opacity)
            if current > 0.0:
                self._last_nonzero_mask_opacity = current
                self.mask_layer.opacity = 0.0
                self.viewer.status = "Mask opacity: off"
            else:
                restore = float(np.clip(getattr(self, '_last_nonzero_mask_opacity', 1.0), 0.0, 1.0)) or 1.0
                self.mask_layer.opacity = restore
                self.viewer.status = f"Mask opacity: {restore:.2f}"
        except Exception:
            pass

    def _setup_manual_box_editing_events(self):
        self._box_edit_timer = None
        @self.box_layer.events.data.connect
        def on_boxes_data_change():
            if not hasattr(self, 'box_layer') or not self.box_layer.editable:
                return
            if getattr(self, '_suppress_box_history', False):
                return
            if getattr(self, '_updating_layers', False):
                return
            if self._box_edit_timer:
                self._box_edit_timer.cancel()
            self._box_edit_timer = threading.Timer(0.3, lambda: self._sync_manual_boxes_with_rectangle_constraint(record_history=True))
            self._box_edit_timer.start()
        # store last known box prompts to detect manual edits even when constraints unchanged
        try:
            self._last_box_prompts_snapshot = list(self.box_prompts)
        except Exception:
            self._last_box_prompts_snapshot = []

    def _sync_manual_boxes_with_rectangle_constraint(self, record_history=False):
        if getattr(self, '_updating_layers', False) or not self.box_layer.editable:
            return
        try:
            if len(self.box_layer.data) == 0:
                return
            self._updating_layers = True
            current_boxes = list(self.box_layer.data)
            updated_boxes = []
            for box in current_boxes:
                if len(box) != 4 or len(box[0]) != 3:
                    updated_boxes.append(box)
                    continue
                frame = box[0][0]
                y_coords = [point[1] for point in box]
                x_coords = [point[2] for point in box]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                if max_x <= min_x:
                    max_x = min_x + 1
                if max_y <= min_y:
                    max_y = min_y + 1
                rect_box = np.array([
                    [frame, min_y, min_x],
                    [frame, min_y, max_x],
                    [frame, max_y, max_x],
                    [frame, max_y, min_x]
                ])
                updated_boxes.append(rect_box)
            changed = not all(np.array_equal(new, old) for new, old in zip(updated_boxes, current_boxes))
            if changed:
                print(f"Applying rectangle constraint to {len(updated_boxes)} manual boxes")
                self.box_layer.data = updated_boxes
        except Exception as e:
            print(f"Error in manual box rectangle constraint: {e}")
        finally:
            self._updating_layers = False
        # sync prompts and record history if requested
        self._sync_boxes_from_layer(record_history=record_history)

    def _sync_boxes_from_layer(self, record_history=False):
        if getattr(self, '_updating_layers', False) or not self.box_layer.editable:
            return
        old_boxes = list(self.box_prompts)
        self.box_prompts.clear()
        if len(self.box_layer.data) > 0:
            for corners in self.box_layer.data:
                if len(corners) >= 4:
                    frame_idx = int(corners[0][0])
                    y1, x1 = int(corners[0][1]), int(corners[0][2])
                    y2, x2 = int(corners[2][1]), int(corners[2][2])
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    reuse_obj_id = None
                    min_distance = float('inf')
                    for old_t, old_obj_id, old_x1, old_y1, old_x2, old_y2 in old_boxes:
                        if old_t != frame_idx:
                            continue
                        overlaps_x = not (x2 < old_x1 or x1 > old_x2)
                        overlaps_y = not (y2 < old_y1 or y1 > old_y2)
                        if not (overlaps_x and overlaps_y):
                            continue
                        old_cx, old_cy = (old_x1 + old_x2) / 2, (old_y1 + old_y2) / 2
                        new_cx, new_cy = (x1 + x2) / 2, (y1 + y2) / 2
                        distance = ((old_cx - new_cx) ** 2 + (old_cy - new_cy) ** 2) ** 0.5
                        # Only reuse when ids match; otherwise respect the user's current selection
                        if distance < min_distance and old_obj_id == self.current_obj_id:
                            min_distance = distance
                            reuse_obj_id = old_obj_id
                    obj_id = reuse_obj_id if reuse_obj_id is not None else self.current_obj_id
                    self.box_prompts.append((frame_idx, obj_id, x1, y1, x2, y2))
                    print(f"  Frame {frame_idx}, ObjID {obj_id}: [{x1}, {y1}, {x2}, {y2}]")
        if record_history and old_boxes != list(self.box_prompts):
            # store previous state so undo restores it
            self.prompt_history.append(('box_state', old_boxes))
            self.redo_history.clear()
        print(f"Synced {len(self.box_prompts)} boxes")

    def _sync_points_from_layer(self):
        if getattr(self, '_updating_layers', False) or not self.user_pts_layer.editable:
            return
        old_pos = self.pos_points.copy()
        old_neg = self.neg_points.copy()
        pos_point_to_objid = {(t, y, x): oid for t, oid, y, x in old_pos}
        neg_point_to_objid = {(t, y, x): oid for t, oid, y, x in old_neg}
        self.pos_points.clear()
        self.neg_points.clear()
        if len(self.user_pts_layer.data) > 0:
            for i, (t, y, x) in enumerate(self.user_pts_layer.data):
                t, y, x = int(t), int(y), int(x)
                color = self.user_pts_layer.face_color[i] if i < len(self.user_pts_layer.face_color) else 'green'
                is_positive = True
                if isinstance(color, str):
                    is_positive = (color.lower() == 'green')
                elif isinstance(color, (list, tuple, np.ndarray)):
                    color_arr = np.array(color, dtype=float)
                    if color_arr.max() > 1.0:
                        color_arr = color_arr / 255.0
                    if len(color_arr) >= 3:
                        is_green = (color_arr[1] > color_arr[0] and color_arr[1] > color_arr[2] and color_arr[1] > 0.5)
                        is_red = (color_arr[0] > color_arr[1] and color_arr[0] > color_arr[2] and color_arr[0] > 0.5)
                        if is_green:
                            is_positive = True
                        elif is_red:
                            is_positive = False
                original_obj_id = None
                if is_positive and (t, y, x) in pos_point_to_objid:
                    original_obj_id = pos_point_to_objid[(t, y, x)]
                elif not is_positive and (t, y, x) in neg_point_to_objid:
                    original_obj_id = neg_point_to_objid[(t, y, x)]
                obj_id_to_use = original_obj_id if original_obj_id is not None else self.current_obj_id
                if is_positive:
                    self.pos_points.append((t, obj_id_to_use, y, x))
                else:
                    self.neg_points.append((t, obj_id_to_use, y, x))
        print(f"Synced {len(self.pos_points)} positive and {len(self.neg_points)} negative points")

    def _create_rectangle_corners(self, frame, x1, y1, x2, y2):
        return np.array([
            [frame, y1, x1],
            [frame, y1, x2],
            [frame, y2, x2],
            [frame, y2, x1]
        ], dtype=np.float64)

    def _record_prompt_event(self, kind, frame_idx, obj_id):
        if not self.metrics or not self.metrics.is_active():
            return
        self.metrics.inc_counter(f'{kind}_count', 1)
        self.metrics.add_event('prompt_added', kind=kind, frame=int(frame_idx), obj_id=int(obj_id))
        if self.first_prompt_ts is None:
            self.first_prompt_ts = time.time()
            self.metrics.set_info('time_to_first_prompt_sec', round(self.first_prompt_ts - self.session_started_at, 3))

    def _build_controls(self):
        layout = QVBoxLayout()
        current_frame_hl = QHBoxLayout()
        current_frame_hl.addWidget(QLabel('Current Frame:'))
        self.current_frame_label = QLabel(str(self.frame_idx))
        self.current_frame_label.setStyleSheet("font-weight: bold; color: blue;")
        current_frame_hl.addWidget(self.current_frame_label)
        layout.addLayout(current_frame_hl)
        frm_hl = QHBoxLayout()
        frm_hl.addWidget(QLabel('Manual Frame:'))
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, self.n_frames-1)
        self.frame_spin.valueChanged.connect(self.on_frame_change)
        frm_hl.addWidget(self.frame_spin)
        layout.addLayout(frm_hl)
        oid_hl = QHBoxLayout()
        oid_hl.addWidget(QLabel('Object id:'))
        self.oid_spin = QSpinBox()
        self.oid_spin.setRange(1, 100)
        self.oid_spin.setValue(self.current_obj_id)
        self.oid_spin.valueChanged.connect(self.on_obj_change)
        oid_hl.addWidget(self.oid_spin)
        layout.addLayout(oid_hl)
        btns = [
            ('Add + Point', self.enable_add_positive, 'btn_add_pos'),
            ('Add - Point', self.enable_add_negative, 'btn_add_neg'),
            ('Add Box',     self.enable_add_box, 'btn_add_box'),
            ('Manual Edit', self.toggle_manual_annotation, 'manual_edit_button'),
            ('Edit Points', self.enable_edit_points, 'btn_edit_points'),
            ('Edit Boxes',  self.enable_edit_boxes, 'btn_edit_boxes'),
            ('Clear All',   self.clear_all_prompts, None),
            ('Propagate',   self.propagate_prompt, None),
            ('3D Volume Render', lambda: render_manual_volume(self), None),
            ('Prompt Undo', self.prompt_undo, None),
            ('Prompt Redo', self.prompt_redo, None),
            ('Mask Undo',   self.mask_undo, None),
            ('Mask Redo',   self.mask_redo, None),
            ('Save Masks',  lambda: save_masks_manual(self), None)
        ]
        for label, func, attr in btns:
            b = QPushButton(label)
            b.clicked.connect(func)
            layout.addWidget(b)
            if attr:
                setattr(self, attr, b)
        self.toggle_id_btn = QPushButton("Hide Object IDs" if self.show_object_ids else "Show Object IDs")
        self.toggle_id_btn.clicked.connect(self.toggle_object_id_visibility)
        self.toggle_id_btn.setStyleSheet("background-color: lightgreen; font-weight: bold;")
        layout.addWidget(self.toggle_id_btn)
        self.setLayout(layout)

    def clear_all_prompts(self):
        self.pos_points.clear()
        self.neg_points.clear()
        self.box_prompts.clear()
        self.prompt_history.clear()
        self.redo_history.clear()
        self._set_mask_data(np.zeros((self.n_frames,) + self.imgs.shape[2:], dtype=np.uint8))
        self.update_layers()
        print("All prompts and masks cleared")
        if self.metrics and self.metrics.is_active():
            self.metrics.add_event('prompts_cleared')

    def on_frame_change(self, val):
        self.frame_idx = val
        current_step = list(self.viewer.dims.current_step)
        current_step[0] = val
        self.viewer.dims.current_step = current_step
        self.current_frame_label.setText(str(val))

    def on_obj_change(self, val):
        self.current_obj_id = val
        if self.manual_edit_enabled:
            self.mask_layer.selected_label = val

    def cancel_prompt_mode(self, viewer=None):
        self.img_layer.mouse_drag_callbacks.clear()
        self.mask_layer.mouse_drag_callbacks.clear()
        try:
            self.box_layer.mouse_drag_callbacks.clear()
        except Exception:
            pass
        if self.active_tool == 'edit_pts':
            self.user_pts_layer.editable = False
        if self.active_tool == 'edit_boxes':
            self.box_layer.editable = False
        if not self.manual_edit_enabled:
            self.user_pts_layer.editable = False
            self.box_layer.editable = False
            self.mask_layer.editable = False
        self._reset_prompt_buttons()
        self.active_tool = None

    def _activate_tool(self, name):
        # Toggle off if the same tool is already active
        if self.manual_edit_enabled and name != 'manual_edit':
            print("Manual Edit is active. Turn it off before using other tools.")
            return False
        if self.active_tool == name:
            self.cancel_prompt_mode()
            return False
        self.cancel_prompt_mode()
        self.active_tool = name
        return True

    def enable_add_positive(self):
        if self.manual_edit_enabled:
            print("Manual Edit is active; point add is disabled.")
            return
        if not self._activate_tool('add_pos'):
            return
        self._select_layer(self.img_layer)
        self._set_button_active(self.btn_add_pos, True)
        self.add_mode = 'pos'
        def cb(layer, event):
            if event.type != 'mouse_press': return
            t = int(self.viewer.dims.current_step[0])
            y, x = map(int, event.position[1:])
            self.pos_points.append((t, self.current_obj_id, y, x))
            self.prompt_history.append(('pos', t, self.current_obj_id, y, x))
            self.redo_history.clear()
            self.update_layers()
            self._record_prompt_event('pos_point', t, self.current_obj_id)
            print(f"Added positive point at frame {t}, position ({x}, {y}) - should be GREEN")
        self.img_layer.mouse_drag_callbacks.append(cb)

    def enable_add_negative(self):
        if self.manual_edit_enabled:
            print("Manual Edit is active; point add is disabled.")
            return
        if not self._activate_tool('add_neg'):
            return
        self._select_layer(self.img_layer)
        self._set_button_active(self.btn_add_neg, True)
        self.add_mode = 'neg'
        def cb(layer, event):
            if event.type != 'mouse_press': return
            t = int(self.viewer.dims.current_step[0])
            y, x = map(int, event.position[1:])
            self.neg_points.append((t, self.current_obj_id, y, x))
            self.prompt_history.append(('neg', t, self.current_obj_id, y, x))
            self.redo_history.clear()
            self.update_layers()
            self._record_prompt_event('neg_point', t, self.current_obj_id)
            print(f"Added negative point at frame {t}, position ({x}, {y}) - should be RED")
        self.img_layer.mouse_drag_callbacks.append(cb)

    def enable_add_box(self):
        if self.manual_edit_enabled:
            print("Manual Edit is active; box add is disabled.")
            return
        if not self._activate_tool('add_box'):
            return
        self._focus_mask_and_box_layers()
        self._select_layer(self.box_layer)
        self.box_layer.editable = True
        try:
            self.box_layer.mode = 'add_rectangle'
        except Exception:
            pass
        self._set_button_active(self.btn_add_box, True)
        self.add_mode = 'box'

        temp_state = {
            'active': False,
            't': None,
            'x0': None,
            'y0': None,
            'rect_idx': None,
        }

        def cb(layer, event):
            etype = getattr(event, 'type', None)
            if etype == 'mouse_press':
                temp_state['t'] = int(self.viewer.dims.current_step[0])
                y0, x0 = map(int, event.position[1:])
                temp_state['x0'], temp_state['y0'] = x0, y0
                temp_state['active'] = True
                temp_state['rect_idx'] = None
                # clear redo stack on new user action
                self.redo_history.clear()
            elif etype == 'mouse_move' and temp_state['active']:
                try:
                    self.box_layer.mode = 'add_rectangle'
                except Exception:
                    pass
                y1, x1 = map(int, event.position[1:])
                x1, x2 = sorted([temp_state['x0'], x1])
                y1, y2 = sorted([temp_state['y0'], y1])
                rect = np.array([
                    [temp_state['t'], y1, x1],
                    [temp_state['t'], y1, x2],
                    [temp_state['t'], y2, x2],
                    [temp_state['t'], y2, x1],
                ], dtype=float)
                try:
                    self._updating_layers = True
                    if temp_state['rect_idx'] is None:
                        self.box_layer.add_rectangles([rect])
                        temp_state['rect_idx'] = len(self.box_layer.data) - 1
                    else:
                        data = list(self.box_layer.data)
                        data[temp_state['rect_idx']] = rect
                        self.box_layer.data = data
                finally:
                    self._updating_layers = False
            elif etype == 'mouse_release' and temp_state['active']:
                try:
                    self.box_layer.mode = 'select'
                except Exception:
                    pass
                y1, x1 = map(int, event.position[1:])
                x1, x2 = sorted([temp_state['x0'], x1])
                y1, y2 = sorted([temp_state['y0'], y1])
                t = temp_state['t']
                self.box_prompts.append((t, self.current_obj_id, x1, y1, x2, y2))
                self.prompt_history.append(('box', t, self.current_obj_id, x1, y1, x2, y2))
                self.update_layers()
                self._record_prompt_event('box', t, self.current_obj_id)
                print(f"Added box at frame {t}, corners ({x1}, {y1}) to ({x2}, {y2})")
                temp_state['active'] = False
                temp_state['rect_idx'] = None
                temp_state['t'] = None
                temp_state['x0'] = None
                temp_state['y0'] = None

        self.box_layer.mouse_drag_callbacks.append(cb)

    def toggle_manual_annotation(self):
        # Enable napari's native painting/rectangle drawing without needing Add Box
        self.img_layer.mouse_drag_callbacks.clear()
        self.mask_layer.mouse_drag_callbacks.clear()
        self.add_mode = None
        self._reset_prompt_buttons()
        self.active_tool = None
        self.manual_edit_enabled = not self.manual_edit_enabled
        if self.manual_edit_enabled:
            self.mask_layer.editable = True
            self.mask_layer.selected_label = self.current_obj_id
            self.box_layer.editable = True
            self.box_layer.mode = 'add_rectangle'
            self._select_layer(self.mask_layer)
            self._set_button_active(self.manual_edit_button, True)
            self.viewer.status = "Manual Edit ON"
            print("Manual annotation enabled: paint on 'mask, box layer' or draw rectangles in 'User Boxes correction layer'.")
            if self.metrics and self.metrics.is_active():
                self.manual_edit_started_at = time.time()
                try:
                    # Ensure the stroke callback is attached only once
                    if self._manual_edit_stroke_callback not in self.mask_layer.mouse_drag_callbacks:
                        self.mask_layer.mouse_drag_callbacks.append(self._manual_edit_stroke_callback)
                except Exception:
                    pass
            try:
                current = self.mask_layer.data.copy()
                self._push_mask_history(current)
                self.mask_redo_stack.clear()
                self._last_mask_state = current
            except Exception:
                pass
            try:
                self._last_mask_state = self.mask_layer.data.copy()
            except Exception:
                pass
        else:
            self.mask_layer.editable = False
            self.box_layer.editable = False
            self._set_button_active(self.manual_edit_button, False)
            self.viewer.status = "Manual Edit OFF"
            print("Manual annotation disabled.")
            if self.metrics and self.metrics.is_active() and self.manual_edit_started_at:
                interval = time.time() - self.manual_edit_started_at
                self.manual_edit_total_sec += interval
                self.metrics.record_stage('manual_edit_interval', self.manual_edit_started_at, time.time(), duration_sec=round(interval, 4))
                self.manual_edit_started_at = None
        if self.metrics and self.metrics.is_active():
            self.metrics.inc_counter('manual_edit_toggles', 1)
            self.metrics.add_event('manual_edit_toggled', enabled=self.manual_edit_enabled)

    def ensure_manual_edit_on(self):
        """Force-enable manual edit and focus the mask/box layer, regardless of current layer."""
        self._select_layer(self.mask_layer)
        if not self.manual_edit_enabled:
            self.toggle_manual_annotation()
        else:
            # Refresh selection/mode even when already on
            try:
                self.mask_layer.editable = True
                self.mask_layer.mode = 'paint'
                self.mask_layer.selected_label = self.current_obj_id
            except Exception:
                pass

    def enable_edit_points(self):
        if self.manual_edit_enabled:
            print("Manual Edit is active; point edit is disabled.")
            return
        if not self._activate_tool('edit_pts'):
            # toggled off
            self.user_pts_layer.editable = False
            try:
                self.user_pts_layer.mode = 'pan_zoom'
            except Exception:
                pass
            return
        self.user_pts_layer.editable = True
        self._select_layer(self.user_pts_layer)
        try:
            self.user_pts_layer.mode = 'select'
        except Exception:
            pass
        self._set_button_active(self.btn_edit_points, True)
        print("Points editing enabled - you can move/delete points")

    def enable_edit_boxes(self):
        if self.manual_edit_enabled:
            print("Manual Edit is active; box edit is disabled.")
            return
        if not self._activate_tool('edit_boxes'):
            # toggled off
            self.box_layer.editable = False
            try:
                self.box_layer.mode = 'pan_zoom'
            except Exception:
                pass
            return
        self.box_layer.visible = True
        self.box_layer.editable = True
        self._select_layer(self.box_layer)
        try:
            self.viewer.layers.selection.active = self.box_layer
        except Exception:
            pass
        try:
            self.box_layer.mode = 'select'
        except Exception:
            pass
        self._set_button_active(self.btn_edit_boxes, True)
        self.viewer.status = "Edit Boxes ON"
        print("Boxes editing enabled - rectangles will maintain shape during editing")

    def update_layers(self):
        # Prevent box edits from recording history while we redraw from prompt state
        self._suppress_box_history = True
        self._updating_layers = True
        pts, colors = [], []
        for t, oid, y, x in self.pos_points:
            pts.append([t, y, x])
            colors.append('green')
        for t, oid, y, x in self.neg_points:
            pts.append([t, y, x])
            colors.append('red')
        if pts:
            self.user_pts_layer.data = np.array(pts, dtype=np.float64)
            self.user_pts_layer.face_color = colors
        else:
            self.user_pts_layer.data = np.empty((0, 3), dtype=np.float64)
            # Use zero-length RGBA arrays to avoid napari warnings about empty/illegal colors
            empty_rgba = np.zeros((0, 4), dtype=float)
            self.user_pts_layer.face_color = empty_rgba
            self.user_pts_layer.edge_color = empty_rgba
        shapes = []
        for (t, oid, x1, y1, x2, y2) in self.box_prompts:
            corners = self._create_rectangle_corners(t, x1, y1, x2, y2)
            shapes.append(corners)
        if shapes:
            self.box_layer.data = np.array(shapes, dtype=np.float64)
            # set colors matching number of boxes to avoid napari color warnings
            n_shapes = len(shapes)
            self.box_layer.edge_color = ['red'] * n_shapes
            self.box_layer.face_color = [[0, 0, 0, 0]] * n_shapes
            try:
                self.box_layer.edge_width = 2
            except Exception:
                pass
        else:
            self.box_layer.data = np.empty((0, 4, 3), dtype=np.float64)
            empty_rgba = np.zeros((0, 4), dtype=float)
            self.box_layer.edge_color = empty_rgba
            self.box_layer.face_color = empty_rgba
        self._update_object_id_text()
        self._updating_layers = False
        self._suppress_box_history = False
        print(f"Updated layers: {len(self.pos_points)} positive, {len(self.neg_points)} negative points, {len(self.box_prompts)} boxes")

    def _update_object_id_text(self):
        if not self.show_object_ids:
            if self.text_layer and self.text_supported:
                self.text_layer.data = np.empty((0, 3), dtype=np.float64)
            return
        text_positions = []
        text_strings = []
        for (t, oid, x1, y1, x2, y2) in self.box_prompts:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            text_positions.append([t, center_y, center_x])
            text_strings.append(str(oid))
        for (t, oid, y, x) in self.pos_points:
            text_positions.append([t, y - 10, x])
            text_strings.append(f"{oid}+")
        for (t, oid, y, x) in self.neg_points:
            text_positions.append([t, y - 10, x])
            text_strings.append(f"{oid}-")
        if self.text_layer and self.text_supported:
            if text_positions:
                self.text_layer.data = np.array(text_positions, dtype=np.float64)
                try:
                    self.text_layer.text = {
                        'string': text_strings,
                        'size': 14,
                        'color': 'yellow',
                        'anchor': 'center'
                    }
                except Exception as e:
                    print(f"Text rendering failed: {e}")
            else:
                self.text_layer.data = np.empty((0, 3), dtype=np.float64)
        else:
            if text_positions:
                print("=== Object ID Information ===")
                for s in text_strings:
                    print(f"  {s}")

    def toggle_object_id_visibility(self):
        self.show_object_ids = not self.show_object_ids
        self._update_object_id_text()
        button_text = "Hide Object IDs" if self.show_object_ids else "Show Object IDs"
        if hasattr(self, 'toggle_id_btn'):
            self.toggle_id_btn.setText(button_text)
        mode_info = "visual" if (self.text_layer and self.text_supported) else "console"
        print(f"Object ID visibility: {'ON' if self.show_object_ids else 'OFF'} (mode: {mode_info})")

    def prompt_undo(self):
        if not self.prompt_history:
            return
        act = self.prompt_history.pop()
        self.redo_history.append(act)
        self.apply_prompt_history()
        if self.metrics and self.metrics.is_active():
            self.metrics.inc_counter('undo_actions', 1)
            self.metrics.add_event('undo', action=act[0])

    def prompt_redo(self):
        if not self.redo_history:
            return
        act = self.redo_history.pop()
        self.prompt_history.append(act)
        self.apply_prompt_history()
        if self.metrics and self.metrics.is_active():
            self.metrics.inc_counter('redo_actions', 1)
            self.metrics.add_event('redo', action=act[0])

    def apply_prompt_history(self):
        self._suppress_box_history = True
        try:
            self.pos_points.clear(); self.neg_points.clear(); self.box_prompts.clear()
            for it in self.prompt_history:
                if not it:
                    continue
                cmd = it[0]
                if cmd == 'pos' and len(it) >= 4:
                    _, t, oid, y, x = it
                    self.pos_points.append((t, oid, y, x))
                elif cmd == 'neg' and len(it) >= 4:
                    _, t, oid, y, x = it
                    self.neg_points.append((t, oid, y, x))
                elif cmd == 'box' and len(it) >= 6:
                    _, t, oid, x1, y1, x2, y2 = it
                    self.box_prompts.append((t, oid, x1, y1, x2, y2))
                elif cmd == 'box_state':
                    snapshot = it[1] if len(it) > 1 else []
                    self.box_prompts = list(snapshot)
            self.update_layers()
        finally:
            self._suppress_box_history = False

    def propagate_prompt(self):
        print("Synchronizing layer data before propagate...")
        prop_start = time.time()
        if hasattr(self, 'box_layer') and len(self.box_layer.data) > 0:
            self._sync_boxes_from_layer()
            print(f"Synced boxes: {self.box_prompts}")
        if hasattr(self, 'user_pts_layer') and len(self.user_pts_layer.data) > 0:
            self._sync_points_from_layer()
            print(f"Synced points: pos={len(self.pos_points)}, neg={len(self.neg_points)}")
        idxs = [t for (t,_,_,_,_,_) in self.box_prompts] + [t for (t,_,_,_) in self.pos_points] + [t for (t,_,_,_) in self.neg_points]
        if not idxs:
            QMessageBox.warning(self, 'No Prompts', 'Please add points or boxes.')
            return
        start, end = min(idxs), max(idxs)
        print(f"Propagating from frame {start} to {end}...")
        self._set_mask_data(np.zeros((self.n_frames,) + self.imgs.shape[2:], dtype=np.uint8))
        sub = self.imgs[start:end+1].to(self.device)
        with torch.no_grad():
            state = self.net.val_init_state(imgs_tensor=sub)
            box_count = 0
            pos_used = 0
            neg_used = 0
            for (t, oid, x1, y1, x2, y2) in self.box_prompts:
                if start <= t <= end:
                    self.net.train_add_new_bbox(
                        state,
                        t - start,
                        oid,
                        torch.tensor([x1, y1, x2, y2], device=self.device),
                        clear_old_points=False
                    )
                    box_count += 1
            pm, nm = defaultdict(list), defaultdict(list)
            for (t, oid, y, x) in self.pos_points:
                if start <= t <= end:
                    pos_used += 1
                    pm[(t-start, oid)].append((x, y))
            for (t, oid, y, x) in self.neg_points:
                if start <= t <= end:
                    neg_used += 1
                    nm[(t-start, oid)].append((x, y))
            for (lt, oid) in set(pm) | set(nm):
                pts, labs = [], []
                for p in pm.get((lt, oid), []):
                    pts.append(p); labs.append(1)
                for p in nm.get((lt, oid), []):
                    pts.append(p); labs.append(0)
                if pts:
                    self.net.train_add_new_points(
                        state,
                        lt,
                        oid,
                        torch.tensor(pts, device=self.device),
                        torch.tensor(labs, device=self.device),
                        clear_old_points=False
                    )
            result = {}
            try:
                for lt, oids, logits in self.net.propagate_in_video(state, start_frame_idx=0):
                    gi = start + lt
                    if len(logits) > 0 and len(oids) > 0:
                        frame_mask = np.zeros(logits[0].shape, dtype=np.uint8)
                        for oid, logit in zip(oids, logits):
                            obj_mask = (logit.cpu().numpy() > 0.5).astype(np.uint8)
                            frame_mask[obj_mask > 0] = oid
                        result[gi] = frame_mask
                        print(f"  Generated mask for frame {gi} with {len(oids)} objects: {oids}")
            except Exception as e:
                print(f"Propagation error: {e}")
                QMessageBox.critical(self, 'Propagation Error', f'Propagation failed: {e}')
                if self.metrics and self.metrics.is_active():
                    self.metrics.add_event('propagation_error', message=str(e))
                return
            finally:
                self.net.reset_state(state)
                del state
            new_mask = np.zeros((self.n_frames,) + self.imgs.shape[2:], dtype=np.uint8)
            for i, m in result.items():
                new_mask[i] = m
            self._set_mask_data(new_mask)
            self._update_object_id_text()
            print(f"Propagation completed. Generated masks for {len(result)} frames")
            QMessageBox.information(self, 'Done', f'Propagated {start}~{end}\nGenerated {len(result)} masks')
            if self.metrics and self.metrics.is_active():
                end_ts = time.time()
                slice_count = int(end - start + 1)
                self.metrics.record_stage(
                    'propagation_manual',
                    prop_start,
                    end_ts,
                    start_frame=int(start),
                    end_frame=int(end),
                    slice_count=slice_count,
                    frames_with_masks=len(result),
                    boxes_used=box_count,
                    pos_points_used=pos_used,
                    neg_points_used=neg_used,
                    avg_sec_per_slice=round((end_ts - prop_start) / slice_count, 4) if slice_count > 0 else None,
                )
                self.metrics.add_event('propagation_completed', frames_with_masks=len(result))

    def save_masks(self):
        save_masks_manual(self)

    def render_3d_volume(self):
        render_manual_volume(self)

    def mask_undo(self):
        if not self.mask_history:
            return
        try:
            current = self.mask_layer.data.copy()
            prev = self.mask_history.pop()
            self.mask_redo_stack.append(current)
            self._updating_layers = True
            self.mask_layer.data = prev
            self._last_mask_state = self.mask_layer.data.copy()
        finally:
            self._updating_layers = False

    def mask_redo(self):
        if not self.mask_redo_stack:
            return
        try:
            current = self.mask_layer.data.copy()
            nxt = self.mask_redo_stack.pop()
            self._push_mask_history(current)
            self._updating_layers = True
            self.mask_layer.data = nxt
            self._last_mask_state = self.mask_layer.data.copy()
        finally:
            self._updating_layers = False
