import os
import numpy as np
import torch
import napari
import time
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QPushButton,
    QMessageBox,
    QShortcut,
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt
from collections import deque, defaultdict
import traceback
import threading
from scipy.ndimage import center_of_mass

from gui.prompts import make_initial_label_stack, normalize_box_prompts, normalize_point_prompts
from gui.rendering import render_manual_volume
from gui.io import save_masks_auto
from gui.metrics import UsageMetricsRecorder


class MedSAM2NapariGUI(QWidget):
    def __init__(
        self,
        imgs,                 # [T,3,H,W]
        video_segments,       # {frame_idx: {obj_id: mask_logits}}
        net,
        device,
        patient_id,
        box_prompts,          # cls-det: {frame_idx: {obj_id: [x1,y1,x2,y2]}}
        point_prompts,        # seg: {frame_idx: {obj_id: {"bboxes": [...], "points": [...]}}}
        start_idx,
        end_idx,
        meta,                 # spacing, origin, direction
        metrics: UsageMetricsRecorder = None
    ):
        super().__init__()
        # State
        # Normalize image tensor to shape [T, C, H, W]
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(1)
        elif imgs.ndim != 4:
            raise ValueError(f"Expected imgs with 3 or 4 dims, got {imgs.shape}")
        self.imgs = imgs
        self.video_segments = video_segments
        self.net = net
        self.device = device
        self.patient_id = patient_id
        self.meta = meta
        self.metrics = metrics
        self.session_started_at = time.time()
        self.first_user_prompt_ts = None
        self.manual_edit_started_at = None
        self.manual_edit_total_sec = 0.0
        self.start_idx = start_idx
        self.end_idx = end_idx
        
        # Normalize prompt structures
        self.box_prompts = normalize_box_prompts(box_prompts)
        self.point_prompts = normalize_point_prompts(point_prompts)
        
        # History for undo/redo
        self.prompt_history = deque()
        self.redo_history = deque()
        
        # GUI state
        self.n_frames = imgs.shape[0]
        self.frame_idx = 0
        self.obj_ids = sorted({oid for seg in video_segments.values() for oid in seg.keys()})
        self.current_obj_id = self.obj_ids[0] if self.obj_ids else 1
        
        # Text display toggle
        self.text_visible = True

        # Napari viewer
        self.viewer = napari.Viewer(title=self._get_patient_display_name())
        self.viewer.bind_key('Escape', self.cancel_prompt_mode)
        try:
            # Track last non-zero opacity for toggle
            self._last_nonzero_mask_opacity = float(1.0)
        except Exception:
            self._last_nonzero_mask_opacity = 1.0

        # Layers
        c = self.imgs.shape[1]
        if c == 1:
            img_data = self.imgs.squeeze(1).cpu().numpy()
            rgb_flag = False
        else:
            img_data = self.imgs[:, :3].permute(0, 2, 3, 1).cpu().numpy()
            rgb_flag = True
        self.img_layer = self.viewer.add_image(
            img_data, name='Image', rgb=rgb_flag
        )
        
        # Initial mask visualization
        self.mask_layer = self.viewer.add_labels(
            make_initial_label_stack(
                video_segments=self.video_segments,
                obj_ids=self.obj_ids,
                n_frames=self.n_frames,
                spatial_shape=self.imgs.shape[2:],
            ),
            name='Mask'
        )
        
        # Add Object ID text layer
        self._init_text_layer()
        
        # Skip auto prompt layers to avoid confusion; all prompts go through unified layers
        self.auto_pts_layer = None
        self.auto_box_layer = None
        
        # User prompts layers
        self.user_pts_layer = self.viewer.add_points(
            np.empty((0,3)), name='Points', face_color='green', size=5
        )
        self.user_box_layer = self.viewer.add_shapes(
            np.empty((0,4,3)), name='Boxes', shape_type='rectangle',
            edge_color='red', face_color=[0,0,0,0], edge_width=2, ndim=3
        )
        self._attach_mode_guard(self.user_pts_layer, allowed_modes=('select', 'pan_zoom'))
        self._attach_mode_guard(self.user_box_layer, allowed_modes=('select', 'pan_zoom', 'add_rectangle'))
        self.user_pts_layer.editable = False
        self.user_box_layer.editable = False
        self.manual_edit_enabled = False  # allow freehand label/box editing on demand
        self.active_tool = None  # track current prompt/edit tool for toggling
        self.mask_history = deque()
        self.mask_redo_stack = deque()
        self._manual_stroke_active = False
        self._stroke_start_state = None
        try:
            self._last_mask_state = self.mask_layer.data.copy()
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

        # Controls
        self._build_controls()
        self.viewer.window.add_dock_widget(self, area='right')
        
        # Detect viewer's current step(frame) changes for automatic updates
        self._setup_viewer_callbacks()
        
        # Setup layer event callbacks
        self._setup_layer_callbacks()
        
        self.update_prompt_layers()
        auto_box_count = sum(len(objs) for objs in self.box_prompts.values())
        auto_point_count = sum(len(pts) for objs in self.point_prompts.values() for pts in objs.values())
        self._bind_shortcuts()
        self._bind_qshortcuts()
        if self.metrics and self.metrics.is_active():
            self.metrics.add_event('auto_gui_initialized', n_frames=self.n_frames, patient_id=str(self.patient_id))
            self.metrics.set_info('n_frames', int(self.n_frames))
            self.metrics.set_info('auto_init_boxes', int(auto_box_count))
            self.metrics.set_info('auto_init_points', int(auto_point_count))

    def _setup_viewer_callbacks(self):
        """Detect Napari viewer step changes to automatically update frame_idx"""
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
        """Setup layer event callbacks"""
        @self.user_pts_layer.events.data.connect
        def on_user_points_data_change():
            if not hasattr(self, 'user_pts_layer') or not self.user_pts_layer.editable:
                return
            if getattr(self, '_updating_layers', False):
                return
            self._sync_user_points_from_layer()

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
        
        self._setup_box_editing_events()

    def _setup_box_editing_events(self):
        """Setup box editing events - debounced data change detection"""
        self._user_box_edit_timer = None
            
        @self.user_box_layer.events.data.connect
        def on_user_boxes_data_change():
            if not hasattr(self, 'user_box_layer') or not self.user_box_layer.editable:
                return
            if getattr(self, '_updating_user_boxes', False):
                return
            if self._user_box_edit_timer:
                self._user_box_edit_timer.cancel()
            self._user_box_edit_timer = threading.Timer(0.3, self._sync_user_boxes_with_rectangle_constraint)
            self._user_box_edit_timer.start()

    def _record_prompt_event(self, kind, frame_idx, obj_id):
        if not self.metrics or not self.metrics.is_active():
            return
        self.metrics.inc_counter(f'{kind}_count', 1)
        self.metrics.add_event('prompt_added', kind=kind, frame=int(frame_idx), obj_id=int(obj_id))
        if self.first_user_prompt_ts is None:
            self.first_user_prompt_ts = time.time()
            self.metrics.set_info('time_to_first_user_prompt_sec', round(self.first_user_prompt_ts - self.session_started_at, 3))

    def _sync_user_boxes_with_rectangle_constraint(self):
        """Sync user boxes with rectangle constraint - support both shrinking/expanding"""
        if getattr(self, '_updating_user_boxes', False):
            return
        
        self._updating_user_boxes = True
        try:
            current_data = list(self.user_box_layer.data)
            if not current_data:
                return
                
            corrected_data = []
            has_changes = False
            
            for shape in current_data:
                if len(shape) >= 4:
                    frame = shape[0][0]
                    y_coords = [pt[1] for pt in shape]  
                    x_coords = [pt[2] for pt in shape]
                    y_min, y_max = min(y_coords), max(y_coords)
                    x_min, x_max = min(x_coords), max(x_coords)
                    if y_max <= y_min:
                        y_max = y_min + 1
                    if x_max <= x_min:
                        x_max = x_min + 1
                    rect_shape = np.array([
                        [frame, y_min, x_min],
                        [frame, y_min, x_max],
                        [frame, y_max, x_max],
                        [frame, y_max, x_min]
                    ], dtype=float)
                    if not np.allclose(rect_shape, shape, atol=1e-6):
                        has_changes = True
                    corrected_data.append(rect_shape)
                else:
                    corrected_data.append(shape)
            if has_changes:
                print(f"Applying rectangle constraint to {len(corrected_data)} user boxes")
                self.user_box_layer.data = corrected_data
            self._sync_user_boxes_from_layer()
        except Exception as e:
            print(f"Error in user box rectangle constraint: {e}")
        finally:
            self._updating_user_boxes = False

    def _color_to_label(self, color):
        """Helper function to convert color to label"""
        if isinstance(color, str):
            return 1 if color.lower() == 'yellow' else 0
        elif isinstance(color, (list, tuple, np.ndarray)):
            color_arr = np.array(color, dtype=float)
            if color_arr.max() > 1.0:
                color_arr = color_arr / 255.0
            if len(color_arr) >= 3:
                is_yellow = (color_arr[0] > 0.5 and color_arr[1] > 0.5 and color_arr[2] < 0.5)
                is_orange = (color_arr[0] > 0.5 and 0.2 < color_arr[1] < 0.8 and color_arr[2] < 0.5)
                if is_yellow:
                    return 1
                if is_orange:
                    return 0
        return 1

    def _sync_user_points_from_layer(self):
        if getattr(self, '_updating_layers', False) or not self.user_pts_layer.editable:
            return
        try:
            obj_ids_prop = None
            try:
                obj_ids_prop = self.user_pts_layer.properties.get('obj_id', None)
            except Exception:
                obj_ids_prop = None
            new_prompts = defaultdict(lambda: defaultdict(list))
            for i, (t, y, x) in enumerate(self.user_pts_layer.data):
                t, y, x = int(t), int(y), int(x)
                obj_id = int(obj_ids_prop[i]) if obj_ids_prop is not None and len(obj_ids_prop) > i else int(self.current_obj_id)
                color = self.user_pts_layer.face_color[i] if i < len(self.user_pts_layer.face_color) else 'green'
                if isinstance(color, str):
                    label = 1 if color.lower() == 'green' else 0
                elif isinstance(color, (list, tuple, np.ndarray)):
                    color_arr = np.array(color, dtype=float)
                    if color_arr.max() > 1.0:
                        color_arr = color_arr / 255.0
                    if len(color_arr) >= 3:
                        is_green = (color_arr[1] > color_arr[0] and color_arr[1] > color_arr[2] and color_arr[1] > 0.5)
                        label = 1 if is_green else 0
                    else:
                        label = 1
                else:
                    label = 1
                new_prompts[t][obj_id].append((x, y, label))
            self.point_prompts = {t: dict(objs) for t, objs in new_prompts.items()}
        except Exception as e:
            print(f"Error syncing user points: {e}")

    def _sync_user_boxes_from_layer(self):
        if getattr(self, '_updating_layers', False) or not self.user_box_layer.editable:
            return
        try:
            obj_ids_prop = None
            try:
                obj_ids_prop = self.user_box_layer.properties.get('obj_id', None)
            except Exception:
                obj_ids_prop = None
            new_boxes = defaultdict(dict)
            for i, corners in enumerate(self.user_box_layer.data):
                if len(corners) < 4:
                    continue
                t = int(corners[0][0])
                y1, x1 = int(corners[0][1]), int(corners[0][2])
                y2, x2 = int(corners[2][1]), int(corners[2][2])
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                obj_id = int(obj_ids_prop[i]) if obj_ids_prop is not None and len(obj_ids_prop) > i else int(self.current_obj_id)
                new_boxes[t][obj_id] = [x1, y1, x2, y2]
            self.box_prompts = {t: dict(objs) for t, objs in new_boxes.items()}
        except Exception as e:
            print(f"Error syncing user boxes: {e}")

    def _init_text_layer(self):
        text_data = self._generate_text_data()
        if text_data:
            self.text_layer = self.viewer.add_points(
                text_data['coordinates'],
                text=text_data['labels'],
                name='Object IDs',
                face_color='transparent',
                size=1,
                visible=self.text_visible
            )
        else:
            self.text_layer = self.viewer.add_points(
                np.empty((0, 3)),
                text=[],
                name='Object IDs',
                face_color='transparent',
                size=1,
                visible=self.text_visible
            )
    
    def _generate_text_data(self):
        coordinates = []
        labels = []
        for frame_idx, segments in self.video_segments.items():
            for obj_id, logits in segments.items():
                if isinstance(logits, torch.Tensor):
                    mask = (logits.sigmoid() > 0.5).cpu().numpy()
                else:
                    mask = (logits > 0.5)
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask[0]
                if mask.sum() > 0:
                    try:
                        cy, cx = center_of_mass(mask)
                        if not (np.isnan(cy) or np.isnan(cx)):
                            coordinates.append([frame_idx, cy, cx])
                            labels.append(f"ID:{obj_id}")
                    except Exception as e:
                        print(f"Warning: Could not calculate center for frame {frame_idx}, obj {obj_id}: {e}")
                        continue
        if coordinates:
            return {'coordinates': np.array(coordinates, dtype=np.float64), 'labels': labels}
        return None

    def _box2corners(self, f, x1, y1, x2, y2):
        return np.array([
            [f, y1, x1],
            [f, y1, x2],
            [f, y2, x2],
            [f, y2, x1]
        ], dtype=np.float64)

    def _build_controls(self):
        layout = QVBoxLayout()
        current_frame_hl = QHBoxLayout()
        current_frame_hl.addWidget(QLabel('Current Frame:'))
        self.current_frame_label = QLabel(str(self.frame_idx))
        self.current_frame_label.setStyleSheet("font-weight: bold; color: blue;")
        current_frame_hl.addWidget(self.current_frame_label)
        layout.addLayout(current_frame_hl)
        fhl = QHBoxLayout()
        fhl.addWidget(QLabel('Manual Frame:'))
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, self.n_frames-1)
        self.frame_spin.valueChanged.connect(self.on_frame_change)
        fhl.addWidget(self.frame_spin)
        layout.addLayout(fhl)
        ohl = QHBoxLayout()
        ohl.addWidget(QLabel('Object id:'))
        self.obj_spin = QSpinBox()
        self.obj_spin.setRange(1,100)
        self.obj_spin.setValue(self.current_obj_id)
        self.obj_spin.valueChanged.connect(self.on_obj_change)
        ohl.addWidget(self.obj_spin)
        layout.addLayout(ohl)
        self.manual_edit_button = None
        btns = [
            ('Add + Point', self.enable_add_user_pos, 'btn_add_pos'),
            ('Add - Point', self.enable_add_user_neg, 'btn_add_neg'),
            ('Add Box', self.enable_add_user_box, 'btn_add_box'),
            ('Manual Edit', self.toggle_manual_annotation, 'manual_edit_button'),
            ('Edit Points', self.toggle_edit_user_pts, 'btn_edit_points'),
            ('Edit Boxes', self.toggle_edit_user_boxes, 'btn_edit_boxes'),
            ('Propagate', self.propagate_prompt, None),
            ('Slice Propagate', self.slicewise_propagate_prompt, None),
            ('3D Volume Render', lambda: render_manual_volume(self), None),
            ('Prompt Undo', self.prompt_undo, None),
            ('Prompt Redo', self.prompt_redo, None),
            ('Mask Undo', self.mask_undo, None),
            ('Mask Redo', self.mask_redo, None),
            ('Save', lambda: save_masks_auto(self), None)
        ]
        for lbl, fn, attr in btns:
            btn = QPushButton(lbl)
            btn.clicked.connect(fn)
            layout.addWidget(btn)
            if attr:
                setattr(self, attr, btn)
        self.setLayout(layout)

    def _select_layer(self, layer):
        try:
            self.viewer.layers.selection = [layer]
            # make it the active layer explicitly so mode/edits apply
            self.viewer.layers.selection.active = layer
            self._focus_viewer_canvas()
        except Exception:
            pass

    def _focus_viewer_canvas(self):
        try:
            if hasattr(self.viewer, 'window') and hasattr(self.viewer.window, 'qt_viewer'):
                canvas = getattr(self.viewer.window.qt_viewer, 'canvas', None)
                if canvas and hasattr(canvas, 'native'):
                    canvas.native.setFocus()
        except Exception:
            pass

    def _bind_shortcuts(self):
        keymap = [
            ('h', self.enable_add_user_pos),
            ('j', self.enable_add_user_neg),
            ('r', self.enable_add_user_box),
            ('t', self.toggle_edit_user_boxes),
            ('q', self.toggle_manual_annotation),
            ('k', self.toggle_edit_user_pts),
            ('y', self._toggle_mask_opacity),
            ('o', lambda: self._set_mask_opacity(1.0)),
            ('u', lambda: self._bump_mask_opacity(-0.1)),
            ('i', lambda: self._bump_mask_opacity(0.1)),
        ]
        for key, handler in keymap:
            try:
                self.viewer.bind_key(key, lambda v, h=handler: h(), overwrite=True, bind_global=True)
            except TypeError:
                self.viewer.bind_key(key, lambda v, h=handler: h(), overwrite=True)

        def _bind_global(seq, fn, overwrite=False):
            try:
                self.viewer.bind_key(seq, lambda v: fn(), overwrite=overwrite, bind_global=True)
            except TypeError:
                self.viewer.bind_key(seq, lambda v: fn(), overwrite=overwrite)

        _bind_global('s', lambda: save_masks_auto(self), overwrite=True)
        _bind_global('Control-Z', self.prompt_undo)
        _bind_global('Control-Y', self.prompt_redo)
        _bind_global('Control-X', self.mask_undo)
        _bind_global('Control-U', self.mask_redo)
        _bind_global('Control-S', lambda: save_masks_auto(self))
        _bind_global('Ctrl+Z', self.prompt_undo)
        _bind_global('Ctrl+Y', self.prompt_redo)
        _bind_global('Ctrl+X', self.mask_undo)
        _bind_global('Ctrl+U', self.mask_redo)
        _bind_global('Ctrl+S', lambda: save_masks_auto(self))
        _bind_global('Alt+Z', self.mask_undo, overwrite=True)
        _bind_global('Alt+Y', self.mask_redo, overwrite=True)
        _bind_global('alt+z', self.mask_undo, overwrite=True)
        _bind_global('alt+y', self.mask_redo, overwrite=True)
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
            ('Ctrl+S', lambda: save_masks_auto(self)),
            ('Alt+Z', self.mask_undo),
            ('Alt+Y', self.mask_redo),
            ('Alt+z', self.mask_undo),
            ('Alt+y', self.mask_redo),
            ('h', self.enable_add_user_pos),
            ('j', self.enable_add_user_neg),
            ('r', self.enable_add_user_box),
            ('t', self.toggle_edit_user_boxes),
            ('q', self.toggle_manual_annotation),
            ('k', self.toggle_edit_user_pts),
            ('y', self._toggle_mask_opacity),
            ('u', lambda: self._bump_mask_opacity(-0.1)),
            ('i', lambda: self._bump_mask_opacity(0.1)),
            ('o', lambda: self._set_mask_opacity(1.0)),
            ('s', lambda: save_masks_auto(self)),
            ('n', self.next_patient if getattr(self, 'navigation_manager', None) is not None else None),
            ('b', self.prev_patient if getattr(self, 'navigation_manager', None) is not None else None),
            ('h', self.enable_add_user_pos),
            ('j', self.enable_add_user_neg),
            ('r', self.enable_add_user_box),
            ('t', self.toggle_edit_user_boxes),
            ('q', self.toggle_manual_annotation),
            ('k', self.toggle_edit_user_pts),
            ('y', self._toggle_mask_opacity),
            ('u', lambda: self._bump_mask_opacity(-0.1)),
            ('i', lambda: self._bump_mask_opacity(0.1)),
            ('o', lambda: self._set_mask_opacity(1.0)),
            ('s', lambda: save_masks_auto(self)),
            ('n', self.next_patient if getattr(self, 'navigation_manager', None) is not None else None),
            ('H', self.enable_add_user_pos),
            ('J', self.enable_add_user_neg),
            ('R', self.enable_add_user_box),
            ('T', self.toggle_edit_user_boxes),
            ('Q', self.toggle_manual_annotation),
            ('K', self.toggle_edit_user_pts),
            ('Y', self._toggle_mask_opacity),
            ('U', lambda: self._bump_mask_opacity(-0.1)),
            ('I', lambda: self._bump_mask_opacity(0.1)),
            ('O', lambda: self._set_mask_opacity(1.0)),
            ('S', lambda: save_masks_auto(self)),
            ('N', self.next_patient if getattr(self, 'navigation_manager', None) is not None else None),
            ('B', self.prev_patient if getattr(self, 'navigation_manager', None) is not None else None),
        ]:
            if handler:
                sc = QShortcut(QKeySequence(seq), self)
                sc.setContext(Qt.ApplicationShortcut)
                sc.activated.connect(handler)
                self._qt_shortcuts.append(sc)
        # Extra Alt shortcuts bound to viewer window/canvas to bypass OS menu focus stealing
        alt_targets = []
        try:
            if hasattr(self.viewer, 'window'):
                alt_targets.append(getattr(self.viewer.window, '_qt_window', None))
                if hasattr(self.viewer.window, 'qt_viewer'):
                    alt_targets.append(getattr(self.viewer.window.qt_viewer, 'canvas', None))
                if hasattr(self.viewer.window, 'qt_viewer') and hasattr(self.viewer.window.qt_viewer, 'canvas'):
                    alt_targets.append(getattr(self.viewer.window.qt_viewer.canvas, 'native', None))
        except Exception:
            pass
        for tgt in alt_targets:
            if not tgt:
                continue
            for seq, handler in [('Alt+Z', self.mask_undo), ('Alt+z', self.mask_undo), ('Alt+Y', self.mask_redo), ('Alt+y', self.mask_redo)]:
                try:
                    sc = QShortcut(QKeySequence(seq), tgt)
                    sc.setContext(Qt.WidgetWithChildrenShortcut)
                    sc.activated.connect(handler)
                    self._qt_shortcuts.append(sc)
                except Exception:
                    pass

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

    def _ensure_box_prompt(self, frame_idx, obj_id):
        # If no box prompt exists for this frame/obj, try to build one from current mask logits
        try:
            if frame_idx in self.box_prompts and obj_id in self.box_prompts[frame_idx]:
                return False
            if frame_idx not in self.video_segments:
                return False
            if obj_id not in self.video_segments[frame_idx]:
                return False
            logits = self.video_segments[frame_idx][obj_id]
            if isinstance(logits, torch.Tensor):
                mask = (logits.sigmoid() > 0.5).cpu().numpy()
            else:
                mask = (logits > 0.5)
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            coords = np.argwhere(mask)
            if coords.size == 0:
                return False
            y_min, x_min = coords[:,0].min(), coords[:,1].min()
            y_max, x_max = coords[:,0].max(), coords[:,1].max()
            if frame_idx not in self.box_prompts:
                self.box_prompts[frame_idx] = {}
            self.box_prompts[frame_idx][obj_id] = [int(x_min), int(y_min), int(x_max), int(y_max)]
            return True
        except Exception as e:
            print(f"Auto box generation failed for frame {frame_idx}, obj {obj_id}: {e}")
            return False

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

    def toggle_text_visibility(self):
        self.text_visible = not self.text_visible
        if hasattr(self, 'text_layer'):
            self.text_layer.visible = self.text_visible
        print(f"Object ID text visibility: {'ON' if self.text_visible else 'OFF'}")

    def update_text_layer(self):
        if hasattr(self, 'text_layer'):
            text_data = self._generate_text_data()
            if text_data:
                self.text_layer.data = text_data['coordinates']
                self.text_layer.text = text_data['labels']
            else:
                self.text_layer.data = np.empty((0, 3))
                self.text_layer.text = []
            self.text_layer.visible = self.text_visible

    def cancel_prompt_mode(self, evt=None):
        self.img_layer.mouse_drag_callbacks.clear()
        self.mask_layer.mouse_drag_callbacks.clear()
        try:
            self.user_box_layer.mouse_drag_callbacks.clear()
        except Exception:
            pass
        if self.active_tool == 'edit_pts':
            self.user_pts_layer.editable = False
        if self.active_tool == 'edit_boxes':
            self.user_box_layer.editable = False
        if not self.manual_edit_enabled:
            self.user_pts_layer.editable = False
            self.user_box_layer.editable = False
            self.mask_layer.editable = False
            try:
                self.mask_layer.mode = 'pan_zoom'
            except Exception:
                pass
        self._reset_prompt_buttons()
        self.active_tool = None

    def _activate_tool(self, name):
        if self.manual_edit_enabled and name != 'manual_edit':
            print("Manual Edit is active; disable it before using other tools.")
            return False
        if self.active_tool == name:
            self.cancel_prompt_mode()
            return False
        self.cancel_prompt_mode()
        self.active_tool = name
        return True

    def enable_add_user_pos(self):
        if self.manual_edit_enabled:
            print("Manual Edit is enabled; point addition is unavailable.")
            return
        if not self._activate_tool('add_pos'):
            return
        self._select_layer(self.img_layer)
        self._set_button_active(self.btn_add_pos, True)
        self._set_button_active(self.btn_add_neg, False)
        self._set_button_active(self.btn_add_box, False)
        def cb(layer, evt):
            if evt.type != 'mouse_press': return
            t = int(self.viewer.dims.current_step[0])
            y, x = map(int, evt.position[1:])
            if t not in self.point_prompts:
                self.point_prompts[t] = {}
            if self.current_obj_id not in self.point_prompts[t]:
                self.point_prompts[t][self.current_obj_id] = []
            self.point_prompts[t][self.current_obj_id].append((x, y, 1))
            self.prompt_history.append(('add_pos_pt', t, self.current_obj_id, x, y))
            self.redo_history.clear()
            self.update_prompt_layers()
            self._record_prompt_event('user_pos_point', t, self.current_obj_id)
            print(f"Added positive point at frame {t}, position ({x}, {y}) - should be GREEN")
        self.img_layer.mouse_drag_callbacks.append(cb)

    def enable_add_user_neg(self):
        if self.manual_edit_enabled:
            print("Manual Edit is enabled; point addition is unavailable.")
            return
        if not self._activate_tool('add_neg'):
            return
        self._select_layer(self.img_layer)
        self._set_button_active(self.btn_add_neg, True)
        self._set_button_active(self.btn_add_pos, False)
        self._set_button_active(self.btn_add_box, False)
        def cb(layer, evt):
            if evt.type != 'mouse_press': return
            t = int(self.viewer.dims.current_step[0])
            y, x = map(int, evt.position[1:])
            if t not in self.point_prompts:
                self.point_prompts[t] = {}
            if self.current_obj_id not in self.point_prompts[t]:
                self.point_prompts[t][self.current_obj_id] = []
            self.point_prompts[t][self.current_obj_id].append((x, y, 0))
            self.prompt_history.append(('add_neg_pt', t, self.current_obj_id, x, y))
            self.redo_history.clear()
            self.update_prompt_layers()
            self._record_prompt_event('user_neg_point', t, self.current_obj_id)
            print(f"Added negative point at frame {t}, position ({x}, {y}) - should be RED")
        self.img_layer.mouse_drag_callbacks.append(cb)

    def enable_add_user_box(self):
        if self.manual_edit_enabled:
            print("Manual Edit is active: box prompt creation is disabled.")
            return
        if self.active_tool == 'add_box':
            self.cancel_prompt_mode()
            return
        if not self._activate_tool('add_box'):
            return
        self._select_layer(self.user_box_layer)
        self.user_box_layer.editable = True
        try:
            self.user_box_layer.mode = 'add_rectangle'
        except Exception:
            pass
        self._set_button_active(self.btn_add_box, True)
        self._set_button_active(self.btn_add_pos, False)
        self._set_button_active(self.btn_add_neg, False)
        pts = []
        def cb(layer, evt):
            if self.manual_edit_enabled:
                return
            if evt.type != 'mouse_press': return
            t = int(self.viewer.dims.current_step[0])
            y, x = map(int, evt.position[1:])
            pts.append((x, y))
            if len(pts) == 2:
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                if t not in self.box_prompts:
                    self.box_prompts[t] = {}
                self.box_prompts[t][self.current_obj_id] = [x1, y1, x2, y2]
                self.prompt_history.append(('add_box', t, self.current_obj_id, x1, y1, x2, y2))
                self.redo_history.clear()
                pts.clear()
                self.update_prompt_layers()
                self._record_prompt_event('user_box', t, self.current_obj_id)
                print(f"Added box at frame {t}, corners ({x1}, {y1}) to ({x2}, {y2})")
        self.mask_layer.mouse_drag_callbacks.clear()
        self.user_box_layer.mouse_drag_callbacks.clear()
        self.user_box_layer.mouse_drag_callbacks.append(cb)

    def toggle_manual_annotation(self):
        # Enable napari's native painting/box drawing without needing prompt buttons
        self.manual_edit_enabled = not self.manual_edit_enabled
        self._reset_prompt_buttons()
        self.active_tool = None
        if self.manual_edit_enabled:
            # Clear drag callbacks to stop creating box prompts while in manual mode
            self.img_layer.mouse_drag_callbacks.clear()
            self.mask_layer.mouse_drag_callbacks.clear()
            self.mask_layer.editable = True
            try:
                self.mask_layer.mode = 'paint'
            except Exception:
                pass
            self.mask_layer.selected_label = self.current_obj_id
            self.user_box_layer.editable = False
            try:
                self.user_box_layer.mode = 'select'
            except Exception:
                pass
            self._select_layer(self.mask_layer)
            self._set_button_active(self.manual_edit_button, True)
            self.viewer.status = "Manual Edit ON"
            print("Manual annotation enabled: paint on 'Mask' layer or draw rectangles in 'User Boxes'.")
            if self.metrics and self.metrics.is_active():
                self.manual_edit_started_at = time.time()
            try:
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
        else:
            self.img_layer.mouse_drag_callbacks.clear()
            self.mask_layer.mouse_drag_callbacks.clear()
            self.mask_layer.editable = False
            try:
                self.mask_layer.mode = 'pan_zoom'
            except Exception:
                pass
            self.user_box_layer.editable = False
            try:
                self.user_box_layer.mode = 'select'
            except Exception:
                pass
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

    def toggle_edit_auto_pts(self):
        print("Auto points layer removed; use user points instead.")

    def toggle_edit_auto_boxes(self):
        print("Auto box layer removed; use user boxes instead.")

    def toggle_edit_user_pts(self):
        if self.manual_edit_enabled:
            print("Manual Edit is enabled; point editing is unavailable.")
            return
        if not self._activate_tool('edit_pts'):
            self.user_pts_layer.editable = False
            return
        self._select_layer(self.user_pts_layer)
        self.user_pts_layer.editable = True
        try:
            self.user_pts_layer.mode = 'select'
        except Exception:
            pass
        self._set_button_active(self.btn_edit_points, True)
        print("User points editing enabled - you can move/delete user points")

    def toggle_edit_user_boxes(self):
        if self.manual_edit_enabled:
            print("Manual Edit is enabled; box editing is unavailable.")
            return
        if self.active_tool == 'edit_boxes':
            self.cancel_prompt_mode()
            self.user_box_layer.editable = False
            try:
                self.user_box_layer.mode = 'select'
            except Exception:
                pass
            return
        if not self._activate_tool('edit_boxes'):
            self.user_box_layer.editable = False
            try:
                self.user_box_layer.mode = 'pan_zoom'
            except Exception:
                pass
            return
        self._select_layer(self.user_box_layer)
        self.user_box_layer.editable = True
        try:
            self.user_box_layer.mode = 'select'
        except Exception:
            pass
        self._set_button_active(self.btn_edit_boxes, True)
        print("User boxes editing enabled - rectangles will maintain shape during editing")

    def update_prompt_layers(self):
        self._updating_layers = True
        # Reset points layer to avoid napari size broadcast issues when length changes
        try:
            self.user_pts_layer.data = np.empty((0,3), dtype=np.float64)
            self.user_pts_layer.face_color = []
            self.user_pts_layer.size = 5
        except Exception:
            pass
        # Build points directly from point_prompts (predicted + user edits)
        user_pts, user_pt_colors, user_ids, user_labels = [], [], [], []
        for frame_idx, objs in self.point_prompts.items():
            for obj_id, pts_list in objs.items():
                for pt_info in pts_list:
                    if len(pt_info) >= 3:
                        x, y, label = pt_info[:3]
                        user_pts.append([int(frame_idx), int(y), int(x)])
                        user_pt_colors.append('green' if label == 1 else 'red')
                        user_ids.append(int(obj_id))
                        user_labels.append(f"{ '+' if label==1 else '-' }{int(obj_id)}")
        if user_pts:
            self.user_pts_layer.data = np.array(user_pts, dtype=np.float64)
            self.user_pts_layer.face_color = user_pt_colors
            try:
                self.user_pts_layer.properties = {'obj_id': np.array(user_ids, dtype=int)}
                self.user_pts_layer.text = user_labels
                self.user_pts_layer.text.color = 'white'
                self.user_pts_layer.text.size = 10
                self.user_pts_layer.text.anchor = 'upper left'
                self.user_pts_layer.text.visible = True
                try:
                    self.user_pts_layer.size = 5
                except Exception:
                    pass
            except Exception:
                pass
        else:
            self.user_pts_layer.data = np.empty((0,3), dtype=np.float64)
            self.user_pts_layer.face_color = []
            try:
                self.user_pts_layer.properties = {'obj_id': np.array([], dtype=int)}
                self.user_pts_layer.text = []
                try:
                    self.user_pts_layer.size = 5
                except Exception:
                    pass
            except Exception:
                pass

        # Build boxes directly from box_prompts
        user_boxes = []
        box_ids = []
        for frame_idx, objs in self.box_prompts.items():
            for obj_id, box in objs.items():
                try:
                    if isinstance(box, (list, tuple, np.ndarray, torch.Tensor)) and len(box) >= 4:
                        box_vals = box.cpu().numpy().tolist() if isinstance(box, torch.Tensor) else (
                            box.tolist() if isinstance(box, np.ndarray) else list(box)
                        )
                        x1, y1, x2, y2 = map(int, box_vals[:4])
                        corners = self._box2corners(int(frame_idx), x1, y1, x2, y2)
                        user_boxes.append(corners)
                        box_ids.append(int(obj_id))
                    else:
                        print(f"Invalid box format for frame {frame_idx}: {box}")
                except Exception as e:
                    print(f"Error processing box for frame {frame_idx}: {e}")
                    continue
        if user_boxes:
            self.user_box_layer.data = np.array(user_boxes, dtype=np.float64)
            try:
                self.user_box_layer.properties = {'obj_id': np.array(box_ids, dtype=int)}
                palette = ['magenta', 'cyan', 'yellow', 'orange', 'blue', 'green', 'white']
                edge_colors = [palette[obj_id % len(palette)] for obj_id in box_ids]
                self.user_box_layer.edge_color = edge_colors
                self.user_box_layer.text = [f"ID:{oid}" for oid in box_ids]
                self.user_box_layer.text.color = 'white'
                self.user_box_layer.text.size = 10
                self.user_box_layer.text.anchor = 'upper left'
                self.user_box_layer.text.visible = True
            except Exception:
                pass
        else:
            self.user_box_layer.data = np.empty((0,4,3), dtype=np.float64)

        self._updating_layers = False
        print(f"Updated prompt layers: {len(user_pts)} points, {len(user_boxes)} boxes")

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

    def prompt_undo(self):
        if not self.prompt_history:
            return
        last_action = self.prompt_history.pop()
        self.redo_history.append(last_action)
        action_type = last_action[0]
        if action_type in ['add_pos_pt', 'add_neg_pt']:
            _, t, obj_id, x, y = last_action
            if (t in self.point_prompts and obj_id in self.point_prompts[t] and self.point_prompts[t][obj_id]):
                self.point_prompts[t][obj_id].pop()
                if not self.point_prompts[t][obj_id]:
                    del self.point_prompts[t][obj_id]
                if not self.point_prompts[t]:
                    del self.point_prompts[t]
        elif action_type == 'add_box':
            _, t, obj_id, x1, y1, x2, y2 = last_action
            if t in self.box_prompts and obj_id in self.box_prompts[t]:
                del self.box_prompts[t][obj_id]
                if not self.box_prompts[t]:
                    del self.box_prompts[t]
        self.update_prompt_layers()
        if self.metrics and self.metrics.is_active():
            self.metrics.inc_counter('undo_actions', 1)
            self.metrics.add_event('undo', action=action_type)

    def prompt_redo(self):
        if not self.redo_history:
            return
        action = self.redo_history.pop()
        self.prompt_history.append(action)
        action_type = action[0]
        if action_type == 'add_pos_pt':
            _, t, obj_id, x, y = action
            if t not in self.point_prompts:
                self.point_prompts[t] = {}
            if obj_id not in self.point_prompts[t]:
                self.point_prompts[t][obj_id] = []
            self.point_prompts[t][obj_id].append((x, y, 1))
        elif action_type == 'add_neg_pt':
            _, t, obj_id, x, y = action
            if t not in self.point_prompts:
                self.point_prompts[t] = {}
            if obj_id not in self.point_prompts[t]:
                self.point_prompts[t][obj_id] = []
            self.point_prompts[t][obj_id].append((x, y, 0))
        elif action_type == 'add_box':
            _, t, obj_id, x1, y1, x2, y2 = action
            if t not in self.box_prompts:
                self.box_prompts[t] = {}
            self.box_prompts[t][obj_id] = [x1, y1, x2, y2]
        self.update_prompt_layers()
        if self.metrics and self.metrics.is_active():
            self.metrics.inc_counter('redo_actions', 1)
            self.metrics.add_event('redo', action=action_type)

    def propagate_prompt(self):
        prop_start = time.time()
        all_prompt_frames = set()
        all_prompt_frames.update(self.box_prompts.keys())
        all_prompt_frames.update(self.point_prompts.keys())
        if not all_prompt_frames:
            QMessageBox.warning(self, 'No Prompts', 'Add some prompts first!')
            return
        start_idx = min(all_prompt_frames)
        end_idx = max(all_prompt_frames)
        boxes_used = sum(len(objs) for frame_idx, objs in self.box_prompts.items() if start_idx <= frame_idx <= end_idx)
        pos_used = 0
        neg_used = 0
        for frame_idx, objs in self.point_prompts.items():
            if start_idx <= frame_idx <= end_idx:
                for pts_list in objs.values():
                    for pt_info in pts_list:
                        if len(pt_info) >= 3:
                            if pt_info[2] == 1:
                                pos_used += 1
                            elif pt_info[2] == 0:
                                neg_used += 1
        sub_imgs = self.imgs[start_idx:end_idx+1].to(self.device)
        propagated_frames = set()
        try:
            with torch.no_grad():
                state = self.net.val_init_state(imgs_tensor=sub_imgs)
                for frame_idx, objs in self.box_prompts.items():
                    if start_idx <= frame_idx <= end_idx:
                        local_idx = frame_idx - start_idx
                        for obj_id, box in objs.items():
                            if isinstance(box, torch.Tensor):
                                box_tensor = box.to(self.device)
                            else:
                                box_tensor = torch.tensor(box, device=self.device)
                            self.net.train_add_new_bbox(
                                inference_state=state,
                                frame_idx=local_idx,
                                obj_id=obj_id,
                                bbox=box_tensor,
                                clear_old_points=False
                            )
                for frame_idx, objs in self.point_prompts.items():
                    if start_idx <= frame_idx <= end_idx:
                        local_idx = frame_idx - start_idx
                        for obj_id, pts_list in objs.items():
                            if pts_list:
                                points = []
                                labels = []
                                for pt_info in pts_list:
                                    if len(pt_info) >= 3:
                                        x, y, label = pt_info[:3]
                                        points.append([x, y])
                                        labels.append(label)
                                if points:
                                    points_tensor = torch.tensor(points, device=self.device)
                                    labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
                                    self.net.train_add_new_points(
                                        inference_state=state,
                                        frame_idx=local_idx,
                                        obj_id=obj_id,
                                        points=points_tensor,
                                        labels=labels_tensor,
                                        clear_old_points=False
                                    )
                for out_local_idx, out_oids, out_logits in self.net.propagate_in_video(state, start_frame_idx=0):
                    global_idx = start_idx + out_local_idx
                    propagated_frames.add(global_idx)
                    self.video_segments[global_idx] = {
                        oid: logits.cpu()
                        for oid, logits in zip(out_oids, out_logits)
                    }
                self.net.reset_state(state)
        except Exception as e:
            if self.metrics and self.metrics.is_active():
                self.metrics.add_event('propagation_error', message=str(e))
            QMessageBox.critical(self, 'Propagation Error', f'Propagation failed: {e}')
            return
        self._set_mask_data(make_initial_label_stack(
            video_segments=self.video_segments,
            obj_ids=self.obj_ids,
            n_frames=self.n_frames,
            spatial_shape=self.imgs.shape[2:],
        ))
        self.update_text_layer()
        print(f'Propagation done for frames {start_idx}-{end_idx}')
        if self.metrics and self.metrics.is_active():
            end_ts = time.time()
            slice_count = int(end_idx - start_idx + 1)
            self.metrics.record_stage(
                'propagation_auto',
                prop_start,
                end_ts,
                start_frame=int(start_idx),
                end_frame=int(end_idx),
                slice_count=slice_count,
                frames_with_masks=len(propagated_frames),
                boxes_used=boxes_used,
                pos_points_used=pos_used,
                neg_points_used=neg_used,
                avg_sec_per_slice=round((end_ts - prop_start) / slice_count, 4) if slice_count > 0 else None,
            )
            self.metrics.add_event('propagation_completed', frames_with_masks=len(propagated_frames))
        try:
            QMessageBox.information(self, 'Propagation Complete', f'Frames {start_idx}-{end_idx} 처리 완료 (마스크 {len(propagated_frames)}장).')
        except Exception:
            pass

    def slicewise_propagate_prompt(self):
        prop_start = time.time()
        prompt_frames = set(self.point_prompts.keys()) | set(self.box_prompts.keys())
        if not prompt_frames:
            QMessageBox.warning(self, 'No Prompts', 'Add some prompts first!')
            return
        updated_frames = set()
        total_boxes = 0
        total_pos = 0
        total_neg = 0
        orig_labels = self.mask_layer.data.copy()
        prompt_modes = defaultdict(dict)  # frame_idx -> {obj_id: (has_pos, has_neg)}
        auto_boxes_added = False
        for frame_idx in sorted(prompt_frames):
            # auto-generate box if none and there are prompts
            frame_points = self.point_prompts.get(frame_idx, {})
            frame_boxes = self.box_prompts.get(frame_idx, {})
            has_prompts = bool(frame_points) or bool(frame_boxes)
            if has_prompts:
                # ensure box for each obj with prompts; if none exist, generate from existing mask
                obj_ids_to_cover = set(frame_points.keys()) | set(frame_boxes.keys())
                if not obj_ids_to_cover:
                    obj_ids_to_cover = {self.current_obj_id}
                for oid in obj_ids_to_cover:
                    added = self._ensure_box_prompt(frame_idx, oid)
                    auto_boxes_added = auto_boxes_added or added
                    pts_list = frame_points.get(oid, [])
                    has_pos = any(len(pt)>=3 and pt[2]==1 for pt in pts_list)
                    has_neg = any(len(pt)>=3 and pt[2]==0 for pt in pts_list)
                    prompt_modes[frame_idx][oid] = (has_pos, has_neg)
            frame_boxes = self.box_prompts.get(frame_idx, {})
            frame_points = self.point_prompts.get(frame_idx, {})
            if not frame_boxes and not frame_points:
                continue
            sub_imgs = self.imgs[frame_idx:frame_idx+1].to(self.device)
            try:
                with torch.no_grad():
                    state = self.net.val_init_state(imgs_tensor=sub_imgs)
                    for oid, box in frame_boxes.items():
                        if isinstance(box, torch.Tensor):
                            box_tensor = box.to(self.device)
                        else:
                            box_tensor = torch.tensor(box, device=self.device)
                        self.net.train_add_new_bbox(
                            inference_state=state,
                            frame_idx=0,
                            obj_id=oid,
                            bbox=box_tensor,
                            clear_old_points=False
                        )
                        total_boxes += 1
                    for oid, pts_list in frame_points.items():
                        if not pts_list:
                            continue
                        points = []
                        labels = []
                        for pt_info in pts_list:
                            if len(pt_info) >= 3:
                                x, y, label = pt_info[:3]
                                points.append([x, y])
                                labels.append(label)
                                if label == 1:
                                    total_pos += 1
                                else:
                                    total_neg += 1
                        if points:
                            points_tensor = torch.tensor(points, device=self.device)
                            labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
                            self.net.train_add_new_points(
                                inference_state=state,
                                frame_idx=0,
                                obj_id=oid,
                                points=points_tensor,
                                labels=labels_tensor,
                                clear_old_points=False
                            )
                    # mark prompts for boxes without points
                    for oid in frame_boxes.keys():
                        if oid not in prompt_modes[frame_idx]:
                            prompt_modes[frame_idx][oid] = (False, False)
                    for out_local_idx, out_oids, out_logits in self.net.propagate_in_video(state, start_frame_idx=0):
                        global_idx = frame_idx
                        self.video_segments[global_idx] = {
                            oid: logits.cpu()
                            for oid, logits in zip(out_oids, out_logits)
                        }
                        updated_frames.add(global_idx)
                    self.net.reset_state(state)
            except Exception as e:
                if self.metrics and self.metrics.is_active():
                    self.metrics.add_event('propagation_error_slice', message=str(e), frame=int(frame_idx))
                QMessageBox.critical(self, 'Slice Propagation Error', f'Propagation failed at frame {frame_idx}: {e}')
                return

        if auto_boxes_added:
            self.update_prompt_layers()

        if updated_frames:
            new_labels = orig_labels.copy()
            for fidx in updated_frames:
                frame_seg = new_labels[fidx].copy()
                for oid, (has_pos, has_neg) in prompt_modes.get(fidx, {}).items():
                    if fidx not in self.video_segments or oid not in self.video_segments[fidx]:
                        continue
                    logits = self.video_segments[fidx][oid]
                    if isinstance(logits, torch.Tensor):
                        mask_new = (logits.sigmoid() > 0.5).cpu().numpy()
                    else:
                        mask_new = (logits > 0.5)
                    if mask_new.ndim == 3 and mask_new.shape[0] == 1:
                        mask_new = mask_new[0]
                    mask_orig = (orig_labels[fidx] == oid)
                    if has_pos and has_neg:
                        mask_final = mask_new
                    elif has_pos:
                        mask_final = mask_orig | (mask_new & ~mask_orig)
                    elif has_neg:
                        mask_final = mask_orig & mask_new
                    else:
                        mask_final = mask_new
                    # clear old obj pixels then apply final
                    frame_seg[frame_seg == oid] = 0
                    frame_seg[mask_final.astype(bool)] = oid
                new_labels[fidx] = frame_seg
            self._set_mask_data(new_labels)
            self.update_text_layer()
        if self.metrics and self.metrics.is_active():
            end_ts = time.time()
            self.metrics.record_stage(
                'propagation_auto_slicewise',
                prop_start,
                end_ts,
                frames_with_masks=len(updated_frames),
                boxes_used=total_boxes,
                pos_points_used=total_pos,
                neg_points_used=total_neg,
            )
            self.metrics.add_event('propagation_completed_slicewise', frames_with_masks=len(updated_frames))
        try:
            QMessageBox.information(self, 'Slice Propagation Complete', f'총 {len(updated_frames)}개 슬라이스 업데이트 완료.')
        except Exception:
            pass

    def on_frame_change(self, val):
        self.frame_idx = val
        current_step = list(self.viewer.dims.current_step)
        current_step[0] = val
        self.viewer.dims.current_step = current_step
        self.current_frame_label.setText(str(val))
        print(f"Frame manually changed to: {val}")

    def on_obj_change(self, val):
        self.current_obj_id = val
        if self.manual_edit_enabled:
            self.mask_layer.selected_label = val

    def save_masks(self):
        save_masks_auto(self)

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
