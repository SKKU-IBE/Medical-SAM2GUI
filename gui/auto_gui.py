import numpy as np
import torch
import napari
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QPushButton,
    QMessageBox,
)
from collections import deque, defaultdict
import traceback
import threading
from scipy.ndimage import center_of_mass

from gui.prompts import make_initial_label_stack, normalize_box_prompts, normalize_point_prompts
from gui.rendering import render_auto_volume
from gui.io import save_masks_auto


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
        meta                  # spacing, origin, direction
    ):
        super().__init__()
        # State
        self.imgs = imgs
        self.video_segments = video_segments
        self.net = net
        self.device = device
        self.patient_id = patient_id
        self.meta = meta
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
        self.viewer = napari.Viewer(title=str(patient_id))
        self.viewer.bind_key('Escape', self.cancel_prompt_mode)

        # Layers
        self.img_layer = self.viewer.add_image(
            imgs.permute(0,2,3,1).cpu().numpy(), name='Image', rgb=True
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
        
        # Auto prompts visualization
        self._init_prompt_layers()
        
        # User prompts layers
        self.user_pts_layer = self.viewer.add_points(
            np.empty((0,3)), name='User Points', face_color='green', size=5
        )
        self.user_box_layer = self.viewer.add_shapes(
            np.empty((0,4,3)), name='User Boxes', shape_type='rectangle',
            edge_color='red', face_color=[0,0,0,0]
        )
        self.user_pts_layer.editable = False
        self.user_box_layer.editable = False
        self.manual_edit_enabled = False  # allow freehand label/box editing on demand

        # Controls
        self._build_controls()
        self.viewer.window.add_dock_widget(self, area='right')
        
        # Detect viewer's current step(frame) changes for automatic updates
        self._setup_viewer_callbacks()
        
        # Setup layer event callbacks
        self._setup_layer_callbacks()
        
        self.update_prompt_layers()

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
        @self.auto_pts_layer.events.data.connect
        def on_auto_points_data_change():
            if not hasattr(self, 'auto_pts_layer') or not self.auto_pts_layer.editable:
                return
            if getattr(self, '_updating_layers', False):
                return
            self._sync_auto_points_from_layer()
            print("Auto points data synchronized from layer")
        
        @self.user_pts_layer.events.data.connect
        def on_user_points_data_change():
            if not hasattr(self, 'user_pts_layer') or not self.user_pts_layer.editable:
                return
            if getattr(self, '_updating_layers', False):
                return
            self._sync_user_points_from_layer()
        
        self._setup_box_editing_events()

    def _setup_box_editing_events(self):
        """Setup box editing events - debounced data change detection"""
        self._auto_box_edit_timer = None
        self._user_box_edit_timer = None
        
        @self.auto_box_layer.events.data.connect
        def on_auto_boxes_data_change():
            if not hasattr(self, 'auto_box_layer') or not self.auto_box_layer.editable:
                return
            if getattr(self, '_updating_auto_boxes', False):
                return
            if self._auto_box_edit_timer:
                self._auto_box_edit_timer.cancel()
            self._auto_box_edit_timer = threading.Timer(0.3, self._sync_auto_boxes_with_rectangle_constraint)
            self._auto_box_edit_timer.start()
            
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

    def _sync_auto_boxes_with_rectangle_constraint(self):
        """Sync auto boxes with rectangle constraint - support both shrinking/expanding"""
        if getattr(self, '_updating_auto_boxes', False):
            return
        
        self._updating_auto_boxes = True
        try:
            current_data = list(self.auto_box_layer.data)
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
                print(f"Applying rectangle constraint to {len(corrected_data)} auto boxes")
                self.auto_box_layer.data = corrected_data
            self._sync_auto_boxes_from_layer()
        except Exception as e:
            print(f"Error in auto box rectangle constraint: {e}")
        finally:
            self._updating_auto_boxes = False

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

    def _sync_auto_points_from_layer(self):
        if getattr(self, '_updating_layers', False) or not self.auto_pts_layer.editable:
            return
        self.point_prompts.clear()
        if len(self.auto_pts_layer.data) > 0:
            for i, (frame_idx, y, x) in enumerate(self.auto_pts_layer.data):
                frame_idx, y, x = int(frame_idx), int(y), int(x)
                color = self.auto_pts_layer.face_color[i] if i < len(self.auto_pts_layer.face_color) else 'yellow'
                label = self._color_to_label(color)
                obj_id = self.current_obj_id
                if frame_idx not in self.point_prompts:
                    self.point_prompts[frame_idx] = {}
                if obj_id not in self.point_prompts[frame_idx]:
                    self.point_prompts[frame_idx][obj_id] = []
                self.point_prompts[frame_idx][obj_id].append((x, y, label))
        print(f"Synced auto points: {sum(len(objs) for objs in self.point_prompts.values())} total points")

    def _sync_auto_boxes_from_layer(self):
        if getattr(self, '_updating_layers', False) or not self.auto_box_layer.editable:
            return
        self.box_prompts.clear()
        if len(self.auto_box_layer.data) > 0:
            for corners in self.auto_box_layer.data:
                if len(corners) >= 4:
                    frame_idx = int(corners[0][0])
                    y1, x1 = int(corners[0][1]), int(corners[0][2])
                    y2, x2 = int(corners[2][1]), int(corners[2][2])
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    obj_id = self.current_obj_id
                    if frame_idx not in self.box_prompts:
                        self.box_prompts[frame_idx] = {}
                    self.box_prompts[frame_idx][obj_id] = [x1, y1, x2, y2]
        print(f"Synced auto boxes: {sum(len(objs) for objs in self.box_prompts.values())} total boxes")

    def _sync_user_points_from_layer(self):
        if getattr(self, '_updating_layers', False) or not self.user_pts_layer.editable:
            return
        user_added_points = set()
        for action in self.prompt_history:
            if action[0] in ['add_pos_pt', 'add_neg_pt']:
                _, t, obj_id, x, y = action
                user_added_points.add((t, obj_id, x, y))
        if len(self.user_pts_layer.data) > 0:
            new_user_points = []
            for i, (t, y, x) in enumerate(self.user_pts_layer.data):
                t, y, x = int(t), int(y), int(x)
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
                if (t, self.current_obj_id, x, y) in user_added_points:
                    new_user_points.append((t, self.current_obj_id, x, y, label))
            for t, obj_id, x, y, label in new_user_points:
                if t not in self.point_prompts:
                    self.point_prompts[t] = {}
                if obj_id not in self.point_prompts[t]:
                    self.point_prompts[t][obj_id] = []
                point_exists = any(pt[:3] == (x, y, label) for pt in self.point_prompts[t][obj_id])
                if not point_exists:
                    self.point_prompts[t][obj_id].append((x, y, label))

    def _sync_user_boxes_from_layer(self):
        if getattr(self, '_updating_layers', False) or not self.user_box_layer.editable:
            return
        user_added_boxes = set()
        for action in self.prompt_history:
            if action[0] == 'add_box':
                _, t, obj_id, x1, y1, x2, y2 = action
                user_added_boxes.add((t, obj_id))
        if len(self.user_box_layer.data) > 0:
            new_user_boxes = []
            for corners in self.user_box_layer.data:
                if len(corners) >= 4:
                    t = int(corners[0][0])
                    y1, x1 = int(corners[0][1]), int(corners[0][2])
                    y2, x2 = int(corners[2][1]), int(corners[2][2])
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    if (t, self.current_obj_id) in user_added_boxes:
                        new_user_boxes.append((t, self.current_obj_id, x1, y1, x2, y2))
            for t, obj_id, x1, y1, x2, y2 in new_user_boxes:
                if t not in self.box_prompts:
                    self.box_prompts[t] = {}
                self.box_prompts[t][obj_id] = [x1, y1, x2, y2]

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

    def _init_prompt_layers(self):
        auto_pts = []
        auto_pt_colors = []
        for frame_idx, objs in self.point_prompts.items():
            for _, pts_list in objs.items():
                for pt_info in pts_list:
                    if len(pt_info) >= 3:
                        x, y, label = pt_info[:3]
                        auto_pts.append([frame_idx, y, x])
                        auto_pt_colors.append('yellow' if label == 1 else 'orange')
        self.auto_pts_layer = self.viewer.add_points(
            np.array(auto_pts) if auto_pts else np.empty((0,3)), 
            name='Auto Points', face_color=auto_pt_colors if auto_pt_colors else 'yellow', size=5
        )
        auto_boxes = []
        for frame_idx, objs in self.box_prompts.items():
            for _, box in objs.items():
                if box is None:
                    continue 
                try:
                    if isinstance(box, torch.Tensor):
                        box_list = box.cpu().numpy().tolist()
                    elif isinstance(box, np.ndarray):
                        box_list = box.tolist()
                    elif isinstance(box, (list, tuple)):
                        box_list = list(box)
                    else:
                        continue
                    if len(box_list) < 4:
                        continue
                    x1, y1, x2, y2 = box_list[:4]
                    coords = []
                    for coord in [x1, y1, x2, y2]:
                        if isinstance(coord, torch.Tensor):
                            coord_val = coord.item()
                        elif isinstance(coord, (int, float, np.integer, np.floating)):
                            coord_val = float(coord)
                        else:
                            coord_val = None
                            break
                        coords.append(coord_val)
                    if None in coords:
                        continue
                    x1, y1, x2, y2 = coords
                    frame_idx_val = float(frame_idx)
                    corners = self._box2corners(frame_idx_val, x1, y1, x2, y2)
                    if corners.shape != (4, 3):
                        continue
                    if np.any(~np.isfinite(corners)):
                        continue
                    auto_boxes.append(corners)
                except Exception as e:
                    print(f"    Error processing box for frame {frame_idx}: {e}")
                    traceback.print_exc()
                    continue
        try:
            if auto_boxes:
                self.auto_box_layer = self.viewer.add_shapes(
                    auto_boxes,
                    name='Auto Boxes',
                    shape_type='rectangle',
                    edge_color='yellow',
                    face_color=[0,0,0,0]
                )
            else:
                self.auto_box_layer = self.viewer.add_shapes(
                    np.empty((0,4,3)),
                    name='Auto Boxes',
                    shape_type='rectangle',
                    edge_color='yellow',
                    face_color=[0,0,0,0]
                )
        except Exception as e:
            print(f"Error creating auto_box_layer: {e}")
            traceback.print_exc()
            self.auto_box_layer = self.viewer.add_shapes(
                np.empty((0,4,3)),
                name='Auto Boxes',
                shape_type='rectangle',
                edge_color='yellow',
                face_color=[0,0,0,0]
            )
        self.auto_pts_layer.editable = False
        self.auto_box_layer.editable = False
        print(f"Prompt layers initialized: {len(auto_pts)} points, {len(auto_boxes)} boxes")

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
            ('Add +Pt', self.enable_add_user_pos),
            ('Add -Pt', self.enable_add_user_neg),
            ('Add Box', self.enable_add_user_box),
            ('Manual Edit', self.toggle_manual_annotation),
            ('Edit Auto Pts', self.toggle_edit_auto_pts),
            ('Edit Auto Boxes', self.toggle_edit_auto_boxes),
            ('Edit User Pts', self.toggle_edit_user_pts),
            ('Edit User Boxes', self.toggle_edit_user_boxes),
            ('Toggle Object IDs', self.toggle_text_visibility),
            ('Propagate', self.propagate_prompt),
            ('3D Volume Render', lambda: render_auto_volume(self)),
            ('Undo', self.prompt_undo),
            ('Redo', self.prompt_redo),
            ('Save', lambda: save_masks_auto(self))
        ]
        for lbl, fn in btns:
            btn = QPushButton(lbl)
            if lbl == 'Manual Edit':
                self.manual_edit_button = btn
            btn.clicked.connect(fn)
            layout.addWidget(btn)
        self.setLayout(layout)

    def _select_layer(self, layer):
        try:
            self.viewer.layers.selection = [layer]
        except Exception:
            pass

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
        self.auto_pts_layer.editable = False
        self.auto_box_layer.editable = False
        if not self.manual_edit_enabled:
            self.user_pts_layer.editable = False
            self.user_box_layer.editable = False
            self.mask_layer.editable = False
            try:
                self.mask_layer.mode = 'pan_zoom'
            except Exception:
                pass

    def enable_add_user_pos(self):
        self.cancel_prompt_mode()
        self._select_layer(self.user_pts_layer)
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
            print(f"Added positive point at frame {t}, position ({x}, {y}) - should be GREEN")
        self.img_layer.mouse_drag_callbacks.append(cb)

    def enable_add_user_neg(self):
        self.cancel_prompt_mode()
        self._select_layer(self.user_pts_layer)
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
            print(f"Added negative point at frame {t}, position ({x}, {y}) - should be RED")
        self.img_layer.mouse_drag_callbacks.append(cb)

    def enable_add_user_box(self):
        if self.manual_edit_enabled:
            print("Manual Edit is active: box prompt creation is disabled.")
            return
        self.cancel_prompt_mode()
        self._select_layer(self.mask_layer)
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
                print(f"Added box at frame {t}, corners ({x1}, {y1}) to ({x2}, {y2})")
        self.mask_layer.mouse_drag_callbacks.append(cb)

    def toggle_manual_annotation(self):
        # Enable napari's native painting/box drawing without needing prompt buttons
        self.manual_edit_enabled = not self.manual_edit_enabled
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
            if self.manual_edit_button:
                self.manual_edit_button.setText('Manual Edit (ON)')
                self.manual_edit_button.setStyleSheet('background-color: lightgreen; font-weight: bold;')
            self.viewer.status = "Manual Edit ON"
            print("Manual annotation enabled: paint on 'Mask' layer or draw rectangles in 'User Boxes'.")
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
            if self.manual_edit_button:
                self.manual_edit_button.setText('Manual Edit')
                self.manual_edit_button.setStyleSheet('')
            self.viewer.status = "Manual Edit OFF"
            print("Manual annotation disabled.")

    def toggle_edit_auto_pts(self):
        self.cancel_prompt_mode()
        self._select_layer(self.auto_pts_layer)
        was_editable = self.auto_pts_layer.editable
        self.auto_pts_layer.editable = not was_editable
        if self.auto_pts_layer.editable:
            print("Auto points editing enabled - you can move/delete auto points")
        else:
            print("Auto points editing disabled")

    def toggle_edit_auto_boxes(self):
        self.cancel_prompt_mode()
        self._select_layer(self.auto_box_layer)
        was_editable = self.auto_box_layer.editable
        self.auto_box_layer.editable = not was_editable
        if self.auto_box_layer.editable:
            self.auto_box_layer.mode = 'select'
            print("Auto boxes editing enabled - rectangles will maintain shape during editing")
        else:
            print("Auto boxes editing disabled")

    def toggle_edit_user_pts(self):
        self.cancel_prompt_mode()
        self._select_layer(self.user_pts_layer)
        was_editable = self.user_pts_layer.editable
        self.user_pts_layer.editable = not was_editable
        if self.user_pts_layer.editable:
            print("User points editing enabled - you can move/delete user points")
        else:
            print("User points editing disabled")

    def toggle_edit_user_boxes(self):
        self.cancel_prompt_mode()
        self._select_layer(self.user_box_layer)
        was_editable = self.user_box_layer.editable
        self.user_box_layer.editable = not was_editable
        if self.user_box_layer.editable:
            self.user_box_layer.mode = 'select'
            print("User boxes editing enabled - rectangles will maintain shape during editing")
        else:
            print("User boxes editing disabled")

    def update_prompt_layers(self):
        self._updating_layers = True
        auto_pts, auto_pt_colors = [], []
        for frame_idx, objs in self.point_prompts.items():
            for _, pts_list in objs.items():
                for pt_info in pts_list:
                    if len(pt_info) >= 3:
                        x, y, label = pt_info[:3]
                        try:
                            if isinstance(x, torch.Tensor):
                                x = x.item()
                            if isinstance(y, torch.Tensor):
                                y = y.item()
                            if isinstance(label, torch.Tensor):
                                label = label.item()
                            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                                auto_pts.append([int(frame_idx), int(y), int(x)])
                                auto_pt_colors.append('yellow' if label == 1 else 'orange')
                        except Exception as e:
                            print(f"Error processing point {pt_info}: {e}")
                            continue
        if auto_pts:
            self.auto_pts_layer.data = np.array(auto_pts, dtype=np.float64)
            self.auto_pts_layer.face_color = auto_pt_colors
        else:
            self.auto_pts_layer.data = np.empty((0,3), dtype=np.float64)
            self.auto_pts_layer.face_color = []
        auto_boxes = []
        for frame_idx, objs in self.box_prompts.items():
            for _, box in objs.items():
                try:
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        coords = []
                        for coord in box[:4]:
                            if isinstance(coord, torch.Tensor):
                                coord_val = coord.item()
                            elif isinstance(coord, (int, float, np.integer, np.floating)):
                                coord_val = float(coord)
                            else:
                                coord_val = None
                                break
                            coords.append(coord_val)
                        if None not in coords:
                            x1, y1, x2, y2 = coords
                            if all(isinstance(c, (int, float)) for c in [x1, y1, x2, y2]):
                                corners = self._box2corners(int(frame_idx), int(x1), int(y1), int(x2), int(y2))
                                auto_boxes.append(corners)
                                print(f"Added auto box for frame {frame_idx}: ({x1},{y1},{x2},{y2})")
                    else:
                        print(f"Invalid box format for frame {frame_idx}: {box}")
                except Exception as e:
                    print(f"Error processing box for frame {frame_idx}: {e}")
                    continue
        if auto_boxes:
            self.auto_box_layer.data = np.array(auto_boxes, dtype=np.float64)
            print(f"Updated auto_box_layer with {len(auto_boxes)} boxes")
        else:
            self.auto_box_layer.data = np.empty((0,4,3), dtype=np.float64)
            print("No valid auto boxes found, setting empty data")
        user_pts, user_pt_colors = [], []
        for action in self.prompt_history:
            if action[0] == 'add_pos_pt':
                _, t, obj_id, x, y = action
                user_pts.append([int(t), int(y), int(x)])
                user_pt_colors.append('green')
            elif action[0] == 'add_neg_pt':
                _, t, obj_id, x, y = action
                user_pts.append([int(t), int(y), int(x)])
                user_pt_colors.append('red')
        if user_pts:
            self.user_pts_layer.data = np.array(user_pts, dtype=np.float64)
            self.user_pts_layer.face_color = user_pt_colors
        else:
            self.user_pts_layer.data = np.empty((0,3), dtype=np.float64)
            self.user_pts_layer.face_color = []
        user_boxes = []
        for action in self.prompt_history:
            if action[0] == 'add_box':
                _, t, obj_id, x1, y1, x2, y2 = action
                corners = self._box2corners(int(t), int(x1), int(y1), int(x2), int(y2))
                user_boxes.append(corners)
        if user_boxes:
            self.user_box_layer.data = np.array(user_boxes, dtype=np.float64)
        else:
            self.user_box_layer.data = np.empty((0,4,3), dtype=np.float64)
        self._updating_layers = False
        print(f"Updated prompt layers: {len(auto_pts)} auto points, {len(auto_boxes)} auto boxes, {len(user_pts)} user points, {len(user_boxes)} user boxes")

    def _get_patient_display_name(self):
        if isinstance(self.patient_id, (list, tuple)):
            return str(self.patient_id[0])
        return str(self.patient_id)

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

    def propagate_prompt(self):
        all_prompt_frames = set()
        all_prompt_frames.update(self.box_prompts.keys())
        all_prompt_frames.update(self.point_prompts.keys())
        if not all_prompt_frames:
            QMessageBox.warning(self, 'No Prompts', 'Add some prompts first!')
            return
        start_idx = min(all_prompt_frames)
        end_idx = max(all_prompt_frames)
        sub_imgs = self.imgs[start_idx:end_idx+1].to(self.device)
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
                self.video_segments[global_idx] = {
                    oid: logits.cpu()
                    for oid, logits in zip(out_oids, out_logits)
                }
            self.net.reset_state(state)
        self.mask_layer.data = make_initial_label_stack(
            video_segments=self.video_segments,
            obj_ids=self.obj_ids,
            n_frames=self.n_frames,
            spatial_shape=self.imgs.shape[2:],
        )
        self.update_text_layer()
        print(f'Propagation done for frames {start_idx}-{end_idx}')

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
        render_auto_volume(self)
