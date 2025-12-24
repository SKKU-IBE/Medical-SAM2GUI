import numpy as np
import torch
import napari
from collections import deque, defaultdict
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QPushButton,
    QMessageBox,
)
import threading
from PIL import Image

from gui.rendering import render_manual_volume
from gui.io import save_masks_manual


class ManualPromptNapariGUI(QWidget):
    """Napari GUI for manual prompt input (points/boxes) with propagate."""
    def __init__(self, imgs, net, device, patient_id, meta):
        super().__init__()
        if imgs.dim() == 5 and imgs.size(0) == 1:
            imgs = imgs.squeeze(0)
        self.imgs = imgs
        self.net = net
        self.device = device
        self.patient_id = patient_id
        self.meta = meta

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
            self.imgs.permute(0,2,3,1).cpu().numpy(), name='image, point layer', rgb=True
        )
        self.mask_layer = self.viewer.add_labels(
            np.zeros((self.n_frames,) + self.imgs.shape[2:], dtype=np.uint8), name='mask, box layer'
        )
        self.user_pts_layer = self.viewer.add_points(
            np.empty((0,3)), name='User Points correction layer', size=5
        )
        self.box_layer = self.viewer.add_shapes(
            np.empty((0,4,3)), name='User Boxes correction layer', shape_type='rectangle',
            edge_color='red', face_color=[0,0,0,0]
        )
        self.manual_edit_enabled = False  # allow napari painting/box drawing when toggled

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

    def _get_patient_display_name(self):
        if isinstance(self.patient_id, (list, tuple)):
            return str(self.patient_id[0])
        return str(self.patient_id)

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

    def _setup_manual_box_editing_events(self):
        self._box_edit_timer = None
        @self.box_layer.events.data.connect
        def on_boxes_data_change():
            if not hasattr(self, 'box_layer') or not self.box_layer.editable:
                return
            if getattr(self, '_updating_layers', False):
                return
            if self._box_edit_timer:
                self._box_edit_timer.cancel()
            self._box_edit_timer = threading.Timer(0.5, self._sync_manual_boxes_with_rectangle_constraint)
            self._box_edit_timer.start()

    def _sync_manual_boxes_with_rectangle_constraint(self):
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
            if not all(np.array_equal(new, old) for new, old in zip(updated_boxes, current_boxes)):
                print(f"Applying rectangle constraint to {len(updated_boxes)} manual boxes")
                self.box_layer.data = updated_boxes
        except Exception as e:
            print(f"Error in manual box rectangle constraint: {e}")
        finally:
            self._updating_layers = False

    def _sync_boxes_from_layer(self):
        if getattr(self, '_updating_layers', False) or not self.box_layer.editable:
            return
        old_boxes = self.box_prompts.copy()
        self.box_prompts.clear()
        if len(self.box_layer.data) > 0:
            for corners in self.box_layer.data:
                if len(corners) >= 4:
                    frame_idx = int(corners[0][0])
                    y1, x1 = int(corners[0][1]), int(corners[0][2])
                    y2, x2 = int(corners[2][1]), int(corners[2][2])
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    best_obj_id = None
                    min_distance = float('inf')
                    for old_t, old_obj_id, old_x1, old_y1, old_x2, old_y2 in old_boxes:
                        if old_t == frame_idx:
                            old_cx, old_cy = (old_x1 + old_x2) / 2, (old_y1 + old_y2) / 2
                            new_cx, new_cy = (x1 + x2) / 2, (y1 + y2) / 2
                            distance = ((old_cx - new_cx) ** 2 + (old_cy - new_cy) ** 2) ** 0.5
                            if distance < min_distance:
                                min_distance = distance
                                best_obj_id = old_obj_id
                    if best_obj_id is None:
                        best_obj_id = self.current_obj_id
                    self.box_prompts.append((frame_idx, best_obj_id, x1, y1, x2, y2))
                    print(f"  Frame {frame_idx}, ObjID {best_obj_id}: [{x1}, {y1}, {x2}, {y2}]")
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
        self.manual_edit_button = None
        btns = [
            ('Add + Point', self.enable_add_positive),
            ('Add - Point', self.enable_add_negative),
            ('Add Box',     self.enable_add_box),
            ('Manual Edit', self.toggle_manual_annotation),
            ('Edit Points', self.enable_edit_points),
            ('Edit Boxes',  self.enable_edit_boxes),
            ('Clear All',   self.clear_all_prompts),
            ('Propagate',   self.propagate_prompt),
            ('3D Volume Render', lambda: render_manual_volume(self)),
            ('Undo',        self.prompt_undo),
            ('Redo',        self.prompt_redo),
            ('Save Masks',  lambda: save_masks_manual(self))
        ]
        for label, func in btns:
            b = QPushButton(label)
            if label == 'Manual Edit':
                self.manual_edit_button = b
            b.clicked.connect(func)
            layout.addWidget(b)
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
        self.mask_layer.data = np.zeros((self.n_frames,) + self.imgs.shape[2:], dtype=np.uint8)
        self.update_layers()
        print("All prompts and masks cleared")

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
        if not self.manual_edit_enabled:
            self.user_pts_layer.editable = False
            self.box_layer.editable = False
            self.mask_layer.editable = False

    def enable_add_positive(self):
        self.cancel_prompt_mode()
        self.add_mode = 'pos'
        def cb(layer, event):
            if event.type != 'mouse_press': return
            t = int(self.viewer.dims.current_step[0])
            y, x = map(int, event.position[1:])
            self.pos_points.append((t, self.current_obj_id, y, x))
            self.prompt_history.append(('pos', t, self.current_obj_id, y, x))
            self.redo_history.clear()
            self.update_layers()
            print(f"Added positive point at frame {t}, position ({x}, {y}) - should be GREEN")
        self.img_layer.mouse_drag_callbacks.append(cb)

    def enable_add_negative(self):
        self.cancel_prompt_mode()
        self.add_mode = 'neg'
        def cb(layer, event):
            if event.type != 'mouse_press': return
            t = int(self.viewer.dims.current_step[0])
            y, x = map(int, event.position[1:])
            self.neg_points.append((t, self.current_obj_id, y, x))
            self.prompt_history.append(('neg', t, self.current_obj_id, y, x))
            self.redo_history.clear()
            self.update_layers()
            print(f"Added negative point at frame {t}, position ({x}, {y}) - should be RED")
        self.img_layer.mouse_drag_callbacks.append(cb)

    def enable_add_box(self):
        self.cancel_prompt_mode()
        self.add_mode = 'box'
        pts = []
        def cb(layer, event):
            if event.type != 'mouse_press': return
            t = int(self.viewer.dims.current_step[0])
            y, x = map(int, event.position[1:])
            pts.append((x, y))
            if len(pts) == 2:
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                self.box_prompts.append((t, self.current_obj_id, x1, y1, x2, y2))
                self.prompt_history.append(('box', t, self.current_obj_id, x1, y1, x2, y2))
                self.redo_history.clear()
                self.update_layers()
                print(f"Added box at frame {t}, corners ({x1}, {y1}) to ({x2}, {y2})")
                pts.clear()
        self.mask_layer.mouse_drag_callbacks.append(cb)

    def toggle_manual_annotation(self):
        # Enable napari's native painting/rectangle drawing without needing Add Box
        self.img_layer.mouse_drag_callbacks.clear()
        self.mask_layer.mouse_drag_callbacks.clear()
        self.add_mode = None
        self.manual_edit_enabled = not self.manual_edit_enabled
        if self.manual_edit_enabled:
            self.mask_layer.editable = True
            self.mask_layer.selected_label = self.current_obj_id
            self.box_layer.editable = True
            self.box_layer.mode = 'add_rectangle'
            if self.manual_edit_button:
                self.manual_edit_button.setText('Manual Edit (ON)')
                self.manual_edit_button.setStyleSheet('background-color: lightgreen; font-weight: bold;')
            self.viewer.status = "Manual Edit ON"
            print("Manual annotation enabled: paint on 'mask, box layer' or draw rectangles in 'User Boxes correction layer'.")
        else:
            self.mask_layer.editable = False
            self.box_layer.editable = False
            if self.manual_edit_button:
                self.manual_edit_button.setText('Manual Edit')
                self.manual_edit_button.setStyleSheet('')
            self.viewer.status = "Manual Edit OFF"
            print("Manual annotation disabled.")

    def enable_edit_points(self):
        self.cancel_prompt_mode()
        was_editable = self.user_pts_layer.editable
        self.user_pts_layer.editable = not was_editable
        if self.user_pts_layer.editable:
            print("Points editing enabled - you can move/delete points")
        else:
            print("Points editing disabled")

    def enable_edit_boxes(self):
        self.cancel_prompt_mode()
        was_editable = self.box_layer.editable
        self.box_layer.editable = not was_editable
        if self.box_layer.editable:
            self.box_layer.mode = 'select'
            print("Boxes editing enabled - rectangles will maintain shape during editing")
        else:
            print("Boxes editing disabled")

    def update_layers(self):
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
            self.user_pts_layer.face_color = []
        shapes = []
        for (t, oid, x1, y1, x2, y2) in self.box_prompts:
            corners = self._create_rectangle_corners(t, x1, y1, x2, y2)
            shapes.append(corners)
        if shapes:
            self.box_layer.data = np.array(shapes, dtype=np.float64)
        else:
            self.box_layer.data = np.empty((0, 4, 3), dtype=np.float64)
        self._update_object_id_text()
        self._updating_layers = False
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

    def prompt_redo(self):
        if not self.redo_history:
            return
        act = self.redo_history.pop()
        self.prompt_history.append(act)
        self.apply_prompt_history()

    def apply_prompt_history(self):
        self.pos_points.clear(); self.neg_points.clear(); self.box_prompts.clear()
        for it in self.prompt_history:
            cmd, t, oid, *rest = it
            if cmd == 'pos': y,x = rest; self.pos_points.append((t,oid,y,x))
            elif cmd == 'neg': y,x = rest; self.neg_points.append((t,oid,y,x))
            elif cmd == 'box': x1,y1,x2,y2 = rest; self.box_prompts.append((t,oid,x1,y1,x2,y2))
        self.update_layers()

    def propagate_prompt(self):
        print("Synchronizing layer data before propagate...")
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
        self.mask_layer.data = np.zeros((self.n_frames,) + self.imgs.shape[2:], dtype=np.uint8)
        sub = self.imgs[start:end+1].to(self.device)
        with torch.no_grad():
            state = self.net.val_init_state(imgs_tensor=sub)
            box_count = 0
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
                    pm[(t-start, oid)].append((x, y))
            for (t, oid, y, x) in self.neg_points:
                if start <= t <= end:
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
                return
            finally:
                self.net.reset_state(state)
                del state
            new_mask = np.zeros((self.n_frames,) + self.imgs.shape[2:], dtype=np.uint8)
            for i, m in result.items():
                new_mask[i] = m
            self.mask_layer.data = new_mask
            self._update_object_id_text()
            print(f"Propagation completed. Generated masks for {len(result)} frames")
            QMessageBox.information(self, 'Done', f'Propagated {start}~{end}\nGenerated {len(result)} masks')

    def save_masks(self):
        save_masks_manual(self)

    def render_3d_volume(self):
        render_manual_volume(self)
