"""Navigation manager and navigation-enabled GUI wrappers."""
import os
import time
import numpy as np
import napari
import torch
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
    QInputDialog,
    QWidget,
    QVBoxLayout,
    QComboBox,
)

from dataloader import SNU3DMRI_MedSAM2Dataset, load_dicom_series, load_nii_image, discover_studies
from gui.segmentation import auto_segmentation
from gui.auto_gui import MedSAM2NapariGUI
from gui.manual_gui import ManualPromptNapariGUI
from gui.metrics import UsageMetricsRecorder


class PatientNavigationManager:
    """Navigation Manager for sequential patient data processing."""
    def __init__(self, data_root, net, device, default_mode='manual', default_method=None, args=None, metrics_recorder=None):
        self.data_root = data_root
        self.net = net
        self.device = device
        self.default_mode = default_mode
        self.default_method = default_method
        self.args = args
        self.metrics = metrics_recorder or UsageMetricsRecorder()
        self.current_gui = None
        self.patient_index = 0
        self.user_inputs = {}
        self.last_user_settings = None
        self.double_viewers = {}
        self.current_patient_idx = 0
        self.patient_list = discover_studies(data_root)
        self._suppress_combo_signal = False
        self.nav_panel = None
        print(f"Found {len(self.patient_list)} patients to process")

    def load_patient_data(self, patient_path, patient_name, mode, method, preprocess, patient_idx_override=None):
        print(f"Loading patient {patient_name} with settings: mode={mode}, method={method}, preprocess={preprocess}")
        temp_dataset = SNU3DMRI_MedSAM2Dataset(
            data_root=self.data_root,
            preprocess=preprocess,
            method=method,
            mode=mode,
            # Save/reuse preprocessed volumes in a stable location alongside the dataset
            save_preproc_dir=os.path.join(self.args.data_path, "preprocessed") if self.args else None,
            img_size=self.args.image_size if self.args else 1024,
        )
        # If the caller provides an explicit index from the selection dialog, trust it first
        if isinstance(patient_idx_override, int) and 0 <= patient_idx_override < len(temp_dataset):
            patient_idx = patient_idx_override
        else:
            patient_idx = None
        def _norm(name):
            return name[:-7] if name.endswith('.nii.gz') else name[:-4] if name.endswith('.nii') else name

        target_name = _norm(patient_name)
        target_base = _norm(os.path.basename(patient_path))
        if patient_idx is None:
            for idx, (path, name) in enumerate(zip(temp_dataset.patient_paths, temp_dataset.patient_names)):
                if _norm(name) == target_name or _norm(os.path.basename(path)) == target_base:
                    patient_idx = idx
                    break
        if patient_idx is None:
            for idx, (path, name) in enumerate(zip(temp_dataset.patient_paths, temp_dataset.patient_names)):
                if path == patient_path:
                    patient_idx = idx
                    break
        if patient_idx is None:
            raise ValueError(f"Patient {patient_name} (path: {patient_path}) not found in dataset")
        return temp_dataset[patient_idx]

    def get_user_input_for_patient(self, patient_id):
        from gui.setup_dialogs import PatientInputDialog

        patient_choices = [self._rel_label(os.path.abspath(path)) for path, _ in self.patient_list]
        dialog = PatientInputDialog(
            patient_id,
            self.patient_index,
            defaults=self.last_user_settings,
            patient_choices=patient_choices,
            current_choice_idx=self.current_patient_idx,
        )
        if dialog.exec_() == dialog.Accepted:
            settings = dialog.get_settings()
            selected_idx = settings.get("selected_patient_idx")
            effective_idx = selected_idx if isinstance(selected_idx, int) and 0 <= selected_idx < len(patient_choices) else self.current_patient_idx
            effective_id = patient_choices[effective_idx] if patient_choices else patient_id
            print(
                f"Patient {effective_id} - Mode: {settings['mode']}, Method: {settings['method']}, Preprocess: {settings['preprocess']}"
            )
            self.last_user_settings = settings
            return settings
        print(f"Patient {patient_id} skipped by user")
        return None

    def create_double_viewer(self, double_path, patient_id):
        if not double_path:
            return None
        try:
            print(f"Creating double viewer for patient {patient_id} with path: {double_path}")
            if not os.path.exists(double_path):
                print(f"Warning: Double viewer path does not exist: {double_path}")
                return None
            if double_path.endswith('.nii') or double_path.endswith('.nii.gz'):
                arr_3d, _ = load_nii_image(double_path)
            else:
                arr_3d, _ = load_dicom_series(double_path)
            images = []
            for s in range(arr_3d.shape[0]):
                arr2d = arr_3d[s]
                arr_min, arr_max = arr2d.min(), arr2d.max()
                if arr_max > arr_min:
                    arr2d_norm = (arr2d - arr_min) / (arr_max - arr_min)
                else:
                    arr2d_norm = np.zeros_like(arr2d)
                img = Image.fromarray((arr2d_norm * 255).astype(np.uint8)).resize((1024, 1024))
                img_tensor = torch.from_numpy(np.array(img)).unsqueeze(0).repeat(3, 1, 1)
                images.append(img_tensor)
            vol = torch.stack(images, dim=0)
            vol = vol.permute(0, 2, 3, 1).cpu().numpy()
            double_viewer = napari.Viewer(title=f'Double Viewer - {patient_id} - {os.path.basename(double_path)}')
            double_viewer.add_image(vol, name=f'{os.path.basename(double_path)}', rgb=True, blending='translucent')
            return double_viewer
        except Exception as e:
            print(f"Error creating double viewer: {e}")
            return None

    def close_current_gui(self):
        if self.current_gui:
            try:
                self.current_gui.viewer.close()
            except Exception:
                pass
            try:
                self.current_gui.close()
            except Exception:
                pass
            self.current_gui = None
        if self.metrics and self.metrics.is_active():
            self.metrics.add_event('session_closed_without_save')
            self.metrics.finalize({'closed_without_save': True})

    def next_patient(self):
        self.close_current_gui()
        self.current_patient_idx += 1
        self.show_current_patient()

    def prev_patient(self):
        if self.current_patient_idx <= 0:
            print("Already at first patient; cannot go back")
            return
        self.close_current_gui()
        self.current_patient_idx -= 1
        self.show_current_patient()

    def pick_patient(self):
        if not self.patient_list:
            return
        try:
            items = []
            root_abs = os.path.abspath(self.data_root)
            for path, name in self.patient_list:
                path_abs = os.path.abspath(path)
                try:
                    rel = os.path.relpath(path_abs, root_abs)
                except Exception:
                    rel = path_abs
                items.append(rel)
            current_item = items[self.current_patient_idx] if 0 <= self.current_patient_idx < len(items) else items[0]
            chosen, ok = QInputDialog.getItem(None, "Select Patient", "Patient path:", items, items.index(current_item), False)
            if ok and chosen in items:
                idx = items.index(chosen)
                if idx == self.current_patient_idx:
                    return
                self.close_current_gui()
                self.current_patient_idx = idx
                self.show_current_patient()
        except Exception as e:
            print(f"Patient selection failed: {e}")

    def _rel_label(self, path_abs):
        try:
            root_abs = os.path.abspath(self.data_root)
            if os.path.commonpath([root_abs, path_abs]) == root_abs:
                return os.path.relpath(path_abs, root_abs)
        except Exception:
            pass
        return path_abs

    def _build_nav_panel(self):
        try:
            panel = QWidget()
            panel.setWindowTitle("Patient Navigation")
            layout = QVBoxLayout()
            layout.addWidget(QLabel("Select patient:"))
            self.patient_combo = QComboBox()
            for path, _ in self.patient_list:
                self.patient_combo.addItem(self._rel_label(os.path.abspath(path)))
            self.patient_combo.currentIndexChanged.connect(self._on_combo_changed)
            layout.addWidget(self.patient_combo)

            btn_layout = QHBoxLayout()
            back_btn = QPushButton('Back')
            back_btn.clicked.connect(self.prev_patient)
            next_btn = QPushButton('Next')
            next_btn.clicked.connect(self.next_patient)
            open_btn = QPushButton('Open Selected')
            open_btn.clicked.connect(self._open_selected)
            close_btn = QPushButton('Close Current')
            close_btn.clicked.connect(self.close_current_gui)
            btn_layout.addWidget(back_btn)
            btn_layout.addWidget(next_btn)
            btn_layout.addWidget(open_btn)
            btn_layout.addWidget(close_btn)
            layout.addLayout(btn_layout)

            panel.setLayout(layout)
            panel.show()
            self.nav_panel = panel
        except Exception as e:
            print(f"Failed to build navigation panel: {e}")

    def _open_selected(self):
        idx = self.patient_combo.currentIndex() if hasattr(self, 'patient_combo') else -1
        if idx < 0 or idx >= len(self.patient_list):
            return
        if idx == self.current_patient_idx:
            return
        self.close_current_gui()
        self.current_patient_idx = idx
        self.show_current_patient()

    def _on_combo_changed(self, idx):
        if self._suppress_combo_signal:
            return
        if idx < 0 or idx >= len(self.patient_list):
            return
        if idx == self.current_patient_idx:
            return
        self._open_selected()

    def _sync_nav_panel(self):
        if not self.nav_panel or not hasattr(self, 'patient_combo'):
            return
        try:
            self._suppress_combo_signal = True
            if 0 <= self.current_patient_idx < self.patient_combo.count():
                self.patient_combo.setCurrentIndex(self.current_patient_idx)
        finally:
            self._suppress_combo_signal = False

    def show_current_patient(self):
        try:
            if self.current_patient_idx >= len(self.patient_list):
                print("All patients have been processed.")
                return
            # Ask user which patient to process (allows jumping via dialog)
            user_input = self.get_user_input_for_patient(self._rel_label(os.path.abspath(self.patient_list[self.current_patient_idx][0])))
            if user_input is None:
                self.current_patient_idx += 1
                self.show_current_patient()
                return
            selected_idx = user_input.get('selected_patient_idx')
            if isinstance(selected_idx, int) and 0 <= selected_idx < len(self.patient_list):
                if selected_idx != self.current_patient_idx:
                    print(f"Patient selection changed in dialog: {self.current_patient_idx} -> {selected_idx}")
                self.current_patient_idx = selected_idx
            # Resolve path/name based on (possibly updated) current_patient_idx
            patient_path, patient_name = self.patient_list[self.current_patient_idx]
            self.patient_index = self.current_patient_idx + 1
            try:
                root_abs = os.path.abspath(self.data_root)
                path_abs = os.path.abspath(patient_path)
                if os.path.commonpath([root_abs, path_abs]) == root_abs:
                    patient_id = os.path.relpath(path_abs, root_abs)
                else:
                    patient_id = path_abs
            except Exception:
                patient_id = str(patient_name)
            # Sync combo selection to reflect final choice
            if hasattr(self, 'patient_combo'):
                try:
                    self._suppress_combo_signal = True
                    self.patient_combo.setCurrentIndex(self.current_patient_idx)
                finally:
                    self._suppress_combo_signal = False
            current_mode = user_input.get('mode', self.default_mode)
            raw_method = user_input.get('method', self.default_method)
            current_method = None
            if isinstance(raw_method, str):
                rm = raw_method.lower()
                if rm.startswith('det') or rm == 'cls-det':
                    current_method = 'det'
                elif rm.startswith('seg'):
                    current_method = 'seg'
            if current_method is None:
                current_method = self.default_method
            current_preprocess = user_input.get('preprocess', False)
            use_double_viewer = user_input.get('use_double_viewer', False)
            double_path = user_input.get('double_path', None)
            det_model = user_input.get('det_model', 'sam2_det')
            seg_model = user_input.get('seg_model', 'sam2_seg')
            nnunet_model_path = user_input.get('nnunet_model_path', None)
            session_context = {
                'patient_id': patient_id,
                'patient_index': self.patient_index,
                'mode': current_mode,
                'method': current_method,
                'preprocess': current_preprocess,
                'model_version': getattr(self.args, 'version', None),
            }
            self.metrics.start_session(session_context)
            self.metrics.add_event('patient_session_started', **session_context)
            self.user_inputs[patient_id] = user_input
            try:
                data_load_start = time.time()
                current_pack = self.load_patient_data(
                    patient_path,
                    patient_name,
                    current_mode,
                    current_method,
                    current_preprocess,
                    patient_idx_override=self.current_patient_idx,
                )
                img_tensor = current_pack.get('image_3d')
                if img_tensor is None:
                    img_tensor = current_pack.get('images')
                slice_count = int(img_tensor.shape[0]) if (img_tensor is not None and hasattr(img_tensor, 'shape')) else None
                self.metrics.record_stage(
                    'data_load',
                    data_load_start,
                    time.time(),
                    patient_path=patient_path,
                    slice_count=slice_count,
                )
            except Exception as e:
                QMessageBox.critical(None, "Data Loading Error", f"Failed to load data for patient {patient_id}:\n{str(e)}")
                self.current_patient_idx += 1
                self.show_current_patient()
                return
            double_viewer = None
            if use_double_viewer and double_path:
                double_viewer = self.create_double_viewer(double_path, patient_id)
                if double_viewer:
                    self.double_viewers[patient_id] = double_viewer
            if current_mode == 'auto':
                results = auto_segmentation(
                    current_pack,
                    self.net,
                    self.device,
                    method=current_method,
                    det_model=det_model,
                    seg_model=seg_model,
                    nnunet_model_path=nnunet_model_path,
                    metrics=self.metrics,
                )
                if results:
                    result = results[0] if isinstance(results, list) else results
                    result_patient_id = result.get('patient_id', patient_id)
                    if current_method in ['det', 'cls-det']:
                        box_prompts = result.get('box_prompts', {})
                        point_prompts = {}
                    else:
                        raw_prompts = result.get('prompts', {})
                        box_prompts = {}
                        point_prompts = {}
                        for frame_idx, objs in raw_prompts.items():
                            for obj_id, prompt_data in objs.items():
                                if 'bboxes' in prompt_data and prompt_data['bboxes'] is not None:
                                    box_prompts.setdefault(frame_idx, {})[obj_id] = prompt_data['bboxes']
                                point_prompts.setdefault(frame_idx, {})[obj_id] = prompt_data
                    self.metrics.add_event('auto_session_ready', start_idx=result.get('start_idx'), end_idx=result.get('end_idx'))
                    self.current_gui = MedSAM2NapariGUIWithNavigation(
                        result['imgs'], result['video_segments'], self.net, self.device,
                        result_patient_id, box_prompts, point_prompts,
                        result.get('start_idx'), result.get('end_idx'), result.get('meta', {}), self,
                        metrics=self.metrics,
                    )
            elif current_mode == 'manual':
                original_patient_id = current_pack['meta']['patient']
                self.current_gui = ManualPromptNapariGUIWithNavigation(
                    current_pack['image_3d'], self.net, self.device, original_patient_id, current_pack['meta'], self,
                    metrics=self.metrics,
                )
        except Exception as e:
            print(f"Error showing patient GUI: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._sync_nav_panel()


class MedSAM2NapariGUIWithNavigation(MedSAM2NapariGUI):
    """Navigation-enabled auto GUI."""
    def __init__(self, imgs, video_segments, net, device, patient_id, box_prompts, point_prompts,
                 start_idx, end_idx, meta, navigation_manager=None, metrics=None):
        self.navigation_manager = navigation_manager
        super().__init__(imgs, video_segments, net, device, patient_id, box_prompts, point_prompts, start_idx, end_idx, meta, metrics=metrics)

    def _build_controls(self):
        super()._build_controls()
        if self.navigation_manager:
            layout = self.layout()
            patient_info_layout = QHBoxLayout()
            patient_info_layout.addWidget(QLabel(f'Patient {self.navigation_manager.patient_index}:'))
            patient_label = QLabel(self._format_patient_id(self.patient_id))
            patient_label.setStyleSheet("font-weight: bold; color: green;")
            patient_info_layout.addWidget(patient_label)
            layout.insertLayout(0, patient_info_layout)

            nav_layout = QHBoxLayout()
            back_btn = QPushButton('Back')
            back_btn.clicked.connect(self.prev_patient)
            back_btn.setStyleSheet("background-color: lightgray; font-weight: bold;")
            next_btn = QPushButton('Next Patient')
            next_btn.clicked.connect(self.next_patient)
            next_btn.setStyleSheet("background-color: lightblue; font-weight: bold;")
            pick_btn = QPushButton('Select Patient')
            pick_btn.clicked.connect(self.pick_patient)
            pick_btn.setStyleSheet("background-color: lightgreen; font-weight: bold;")
            close_btn = QPushButton('Close All')
            close_btn.clicked.connect(self.close_all)
            close_btn.setStyleSheet("background-color: lightcoral; font-weight: bold;")
            nav_layout.addWidget(back_btn)
            nav_layout.addWidget(next_btn)
            nav_layout.addWidget(pick_btn)
            nav_layout.addWidget(close_btn)
            layout.insertLayout(1, nav_layout)

    def next_patient(self):
        if self.navigation_manager:
            self.navigation_manager.next_patient()

    def prev_patient(self):
        if self.navigation_manager:
            self.navigation_manager.prev_patient()

    def pick_patient(self):
        if self.navigation_manager:
            self.navigation_manager.pick_patient()

    def close_all(self):
        if self.navigation_manager:
            self.navigation_manager.close_current_gui()


class ManualPromptNapariGUIWithNavigation(ManualPromptNapariGUI):
    """Navigation-enabled manual GUI."""
    def __init__(self, imgs, net, device, patient_id, meta, navigation_manager=None, metrics=None):
        self.navigation_manager = navigation_manager
        super().__init__(imgs, net, device, patient_id, meta, metrics=metrics)

    def _build_controls(self):
        super()._build_controls()
        if self.navigation_manager:
            layout = self.layout()
            patient_info_layout = QHBoxLayout()
            patient_info_layout.addWidget(QLabel(f'Patient {self.navigation_manager.patient_index}:'))
            patient_label = QLabel(self._format_patient_id(self.patient_id))
            patient_label.setStyleSheet("font-weight: bold; color: green;")
            patient_info_layout.addWidget(patient_label)
            layout.insertLayout(0, patient_info_layout)

            nav_layout = QHBoxLayout()
            back_btn = QPushButton('Back')
            back_btn.clicked.connect(self.prev_patient)
            back_btn.setStyleSheet("background-color: lightgray; font-weight: bold;")
            next_btn = QPushButton('Next Patient')
            next_btn.clicked.connect(self.next_patient)
            next_btn.setStyleSheet("background-color: lightblue; font-weight: bold;")
            pick_btn = QPushButton('Select Patient')
            pick_btn.clicked.connect(self.pick_patient)
            pick_btn.setStyleSheet("background-color: lightgreen; font-weight: bold;")
            close_btn = QPushButton('Close All')
            close_btn.clicked.connect(self.close_all)
            close_btn.setStyleSheet("background-color: lightcoral; font-weight: bold;")
            nav_layout.addWidget(back_btn)
            nav_layout.addWidget(next_btn)
            nav_layout.addWidget(pick_btn)
            nav_layout.addWidget(close_btn)
            layout.insertLayout(1, nav_layout)

    def next_patient(self):
        if self.navigation_manager:
            self.navigation_manager.next_patient()

    def prev_patient(self):
        if self.navigation_manager:
            self.navigation_manager.prev_patient()

    def pick_patient(self):
        if self.navigation_manager:
            self.navigation_manager.pick_patient()

    def close_all(self):
        if self.navigation_manager:
            self.navigation_manager.close_current_gui()


def run_napari_gui_with_navigation(data_root, net, device, args, default_mode='manual', default_method=None, metrics_recorder=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    nav_manager = PatientNavigationManager(data_root, net, device, default_mode, default_method, args, metrics_recorder)
    nav_manager.show_current_patient()
    napari.run()
