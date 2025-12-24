"""Navigation manager and navigation-enabled GUI wrappers."""
import os
import numpy as np
import napari
import torch
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QHBoxLayout, QPushButton, QMessageBox

from medsam_gui_dataloader_v2 import SNU3DMRI_MedSAM2Dataset, load_dicom_series, load_nii_image
from gui.segmentation import auto_segmentation
from gui.auto_gui import MedSAM2NapariGUI
from gui.manual_gui import ManualPromptNapariGUI


class PatientNavigationManager:
    """Navigation Manager for sequential patient data processing."""
    def __init__(self, data_root, net, device, default_mode='manual', default_method=None, args=None):
        self.data_root = data_root
        self.net = net
        self.device = device
        self.default_mode = default_mode
        self.default_method = default_method
        self.args = args
        self.current_gui = None
        self.patient_index = 0
        self.user_inputs = {}
        self.double_viewers = {}
        self.current_patient_idx = 0

        self.patient_list = []
        for p in sorted(os.listdir(data_root)):
            full_path = os.path.join(data_root, p)
            if os.path.isdir(full_path):
                nii_files = [f for f in os.listdir(full_path) if f.endswith('.nii') or f.endswith('.nii.gz')]
                if len(nii_files) > 0:
                    for nii_file in nii_files:
                        nii_path = os.path.join(full_path, nii_file)
                        patient_name = os.path.splitext(nii_file)[0]
                        self.patient_list.append((nii_path, patient_name))
                else:
                    self.patient_list.append((full_path, p))
            elif p.endswith('_image.nii') or p.endswith('_image.nii.gz'):
                patient_name = os.path.splitext(os.path.basename(p))[0]
                self.patient_list.append((full_path, patient_name))
        print(f"Found {len(self.patient_list)} patients to process")

    def load_patient_data(self, patient_path, patient_name, mode, method, preprocess):
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
        patient_idx = None
        for idx, (path, name) in enumerate(zip(temp_dataset.patient_paths, temp_dataset.patient_names)):
            if name == patient_name or os.path.basename(path) == os.path.basename(patient_path):
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
        dialog = PatientInputDialog(patient_id, self.patient_index)
        if dialog.exec_() == dialog.Accepted:
            settings = dialog.get_settings()
            print(f"Patient {patient_id} - Mode: {settings['mode']}, Method: {settings['method']}, Preprocess: {settings['preprocess']}")
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
            self.current_gui.viewer.close()
            self.current_gui = None

    def next_patient(self):
        self.close_current_gui()
        self.current_patient_idx += 1
        self.show_current_patient()

    def show_current_patient(self):
        try:
            if self.current_patient_idx >= len(self.patient_list):
                print("All patients have been processed.")
                return
            patient_path, patient_name = self.patient_list[self.current_patient_idx]
            self.patient_index = self.current_patient_idx + 1
            patient_id = str(patient_name)
            user_input = self.get_user_input_for_patient(patient_id)
            if user_input is None:
                self.current_patient_idx += 1
                self.show_current_patient()
                return
            current_mode = user_input.get('mode', self.default_mode)
            current_method = user_input.get('method', self.default_method)
            current_preprocess = user_input.get('preprocess', False)
            use_double_viewer = user_input.get('use_double_viewer', False)
            double_path = user_input.get('double_path', None)
            self.user_inputs[patient_id] = user_input
            try:
                current_pack = self.load_patient_data(patient_path, patient_name, current_mode, current_method, current_preprocess)
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
                results = auto_segmentation(current_pack, self.net, self.device, method=current_method)
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
                    self.current_gui = MedSAM2NapariGUIWithNavigation(
                        result['imgs'], result['video_segments'], self.net, self.device,
                        result_patient_id, box_prompts, point_prompts,
                        result.get('start_idx'), result.get('end_idx'), result.get('meta', {}), self
                    )
            elif current_mode == 'manual':
                original_patient_id = current_pack['meta']['patient']
                self.current_gui = ManualPromptNapariGUIWithNavigation(
                    current_pack['image_3d'], self.net, self.device, original_patient_id, current_pack['meta'], self
                )
        except Exception as e:
            print(f"Error showing patient GUI: {e}")
            import traceback
            traceback.print_exc()


class MedSAM2NapariGUIWithNavigation(MedSAM2NapariGUI):
    """Navigation-enabled auto GUI."""
    def __init__(self, imgs, video_segments, net, device, patient_id, box_prompts, point_prompts,
                 start_idx, end_idx, meta, navigation_manager=None):
        self.navigation_manager = navigation_manager
        super().__init__(imgs, video_segments, net, device, patient_id, box_prompts, point_prompts, start_idx, end_idx, meta)

    def _build_controls(self):
        super()._build_controls()
        if self.navigation_manager:
            layout = self.layout()
            patient_info_layout = QHBoxLayout()
            patient_info_layout.addWidget(QLabel(f'Patient {self.navigation_manager.patient_index}:'))
            patient_label = QLabel(str(self.patient_id))
            patient_label.setStyleSheet("font-weight: bold; color: green;")
            patient_info_layout.addWidget(patient_label)
            layout.insertLayout(0, patient_info_layout)

            nav_layout = QHBoxLayout()
            next_btn = QPushButton('Next Patient')
            next_btn.clicked.connect(self.next_patient)
            next_btn.setStyleSheet("background-color: lightblue; font-weight: bold;")
            close_btn = QPushButton('Close All')
            close_btn.clicked.connect(self.close_all)
            close_btn.setStyleSheet("background-color: lightcoral; font-weight: bold;")
            nav_layout.addWidget(next_btn)
            nav_layout.addWidget(close_btn)
            layout.insertLayout(1, nav_layout)

    def next_patient(self):
        if self.navigation_manager:
            self.navigation_manager.next_patient()

    def close_all(self):
        if self.navigation_manager:
            self.navigation_manager.close_current_gui()


class ManualPromptNapariGUIWithNavigation(ManualPromptNapariGUI):
    """Navigation-enabled manual GUI."""
    def __init__(self, imgs, net, device, patient_id, meta, navigation_manager=None):
        self.navigation_manager = navigation_manager
        super().__init__(imgs, net, device, patient_id, meta)

    def _build_controls(self):
        super()._build_controls()
        if self.navigation_manager:
            layout = self.layout()
            patient_info_layout = QHBoxLayout()
            patient_info_layout.addWidget(QLabel(f'Patient {self.navigation_manager.patient_index}:'))
            patient_label = QLabel(str(self.patient_id))
            patient_label.setStyleSheet("font-weight: bold; color: green;")
            patient_info_layout.addWidget(patient_label)
            layout.insertLayout(0, patient_info_layout)

            nav_layout = QHBoxLayout()
            next_btn = QPushButton('Next Patient')
            next_btn.clicked.connect(self.next_patient)
            next_btn.setStyleSheet("background-color: lightblue; font-weight: bold;")
            close_btn = QPushButton('Close All')
            close_btn.clicked.connect(self.close_all)
            close_btn.setStyleSheet("background-color: lightcoral; font-weight: bold;")
            nav_layout.addWidget(next_btn)
            nav_layout.addWidget(close_btn)
            layout.insertLayout(1, nav_layout)

    def next_patient(self):
        if self.navigation_manager:
            self.navigation_manager.next_patient()

    def close_all(self):
        if self.navigation_manager:
            self.navigation_manager.close_current_gui()


def run_napari_gui_with_navigation(data_root, net, device, args, default_mode='manual', default_method=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    nav_manager = PatientNavigationManager(data_root, net, device, default_mode, default_method, args)
    nav_manager.show_current_patient()
    napari.run()
