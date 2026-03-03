"""Setup dialogs for the Medical SAM2 GUI."""

import os

from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)


class InitialSetupDialog(QDialog):
    """Initial setup dialog for Medical-SAM2 GUI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Medical-SAM2 - Initial Setup")
        self.setModal(True)
        self.resize(520, 420)

        self.mode = "manual"
        self.method = None
        self.preprocess = False
        self.show_initial_mask = False
        self.data_path = os.getcwd()
        self.version = "Medical_sam2"

        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout()

        title_label = QLabel("Interactive Medical-SAM2 GUI Setup")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title_label)

        # Mode selection
        mode_group = QGroupBox("Mode")
        mode_layout = QHBoxLayout()
        self.auto_radio = QRadioButton("Automatic")
        self.manual_radio = QRadioButton("Manual")
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.auto_radio)
        self.mode_group.addButton(self.manual_radio)
        if self.mode == "auto":
            self.auto_radio.setChecked(True)
        else:
            self.manual_radio.setChecked(True)
        mode_layout.addWidget(self.auto_radio)
        mode_layout.addWidget(self.manual_radio)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)

        # Method selection (only for auto)
        method_group = QGroupBox("Method (auto only)")
        method_layout = QHBoxLayout()
        self.cls_det_radio = QRadioButton("Detection")
        self.seg_radio = QRadioButton("Segmentation")
        self.method_group = QButtonGroup(self)
        self.method_group.addButton(self.cls_det_radio)
        self.method_group.addButton(self.seg_radio)
        if self.method == "det":
            self.cls_det_radio.setChecked(True)
        else:
            self.seg_radio.setChecked(True)
        method_layout.addWidget(self.cls_det_radio)
        method_layout.addWidget(self.seg_radio)
        method_group.setLayout(method_layout)
        main_layout.addWidget(method_group)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        self.preprocess_check = QCheckBox("Perform preprocessing")
        self.preprocess_check.setChecked(self.preprocess)
        self.show_mask_check = QCheckBox("Show initial mask (auto segmentation)")
        self.show_mask_check.setChecked(self.show_initial_mask)
        options_layout.addWidget(self.preprocess_check)
        options_layout.addWidget(self.show_mask_check)
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)

        # Data path
        path_group = QGroupBox("Data Path")
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit(self.data_path)
        self.path_input.setPlaceholderText("Select data folder")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_data_path)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(browse_button)
        path_group.setLayout(path_layout)
        main_layout.addWidget(path_group)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept_settings)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        self.mode_group.buttonClicked.connect(self.on_mode_changed)
        self.method_group.buttonClicked.connect(self.on_method_changed)

        self.setLayout(main_layout)
        self.on_mode_changed()

    def browse_data_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Data Folder", self.path_input.text())
        if path:
            self.path_input.setText(path)

    def on_mode_changed(self):
        is_auto = self.auto_radio.isChecked()
        for btn in self.method_group.buttons():
            btn.setEnabled(is_auto)
        self.show_mask_check.setEnabled(is_auto and self.seg_radio.isChecked())

    def on_method_changed(self):
        is_seg = self.seg_radio.isChecked()
        is_auto = self.auto_radio.isChecked()
        self.show_mask_check.setEnabled(is_auto and is_seg)

    def accept_settings(self):
        self.mode = "auto" if self.auto_radio.isChecked() else "manual"
        if self.mode == "auto":
            self.method = "det" if self.cls_det_radio.isChecked() else "seg"
        else:
            self.method = None

        self.preprocess = self.preprocess_check.isChecked()
        self.show_initial_mask = self.show_mask_check.isChecked()
        self.data_path = self.path_input.text().strip()
        if not self.data_path:
            QMessageBox.warning(self, "Warning", "Please enter a data path.")
            return

        self.version = "Medical_sam2"
        self.accept()

    def get_settings(self):
        return {
            "mode": self.mode,
            "method": self.method,
            "preprocess": self.preprocess,
            "show_initial_mask": self.show_initial_mask,
            "data_path": self.data_path,
            "version": "Medical_sam2",
        }


class PatientInputDialog(QDialog):
    """Dialog to input mode/method and optionally pick a patient from the list."""

    def __init__(
        self,
        patient_id,
        patient_index,
        defaults=None,
        parent=None,
        patient_choices=None,
        current_choice_idx=0,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Patient {patient_index}: {patient_id} - Select settings")
        self.setModal(True)
        self.resize(540, 580)

        defaults = defaults or {}
        self.mode = defaults.get("mode", "manual")
        self.method = defaults.get("method", None)
        self.use_double_viewer = defaults.get("use_double_viewer", False)
        self.double_path = defaults.get("double_path", None)
        self.preprocess = defaults.get("preprocess", False)
        self.det_model = defaults.get("det_model", "sam2_det")
        self.seg_model = defaults.get("seg_model", "sam2_seg")
        self.nnunet_model_path = defaults.get("nnunet_model_path", None)
        self.patient_choices = patient_choices or []
        self.selected_patient_idx = current_choice_idx if self.patient_choices else 0

        layout = QVBoxLayout()

        self.patient_info = QLabel(f"Patient {patient_index}: {patient_id}")
        self.patient_info.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")
        layout.addWidget(self.patient_info)
        layout.addWidget(QLabel(""))

        self.patient_combo = None
        if self.patient_choices:
            layout.addWidget(QLabel("Select patient:"))
            self.patient_combo = QComboBox()
            self.patient_combo.addItems(self.patient_choices)
            if 0 <= current_choice_idx < len(self.patient_choices):
                self.patient_combo.setCurrentIndex(current_choice_idx)
            self.patient_combo.currentIndexChanged.connect(self.on_patient_changed)
            layout.addWidget(self.patient_combo)
            layout.addWidget(QLabel(""))

        preprocess_label = QLabel("Preprocessing Setup:")
        preprocess_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(preprocess_label)

        self.preprocess_checkbox = QCheckBox(
            "Perform preprocessing (N4 bias correction + intensity normalization)"
        )
        self.preprocess_checkbox.setChecked(self.preprocess)
        layout.addWidget(self.preprocess_checkbox)

        preprocess_desc = QLabel(
            " Correct MRI inhomogeneity with N4 bias field correction\n"
            " Apply intensity clipping and normalization\n"
            " Preprocessed files are saved in preprocessed folder"
        )
        preprocess_desc.setStyleSheet("font-size: 9px; color: gray; margin-left: 20px;")
        layout.addWidget(preprocess_desc)
        layout.addWidget(QLabel(""))

        mode_label = QLabel("Mode Selection:")
        mode_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["auto", "manual"])
        self.mode_combo.setCurrentText(self.mode)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        layout.addWidget(self.mode_combo)

        self.method_label = QLabel("Method Selection:")
        self.method_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.method_label)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Detection", "Segmentation"])
        display_method = "Segmentation"
        if isinstance(self.method, str):
            low = self.method.lower()
            if low.startswith("det"):
                display_method = "Detection"
            elif low.startswith("seg"):
                display_method = "Segmentation"
        self.method_combo.setCurrentText(display_method)
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        layout.addWidget(self.method_combo)

        self.det_model_label = QLabel("Detection Model:")
        self.det_model_label.setStyleSheet("font-weight: bold;")
        self.det_model_combo = QComboBox()
        self.det_model_combo.addItems(["sam2_det", "custom"])
        self.det_model_combo.setCurrentText(self.det_model if self.det_model in ["sam2_det", "custom"] else "custom")
        self.det_model_combo.currentTextChanged.connect(self.on_det_model_changed)
        self.det_model_custom_input = QLineEdit()
        self.det_model_custom_input.setPlaceholderText("Enter custom detection model name")
        if self.det_model not in ["sam2_det", "custom"]:
            self.det_model_custom_input.setText(self.det_model)
        layout.addWidget(self.det_model_label)
        layout.addWidget(self.det_model_combo)
        layout.addWidget(self.det_model_custom_input)

        self.seg_model_label = QLabel("Segmentation Model:")
        self.seg_model_label.setStyleSheet("font-weight: bold;")
        self.seg_model_combo = QComboBox()
        self.seg_model_combo.addItems(["sam2_seg", "nnUNetv2"])
        self.seg_model_combo.setCurrentText(self.seg_model if self.seg_model in ["sam2_seg", "nnUNetv2"] else "sam2_seg")
        self.seg_model_combo.currentTextChanged.connect(lambda _: self.on_method_changed(self.method_combo.currentText()))
        layout.addWidget(self.seg_model_label)
        layout.addWidget(self.seg_model_combo)

        self.nnunet_path_label = QLabel("nnUNetv2 model folder:")
        self.nnunet_path_input = QLineEdit()
        self.nnunet_path_input.setPlaceholderText("/path/to/nnunetv2_model")
        if self.nnunet_model_path:
            self.nnunet_path_input.setText(self.nnunet_model_path)
        self.nnunet_path_browse = QPushButton("Browse")
        self.nnunet_path_browse.clicked.connect(self.browse_nnunet_path)
        nnunet_path_layout = QHBoxLayout()
        nnunet_path_layout.addWidget(self.nnunet_path_label)
        nnunet_path_layout.addWidget(self.nnunet_path_input)
        nnunet_path_layout.addWidget(self.nnunet_path_browse)
        layout.addLayout(nnunet_path_layout)

        layout.addWidget(QLabel(""))
        double_viewer_label = QLabel("Double Viewer Setup:")
        double_viewer_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(double_viewer_label)

        self.double_viewer_checkbox = QCheckBox("Use Double Viewer")
        self.double_viewer_checkbox.toggled.connect(self.on_double_viewer_toggled)
        self.double_viewer_checkbox.setChecked(self.use_double_viewer)
        layout.addWidget(self.double_viewer_checkbox)

        path_layout = QHBoxLayout()
        self.path_label = QLabel("Path:")
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Select file/folder path for double viewer")
        if self.double_path:
            self.path_input.setText(self.double_path)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_path)
        path_layout.addWidget(self.path_label)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_button)
        layout.addLayout(path_layout)

        layout.addWidget(QLabel(""))
        desc_text = QLabel(
            """
    Mode Description:
    - auto: Perform automatic segmentation
    - manual: Manual prompt input

    Method Description (for auto mode):
    - Detection: Detection
    - Segmentation: Segmentation with prompts

    Preprocessing:
    - N4 bias field correction to correct MRI signal inhomogeneity
    - Intensity clipping (0.5%~99.5%) and Z-score normalization
    - Preprocessing results are automatically saved in preprocessed folder

    Double Viewer:
    - Display another image simultaneously in an additional viewer
    - Can select DICOM folder or NIfTI file
                """
        )
        desc_text.setStyleSheet("font-size: 10px; color: gray;")
        layout.addWidget(desc_text)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        ok_button.setStyleSheet("background-color: lightgreen; font-weight: bold;")

        skip_button = QPushButton("Skip This Patient")
        skip_button.clicked.connect(self.reject)
        skip_button.setStyleSheet("background-color: lightcoral; font-weight: bold;")

        button_layout.addWidget(ok_button)
        button_layout.addWidget(skip_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Initialize control states
        self.on_mode_changed(self.mode_combo.currentText())
        self.on_det_model_changed(self.det_model_combo.currentText())
        self.on_double_viewer_toggled(self.use_double_viewer)
        if self.patient_choices:
            self.on_patient_changed(self.patient_combo.currentIndex())

    def on_mode_changed(self, mode):
        is_manual = mode == "manual"
        self.method_combo.setEnabled(not is_manual)
        self.method_label.setEnabled(not is_manual)
        self.det_model_combo.setEnabled(not is_manual and self.method_combo.currentText() == "Detection")
        self.det_model_label.setEnabled(not is_manual and self.method_combo.currentText() == "Detection")
        self.det_model_custom_input.setEnabled(
            not is_manual
            and self.method_combo.currentText() == "Detection"
            and self.det_model_combo.currentText() == "custom"
        )
        self.seg_model_combo.setEnabled(not is_manual and self.method_combo.currentText() == "Segmentation")
        self.seg_model_label.setEnabled(not is_manual and self.method_combo.currentText() == "Segmentation")
        if is_manual:
            self.method = None
            self.nnunet_path_input.setEnabled(False)
            self.nnunet_path_label.setEnabled(False)
            self.nnunet_path_browse.setEnabled(False)
        else:
            self.method = self.method_combo.currentText()
            self.on_method_changed(self.method_combo.currentText())

    def on_method_changed(self, method):
        is_det = method == "Detection"
        is_seg = method == "Segmentation"
        self.det_model_combo.setEnabled(is_det and self.mode_combo.currentText() == "auto")
        self.det_model_label.setEnabled(is_det and self.mode_combo.currentText() == "auto")
        self.det_model_custom_input.setEnabled(is_det and self.det_model_combo.currentText() == "custom" and self.mode_combo.currentText() == "auto")
        self.seg_model_combo.setEnabled(is_seg and self.mode_combo.currentText() == "auto")
        self.seg_model_label.setEnabled(is_seg and self.mode_combo.currentText() == "auto")
        need_nnunet = is_seg and self.seg_model_combo.currentText() == "nnUNetv2" and self.mode_combo.currentText() == "auto"
        self.nnunet_path_input.setEnabled(need_nnunet)
        self.nnunet_path_label.setEnabled(need_nnunet)
        self.nnunet_path_browse.setEnabled(need_nnunet)
        if not need_nnunet:
            self.nnunet_model_path = None

    def on_det_model_changed(self, det_model):
        need_custom = det_model == "custom" and self.mode_combo.currentText() == "auto" and self.method_combo.currentText() == "Detection"
        self.det_model_custom_input.setEnabled(need_custom)
        if not need_custom:
            self.det_model_custom_input.clear()

    def on_double_viewer_toggled(self, checked):
        self.path_label.setEnabled(checked)
        self.path_input.setEnabled(checked)
        self.browse_button.setEnabled(checked)
        self.use_double_viewer = checked
        if not checked:
            self.path_input.clear()
            self.double_path = None

    def browse_path(self):
        options = ["Select DICOM folder", "Select NIfTI file (.nii/.nii.gz)"]
        option, ok = QInputDialog.getItem(
            self,
            "Select Path Type",
            "Select data type for double viewer:",
            options,
            0,
            False,
        )
        if not ok:
            return

        if option == options[0]:
            folder = QFileDialog.getExistingDirectory(
                self,
                "Select DICOM folder",
                "",
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
            )
            if folder:
                self.path_input.setText(folder)
                self.double_path = folder
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select NIfTI file",
                "",
                "NIfTI Files (*.nii *.nii.gz);;All Files (*)",
            )
            if file_path:
                self.path_input.setText(file_path)
                self.double_path = file_path

    def browse_nnunet_path(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select nnUNetv2 model folder",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if folder:
            self.nnunet_path_input.setText(folder)
            self.nnunet_model_path = folder

    def on_patient_changed(self, idx):
        self.selected_patient_idx = idx
        try:
            text = self.patient_combo.itemText(idx)
            self.patient_info.setText(f"Patient {idx+1}: {text}")
        except Exception:
            pass

    def get_settings(self):
        mode_value = self.mode_combo.currentText()
        self.mode = mode_value
        if mode_value == "manual":
            self.method = None
        else:
            self.method = self.method_combo.currentText()

        det_model_choice = self.det_model_combo.currentText()
        if det_model_choice == "custom":
            custom_name = self.det_model_custom_input.text().strip()
            self.det_model = custom_name if custom_name else "custom"
        else:
            self.det_model = det_model_choice

        self.seg_model = self.seg_model_combo.currentText()
        if self.seg_model == "nnUNetv2":
            self.nnunet_model_path = self.nnunet_path_input.text().strip() or None
        else:
            self.nnunet_model_path = None

        self.preprocess = self.preprocess_checkbox.isChecked()
        self.use_double_viewer = self.double_viewer_checkbox.isChecked()
        if self.use_double_viewer:
            self.double_path = self.path_input.text().strip() or None
        else:
            self.double_path = None

        selected_idx = self.selected_patient_idx
        if self.patient_combo is not None:
            selected_idx = self.patient_combo.currentIndex()

        return {
            "mode": self.mode,
            "method": self.method,
            "preprocess": self.preprocess,
            "show_initial_mask": getattr(self, "show_initial_mask", False),
            "data_path": getattr(self, "data_path", None),
            "version": "Medical_sam2",
            "use_double_viewer": self.use_double_viewer,
            "double_path": self.double_path,
            "det_model": self.det_model,
            "seg_model": self.seg_model,
            "nnunet_model_path": self.nnunet_model_path,
            "selected_patient_idx": selected_idx,
        }
