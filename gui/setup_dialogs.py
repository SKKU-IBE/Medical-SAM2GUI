"""Setup dialogs for the Medical SAM2 GUI."""

import os

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QButtonGroup,
)


class InitialSetupDialog(QDialog):
    """Initial setup dialog for Medical SAM2 GUI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Medical SAM2 - Initial Setup")
        self.setModal(True)
        self.resize(500, 400)

        self.mode = "manual"
        self.method = None
        self.preprocess = False
        self.show_initial_mask = False
        self.data_path = os.getcwd()
        self.version = "Medical_sam2"

        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout()

        title_label = QLabel("Interactive Medical SAM2 GUI Setup")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title_label)

        mode_group = QGroupBox("Work Mode")
        mode_layout = QVBoxLayout()

        self.mode_group = QButtonGroup()
        self.auto_radio = QRadioButton("Auto (Automatic Segmentation)")
        self.manual_radio = QRadioButton("Manual (Manual Prompts)")
        self.manual_radio.setChecked(True)

        self.mode_group.addButton(self.auto_radio, 0)
        self.mode_group.addButton(self.manual_radio, 1)

        mode_layout.addWidget(self.auto_radio)
        mode_layout.addWidget(self.manual_radio)
        mode_group.setLayout(mode_layout)

        method_group = QGroupBox("Method (For Auto Mode)")
        method_layout = QVBoxLayout()

        self.method_group = QButtonGroup()
        self.cls_det_radio = QRadioButton("det (Detection)")
        self.seg_radio = QRadioButton("seg (Segmentation)")
        self.seg_radio.setChecked(True)

        self.method_group.addButton(self.cls_det_radio, 0)
        self.method_group.addButton(self.seg_radio, 1)

        method_layout.addWidget(self.cls_det_radio)
        method_layout.addWidget(self.seg_radio)
        method_group.setLayout(method_layout)

        options_group = QGroupBox("Additional Options")
        options_layout = QVBoxLayout()

        self.preprocess_check = QCheckBox("Perform Preprocessing")
        self.show_mask_check = QCheckBox("Show Initial Segmentation Mask (Auto Seg Mode)")

        options_layout.addWidget(self.preprocess_check)
        options_layout.addWidget(self.show_mask_check)
        options_group.setLayout(options_layout)

        path_group = QGroupBox("Data Path")
        path_layout = QVBoxLayout()

        path_input_layout = QHBoxLayout()
        self.path_input = QLineEdit(self.data_path)
        path_browse_btn = QPushButton("Browse...")
        path_browse_btn.clicked.connect(self.browse_data_path)

        path_input_layout.addWidget(self.path_input)
        path_input_layout.addWidget(path_browse_btn)

        path_layout.addLayout(path_input_layout)
        path_group.setLayout(path_layout)

        main_layout.addWidget(mode_group)
        main_layout.addWidget(method_group)
        main_layout.addWidget(options_group)
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
        for i in range(self.method_group.buttons().__len__()):
            self.method_group.buttons()[i].setEnabled(is_auto)
        self.show_mask_check.setEnabled(is_auto and self.seg_radio.isChecked())

    def on_method_changed(self):
        is_seg = self.seg_radio.isChecked()
        is_auto = self.auto_radio.isChecked()
        self.show_mask_check.setEnabled(is_auto and is_seg)

    def accept_settings(self):
        if self.auto_radio.isChecked():
            self.mode = "auto"
        else:
            self.mode = "manual"

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
    """Dialog to input mode and method for each patient."""

    def __init__(self, patient_id, patient_index, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Patient {patient_index}: {patient_id} - Select settings")
        self.setModal(True)
        self.resize(500, 500)

        self.mode = "manual"
        self.method = None
        self.use_double_viewer = False
        self.double_path = None
        self.preprocess = False

        layout = QVBoxLayout()

        patient_info = QLabel(f"Patient {patient_index}: {patient_id}")
        patient_info.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")
        layout.addWidget(patient_info)
        layout.addWidget(QLabel(""))

        preprocess_label = QLabel("Preprocessing Setup:")
        preprocess_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(preprocess_label)

        self.preprocess_checkbox = QCheckBox(
            "Perform preprocessing (N4 bias correction + intensity normalization)"
        )
        self.preprocess_checkbox.setChecked(False)
        layout.addWidget(self.preprocess_checkbox)

        preprocess_desc = QLabel(
            "• Correct MRI inhomogeneity with N4 bias field correction\n"
            "• Apply intensity clipping and normalization\n"
            "• Preprocessed files are saved in preprocessed folder"
        )
        preprocess_desc.setStyleSheet("font-size: 9px; color: gray; margin-left: 20px;")
        layout.addWidget(preprocess_desc)
        layout.addWidget(QLabel(""))

        mode_label = QLabel("Mode Selection:")
        mode_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["auto", "manual"])
        self.mode_combo.setCurrentText("manual")
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        layout.addWidget(self.mode_combo)

        self.method_label = QLabel("Method Selection:")
        self.method_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.method_label)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["det", "seg"])
        self.method_combo.setCurrentText("seg")
        layout.addWidget(self.method_combo)

        self.on_mode_changed("manual")
        layout.addWidget(QLabel(""))

        double_viewer_label = QLabel("Double Viewer Setup:")
        double_viewer_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(double_viewer_label)

        self.double_viewer_checkbox = QCheckBox("Use Double Viewer")
        self.double_viewer_checkbox.toggled.connect(self.on_double_viewer_toggled)
        layout.addWidget(self.double_viewer_checkbox)

        path_layout = QHBoxLayout()
        self.path_label = QLabel("Path:")
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Select file/folder path for double viewer")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_path)

        path_layout.addWidget(self.path_label)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_button)
        layout.addLayout(path_layout)

        self.on_double_viewer_toggled(False)
        layout.addWidget(QLabel(""))

        desc_text = QLabel(
            """
Mode Description:
• auto: Perform automatic segmentation
• manual: Manual prompt input

Method Description (for auto mode):
• det: Detection
• seg: Segmentation with prompts

Preprocessing:
• N4 bias field correction to correct MRI signal inhomogeneity
• Intensity clipping (0.5%~99.5%) and Z-score normalization
• Preprocessing results are automatically saved in preprocessed folder

Double Viewer:
• Display another image simultaneously in an additional viewer
• Can select DICOM folder or NIfTI file
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

    def on_mode_changed(self, mode):
        if mode == "manual":
            self.method_combo.setEnabled(False)
            self.method_label.setEnabled(False)
            self.method = None
        else:
            self.method_combo.setEnabled(True)
            self.method_label.setEnabled(True)
            self.method = self.method_combo.currentText()

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

    def get_settings(self):
        self.mode = self.mode_combo.currentText()
        if self.mode == "manual":
            self.method = None
        else:
            self.method = self.method_combo.currentText()

        self.preprocess = self.preprocess_checkbox.isChecked()
        self.use_double_viewer = self.double_viewer_checkbox.isChecked()
        if self.use_double_viewer:
            self.double_path = self.path_input.text().strip() or None
        else:
            self.double_path = None

        return {
            "mode": self.mode,
            "method": self.method,
            "preprocess": self.preprocess,
            "use_double_viewer": self.use_double_viewer,
            "double_path": self.double_path,
        }
