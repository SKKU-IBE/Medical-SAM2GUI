---
title: "Interactive Medical SAM2 GUI: Napari-based prompting and propagation for medical images"
tags:
  - Python
  - 
authors:
  - name: "Woojae Hong"
    affiliation: 1
  - name: "Seong-min Kim"
    affiliation: 1
  - name: "Jaewoo Kim"
    affiliation: 1
  - name: "Joongyeon Choi"
    affiliation: 1
  - name: "Minsung Ko"
    affiliation: 1
  - name: "Jiyong Chung"
    affiliation: 1
  - name: "Hyunggun Kim"
    affiliation: 1
  - name: "Jong Ha Hwang"
    affiliation: 2
  - name: "Yong Hwy Kim"
    affiliation: 2

affiliations:
  - name: "Department of Biomechatronic Engineering, Sungkyunkwan University, Suwon, Gyeonggi, Republic of Korea"
    index: 1
  - name: "Pituitary Center, Department of Neurosurgery, Seoul National University Hospital, Seoul National University College of Medicine, Seoul, Republic of Korea"
    index: 2
date: 2025-12-24
bibliography: paper.bib
---

# Summary

Interactive Medical SAM2 GUI is an open-source, Napari-based application that wraps the Medical-SAM2 model (https://github.com/ImprintLab/Medical-SAM2) to make interactive segmentation accessible to clinicians and researchers. The tool supports DICOM and NIfTI inputs, allows point and box prompts, propagates masks across slices, enables manual refinements (including brush-based edits), renders multi-object 3D volumes, reports per-object volumes, and exports masks aligned to the original geometry. The application targets researchers and translational teams who need rapid, auditable, and geometry-faithful segmentations without writing code.

# Statement of need

Medical segmentation models often require Python expertise and ad hoc preprocessing. Existing GUIs for foundation models (e.g., napari-SAM plug-ins, 3D Slicer SAM extensions, MONAI Label SAM demos) typically operate slice-by-slice, do not maintain geometry-aligned multi-object propagation, or omit integrated volume QA. Conventional tools like ITK-SNAP/MITK offer semi-automatic modes but lack prompt-based propagation across slices and per-object volume reporting. In contrast, this tool couples Medical-SAM2’s video-style API with point/box prompts, multi-object propagation, brush-based refinements, geometry-preserving saves, per-object volume renders, and per-patient preprocessing/method selection in a single workflow. The goal is to lower the barrier for clinical and research teams who need reproducible, geometry-faithful masks without custom coding, while supporting batch navigation over DICOM/NIfTI folders.

# Design and implementation

The GUI is built on Napari for visualization and PyQt5 for controls, exposing Medical-SAM2 inference through a minimal interface. Data loading supports folders of DICOM series and NIfTI volumes; geometry (spacing, origin, direction) is preserved on save. Preprocessing includes bias-field correction (optional for MRI), percentile clipping, and per-slice normalization to 0–255 `uint8` for consistent display and model input. Prompting supports points and boxes with undo/redo history; a manual edit mode leverages Napari painting for label cleanup or full brush-based mask creation when the model output is insufficient. Propagation reuses the Medical-SAM2 video-style API to maintain object identities across slices. Users can browse a directory containing multiple patients (DICOM folders or NIfTI files), process them sequentially, and choose per-patient preprocessing and prompting methods. 3D volume rendering and per-object volume reporting provide quick quality checks. Planned extensions include inserting upstream detection/segmentation models that auto-generate prompts before refinement in the GUI.

The codebase is structured as follows: `medsam_gui_v5_multi.py` bootstraps the application; `gui/navigation.py` manages patient-level iteration; `gui/manual_gui.py` and `gui/auto_gui.py` implement manual and auto prompting workflows; `medsam_gui_dataloader_v2.py` handles I/O, preprocessing, and geometry-safe saves; `gui/segmentation.py` bridges prompts to the Medical-SAM2 network. Dependencies are intentionally lightweight: PyTorch for inference [@paszke2019pytorch], Napari for visualization [@sofroniew2022napari], and SimpleITK for medical image I/O [@lowekamp2013simpleitk].

# Functionality

- **Prompting and propagation:** point and box prompts with per-slice undo/redo; propagation uses Medical-SAM2 to maintain object IDs across frames. Boxes typically yield higher fidelity than points, and users are encouraged to fit boxes tightly around target objects.
- **Manual refinement:** Napari paint/erase and box editing; manual mode disables box creation callbacks to avoid accidental prompts.
- **Multi-object handling:** connected-component analysis to initialize IDs; per-object volume computation; 3D volume rendering for visual QA.
- **Geometry preservation:** masks are saved with original spacing, origin, and direction; supports DICOM series and NIfTI inputs.
- **Intensity handling:** per-slice percentile clip (0.5/99.5), normalize, scale to 0–255 `uint8`; tensors are cast to `float32` but retain the 0–255 range when entering the model.
- **Preprocessing for raw MRI:** optional N4 bias field correction plus intensity normalization when raw inputs need harmonization.
- **Patient navigation:** browse a root folder, iterate through patients in order, and set per-patient preprocess/method choices for convenience.

# Quality control

- Sanity checks for DICOM properties (window center/width, rescale slope/intercept) and geometry warnings.
- Pre/post preprocess stats (range/mean/std) for reproducibility.
- Deterministic prompt history with undo/redo and explicit manual-edit toggles.
- Volume rendering and per-object volume summaries for quick visual validation.

# Availability and installation

The project is hosted on GitHub: https://github.com/SKKU-IBE/SNU_MedSAM2_GUI. A conda recipe is provided in `README.md`: create the environment with `python>=3.10`, `pyqt=5.15`, `pyopengl`, then install the appropriate PyTorch wheel for your CUDA/CPU target, followed by `pip install -r requirements_medsam2_gui.txt`. The same steps work on Linux and Windows; Linux users should match the PyTorch CUDA build to the installed NVIDIA driver.

# Competing interests

The authors declare no competing interests.

# Funding

This development was supported by the Natiional Research Foundation of Korea (NRF) grant funded by the Korea government (MEST) (No.RS-2025-00517614).

# Reuse potential and impact

The GUI enables non-programmers to harness Medical-SAM2 for interactive segmentation with geometry fidelity. It is suitable for dataset creation, rapid QA of automatic segmentations, and small-batch clinical research where auditability and export fidelity matter. Because it is built on standard Python libraries and ships with permissive licensing, the tool can be adapted for site-specific protocols, alternative backbones, or additional measurement/reporting plugins. Future work will add front-end detection/segmentation modules that auto-propose prompts, followed by the same interactive refinement and brushing workflow described here.

# Acknowledgements

We thank the open-source communities behind PyTorch, Napari, and SimpleITK for making the underlying tooling accessible. Replace the placeholders above with the project contributors and institutional affiliations for submission.

# References
