---
title: "Interactive Medical SAM2 GUI: Napari-based semi-automatic annotation for medical images"
tags:
  - Python
  - Medical-imaging
  - segmentation
  - Napari
authors:
  - name: "Woojae Hong"
    affiliation: 1
  - name: "Hyunggun Kim"
    affiliation: 1

affiliations:
  - name: "Department of Biomechatronic Engineering, Sungkyunkwan University, Suwon, Gyeonggi, Republic of Korea"
    index: 1
  - name: "Pituitary Center, Department of Neurosurgery, Seoul National University Hospital, Seoul National University College of Medicine, Seoul, Republic of Korea"
    index: 2
date: 24 December 2025
bibliography: paper.bib
---

# Summary

Interactive Medical SAM2 GUI is an open-source, Napari-based application that wraps the Medical-SAM2 model (https://github.com/ImprintLab/Medical-SAM2) to make interactive segmentation accessible to clinicians and researchers. The tool supports DICOM and NIfTI inputs, allows point and box prompts, propagates masks across slices, enables manual refinements (including brush-based edits), renders multi-object 3D volumes, reports per-object volumes, and exports masks aligned to the original geometry. The application targets researchers and translational teams who need rapid, auditable, and geometry-faithful segmentations without writing code.

# Statement of need

Medical segmentation models often require Python expertise and ad hoc preprocessing. Existing GUIs for foundation models such as 3D Slicer SAM extensions [@fedorov2012slicer], MONAI Label SAM demos [@diaz2022monailabel], and napari plug-ins allow semi-automatic interaction, but propagation is commonly scoped to limited slice ranges, depends on manual parameter tuning, and rarely tracks object identity or per-object volume through the stack. Conventional tools like ITK-SNAP [@yushkevich2006itksnap] and MITK [@wolf2005mitk] provide region-growing and manual modes yet still place most of the burden on slice-by-slice edits and do not integrate prompt-based propagation with volumetric QA. Clinicians also report friction from repeatedly configuring data paths and preprocessing settings for each study and from switching tools to view 3D renderings or volume metrics. This work couples Medical-SAM2’s video-style API [@imprintlab_medical_sam2] with point/box prompting, multi-object propagation, manual refinements, geometry-preserving saves, per-object volume rendering, and per-patient preprocessing/method selection in a single workflow that can step through datasets after a one-time root path selection. Real-time volume readouts and 3D volume rendering inside the same GUI reduce context switching when annotating tumors or organs while keeping outputs aligned to the original geometry.

![Figure 1. Interactive Medical SAM2 GUI running inside Napari with segmentation overlays and control panel visible.](images/image-2.png){#fig:gui width=80%}

# Usage and availability

The project is hosted at https://github.com/SKKU-IBE/SNU_MedSAM2_GUI. Installation instructions in the README describe creating a conda environment (python>=3.10, pyqt=5.15, pyopengl) and installing PyTorch matched to the available CUDA/CPU target, followed by `pip install -r requirements_medsam2_gui.txt`. The GUI loads DICOM folders or NIfTI files, walks patients sequentially, lets users choose preprocessing and prompting mode per patient, and then runs Medical-SAM2 to obtain initial masks before manual brush refinements and final saves. Box prompts generally perform best; fitting boxes tightly to objects and placing them on the first and last visible slices triggers automatic propagation through the intervening slices, while multi-object cases benefit from boxes on every slice where each object appears. Point prompts are suited to adding or removing local regions after boxes, and manual brush edits finalize the segmentation. Outputs include geometry-preserving masks in NIfTI format and per-object volume summaries with 3D renderings.

# Implementation

The application uses Napari for visualization and PyQt5 for controls. Data loading supports DICOM and NIfTI while preserving spacing, origin, and direction on save. Optional preprocessing includes N4 bias-field correction for MRI, percentile clipping, and per-slice normalization to 0–255 `uint8` for consistent display and model input. Prompting supports points and boxes with undo/redo history, and a manual edit mode allows brush-based mask creation or cleanup. Propagation uses the Medical-SAM2 video-style API to maintain object identities across slices, with real-time per-object volume updates and an embedded 3D volume rendering pane to verify shape without exporting. Patient navigation enumerates a root folder of studies, enabling per-patient preprocessing and mode selection after a single path selection. The codebase centers on `medsam_gui_v5_multi.py` (entry), `gui/navigation.py` (patient iteration), `gui/manual_gui.py` and `gui/auto_gui.py` (workflows), `medsam_gui_dataloader_v2.py` (I/O, preprocessing, geometry-safe saves), and `gui/segmentation.py` (bridge to Medical-SAM2). Dependencies include PyTorch [@paszke2019pytorch], Napari [@sofroniew2022napari], and SimpleITK [@lowekamp2013simpleitk].

# Conflict of interest

The authors declare no competing interests.

# Acknowledgements

This development was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MEST) (No.RS-2025-00517614). We thank the open-source communities behind PyTorch, Napari, and SimpleITK for making the underlying tooling accessible.

# References
