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
  - name: "Jong Ha Hwang"
    affiliation: 2
  - name: "Jiyong Chung"
    affiliation: 1
  - name: "Joongyeon Choi"
    affiliation: 1
  - name: "Hyunggun Kim"
    affiliation: 1
  - name: "Yong Hwy Kim"
    affiliation: 2

affiliations:
  - name: "Department of Biomechatronic Engineering, Sungkyunkwan University, Suwon, Gyeonggi, Republic of Korea"
    index: 1
  - name: "Pituitary Center, Department of Neurosurgery, Seoul National University Hospital, Seoul National University College of Medicine, Seoul, Republic of Korea"
    index: 2
date: 24 December 2025
bibliography: paper.bib
---

# Summary

Interactive Medical-SAM2 GUI is an open-source desktop application for semi-automatic annotation of 2D and 3D medical images (Figure 2a). Built on the Napari multi-dimensional viewer [@sofroniew2022napari], it integrates box/point prompting with SAM2-style propagation (treating a 3D scan as a “video” of slices) using Medical-SAM2 [@zhu2024medical] on top of SAM2 [@ravi2024sam2]. The tool is designed for clinician-friendly workflows: users can place DICOM series and/or NIfTI volumes under a single root folder (Figure 1a) and annotate cases sequentially, choosing to proceed or skip each case without repeatedly browsing individual patient files (Figure 1b). During saving, the tool reports per-object volumetry and provides 3D volume rendering to support rapid inspection and quantitative tracking (e.g., tumor burden) (Figure 2b).

![Interactive Medical-SAM2 GUI inside Napari.](./images/Figure1.png){#fig:gui width=85%}

![Interactive Medical-SAM2 GUI inside Napari.](./images/Figure2.png)

# Statement of need

Voxel-level annotation is essential for developing and validating medical imaging algorithms, yet manual labeling is slow and expensive, especially for 3D scans with hundreds of slices. Expert-friendly platforms such as ITK-SNAP [@yushkevich2006itksnap], 3D Slicer [@fedorov2012slicer], and MITK [@wolf2005mitk] provide robust visualization and classical semi-automatic segmentation tools, but producing consistent 3D labels at cohort scale still requires substantial manual work and careful data handling.

AI-assisted labeling frameworks have improved throughput by integrating model inference and active learning into annotation workflows. MONAI Label supports both local (3D Slicer) and web frontends and provides a framework for deploying labeling applications around AI models [@diazpinto2024monailabel]. Interactive refinement methods such as DeepEdit aim to reduce the number of user interactions needed to reach high-quality 3D segmentations by learning from simulated edits [@diazpinto2023deepedit].

Promptable foundation models have recently lowered the barrier to interactive segmentation. Segment Anything (SAM) [@kirillov2023sam] and medical adaptations such as MedSAM [@ma2024medsam] have motivated integrations into common annotation environments, including 3D Slicer extensions (e.g., MedSAMSlicer [@medsamslicer2023]) and Napari plugins (e.g., napari-sam [@naparisam2023]). Medical-SAM2 extends SAM2’s memory-based video segmentation paradigm by treating medical volumes as slice sequences, enabling propagation from sparse prompts across slices [@zhu2024medical; @ravi2024sam2]. However, many existing integrations emphasize per-slice interaction and do not provide a unified, cohort-oriented workflow that combines navigation, propagation, final correction, and quantitative export in a single local pipeline.

Interactive Medical-SAM2 GUI targets this practical gap by packaging Medical-SAM2 propagation into a local-first Napari workflow designed for efficient 3D annotation across many patient studies using only DICOM or NIfTI inputs.

# State of the field and differentiation

**General medical imaging workbenches.** 3D Slicer and MITK offer broad ecosystems of modules for segmentation, registration, and visualization [@fedorov2012slicer; @wolf2005mitk]. ITK-SNAP remains widely used for interactive 3D segmentation with user-guided active contour methods [@yushkevich2006itksnap]. These environments are powerful, but teams focused primarily on repetitive annotation may still need additional tooling to standardize navigation, prompt-based propagation, and quantitative export across many cases.

**Interactive ML labeling tools and general annotators.** ilastik provides interactive machine-learning workflows (segmentation/classification/tracking) that adapt to a task using sparse user annotations and can process up to 5D data [@berg2019ilastik]. In digital pathology, QuPath supports efficient annotation and scripting for large whole-slide images [@bankhead2017qupath]. Generic data-labeling platforms (e.g., CVAT [@cvat] and Label Studio [@labelstudio]) provide flexible web-based segmentation interfaces, but typically require additional engineering to handle DICOM/NIfTI conventions, geometry preservation, and radiology-style workflows.

**Promptable foundation-model integrations.** Community integrations such as MedSAMSlicer [@medsamslicer2023] and napari-sam [@naparisam2023] demonstrate strong demand for prompt-based labeling inside established viewers. Interactive Medical-SAM2 GUI differentiates itself by focusing on a single, clinician-oriented pipeline for **navigation → prompting/propagation → final correction → quantitative export**:

1. **Cohort navigation:** users provide one root path containing patient studies and annotate cases sequentially with explicit actions to proceed or skip, reducing manual file handling during routine labeling.
2. **Box-first prompting and propagation:** box prompts are the primary interaction for initializing objects. For single-object annotation, the user can place box prompts on the first and last slices where the object appears and run propagation to generate masks for intermediate slices using Medical-SAM2.
3. **Multi-object support with explicit control:** multiple objects can be annotated within the same volume. For multi-object scenarios, prompts can be provided on relevant slices for each object to maintain user control in complex cases.
4. **Point prompts for refinement:** point prompts can be added to refine predictions on a slice; in the current workflow, a box prompt defines the object on that slice and points provide additional guidance for small additions or corrections.
5. **Prompt-first correction workflow:** users typically obtain the best possible segmentation from prompts and propagation, and then perform a final manual correction step to “lock in” the label before saving. This workflow aligns with a propagation engine that is primarily driven by prompts and supports consistent, reproducible interaction.
6. **Quantitative export and visualization:** when saving masks, the tool computes per-object volumetry (e.g., for tumor volume monitoring) and offers 3D volume rendering to visually inspect the reconstructed shape. Saved masks preserve image geometry via SimpleITK [@lowekamp2013simpleitk].

# Implementation

The GUI is implemented in Python using Napari for multi-dimensional visualization [@sofroniew2022napari] and PyTorch for model execution [@paszke2019pytorch]. Medical-SAM2 [@zhu2024medical] provides SAM2-style memory-based propagation across slice sequences [@ravi2024sam2]. Image I/O, geometry preservation (spacing/origin/direction), and mask saving are handled with SimpleITK [@lowekamp2013simpleitk]. Optional MRI preprocessing includes N4 bias-field correction [@tustison2010n4]. The software is intended for research annotation workflows and does not provide clinical decision support.

# Conflict of interest

The authors declare no competing interests.

# Acknowledgements

This development was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MEST) (No.RS-2025-00517614). We thank the developers of Napari [@sofroniew2022napari], SimpleITK [@lowekamp2013simpleitk], SAM [@kirillov2023sam], SAM2 [@ravi2024sam2], and Medical-SAM2 [@zhu2024medical] for releasing open-source software and models.

# References