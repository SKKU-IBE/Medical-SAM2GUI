---
title: "Interactive Medical-SAM2 GUI: A Napari-based semi-automatic annotation tool for medical images"
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

![Figure 1. Interactive Medical-SAM2 GUI main view in Napari with image, prompt layers, and mask overlays.](./images/image-2.png)

Interactive Medical-SAM2 GUI is an open-source desktop application for semi-automatic annotation of 2D and 3D medical images (Figure 1). Built on the Napari multi-dimensional viewer [@sofroniew2022napari], it integrates box/point prompting with SAM2-style propagation (treating a 3D scan as a “video” of slices) using Medical-SAM2 [@zhu2024medical] on top of SAM2 [@ravi2024sam2]. The tool is designed for clinician-friendly workflows: users can place DICOM series and/or NIfTI volumes under a single root folder (Figure 2a) and annotate cases sequentially, choosing to proceed or skip each case without repeatedly browsing individual patient files (Figure 2b). During saving, the tool reports per-object volumetry and provides 3D volume rendering to support rapid inspection and quantitative tracking (e.g., tumor burden) (Figure 2c).

![Figure 2. (a) Root-folder selection for DICOM/NIfTI discovery; (b) patient-by-patient navigation with proceed/skip; (c) 3D volume rendering that reports per-object volumetry from the saved masks (voxel counts × spacing).](./images/Figure-2.png)

# Statement of need

Voxel-level annotation is essential for developing and validating medical imaging algorithms, yet manual labeling is slow and expensive, especially for 3D scans containing hundreds of slices. Expert-friendly platforms such as ITK-SNAP [@yushkevich2006itksnap], 3D Slicer [@fedorov2012slicer], and MITK [@wolf2005mitk] provide robust visualization and classical semi-automatic segmentation tools. However, it still requires substantial manual work and careful data handling to produce consistent 3D labels at cohort scale.

AI-assisted labeling frameworks have improved annotation efficiency by combining model inference and active learning strategies. MONAI Label supports both local (3D Slicer) and web frontends and provides a comprehensive framework for deploying around AI-driven annotation applications [@diazpinto2024monailabel]. While web-based labeling can be attractive for accessibility, clinical deployment is often constrained by institutional data governance and privacy requirements unless de-identification and secure hosting are rigorously validated, motivating local-first workflows for routine annotation. DeepEdit and similar interactive refinement approaches learn from simulated edits to reduce the number of user interactions necessary for generating accurate 3D segmentations [@diazpinto2023deepedit].

Promptable foundation models have recently lowered the barrier to interactive segmentation. Segment Anything (SAM) [@kirillov2023sam] and its medical adaptations such as MedSAM [@ma2024medsam] have motivated integrations into standard annotation platforms, including 3D Slicer extensions (e.g., MedSAMSlicer [@medsamslicer2023]) and Napari plugins (e.g., napari-sam [@naparisam2023]). Medical-SAM2 extends SAM2’s memory-based video segmentation approach to volumetric medcial imaging by treating 3D volumes as slice sequences, thereby enabling segmentation propagation from sparse annotations across multiple slices [@zhu2024medical; @ravi2024sam2]. However, existing integrations predominantly emphasize per-slice interaction and lack a unified, cohort-oriented workflow that seamlessly integrates navigation, propagation, interactive correction, and quantitative export within a single local pipeline.

Interactive Medical-SAM2 GUI addresses this practical limitation by integrating Medical-SAM2 propagation into a local-first Napari workflow designed for efficient 3D annotation across multiple patient studies using only standard DICOM or NIfTI inputs.

# State of the field and differentiation

**General medical imaging workbenches.** 
3D Slicer and MITK offer broad ecosystems of modules for segmentation, registration, and visualization [@fedorov2012slicer; @wolf2005mitk], while ITK-SNAP remains widely used for interactive 3D segmentation using user-guided active contour methods [@yushkevich2006itksnap]. Although these platforms are robust, repetitive annotation tasks may require additional tooling to ensure standardized navigation, prompt-based propagation, and consistent quantitative data export across large dataset.

**Interactive ML labeling tools and general annotators.** 
The interactive learning and segmentation toolkit (ilastik) provides interactive machine-learning workflows for segmentation, classification, and tracking that adapt to a task using sparse user annotations and supports data processing to 5D [@berg2019ilastik]. In digital pathology, QuPath supports efficient annotation and scripting for large whole-slide images [@bankhead2017qupath]. While generic data-labeling platforms (e.g., CVAT [@cvat] and Label Studio [@labelstudio]) provide flexible web-based segmentation interfaces, these platforms typically require additional engineering to handle medical imaging standards (DICOM/NIfTI), preseve geometry integrit, and support radiology-style workflows.

**Promptable foundation-model integrations.** 
Community integrations such as MedSAMSlicer [@medsamslicer2023] and napari-sam [@naparisam2023] have demonstrated strong demand for prompt-based labeling within established viewers. Interactive Medical-SAM2 GUI adopts an alternative strategy to provide clinician-oriented pipeline for **navigation → prompting/propagation → final correction → quantitative export**:

1. **Cohort navigation:** Users provide a single root path containing patient studies and annotate cases sequentially through explicit actions to proceed or skip, thereby minimizing manual file handling during routine labeling process.
2. **Box-first prompting and propagation:** Box prompts are the primary interaction for object initialization. For single-object annotation, the user can place box prompts on the first and last slices containing the target object, after which propagation generates masks for intermediate slices using Medical-SAM2.
3. **Multi-object support with explicit control:** Multiple objects can be annotated within the same volume. In multi-object scenarios, prompts can be provided on relevant slices for each object to maintain user control in complex cases.
4. **Point prompts for refinement:** Point prompts can be incorporated to refine slice-level predictions. In the current workflow, a box prompt initially defines the object on a given slice, while point prompts provide additional guidance for small additions or corrections.
5. **Prompt-first correction workflow:** Users typically obtain optimal segmentation through prompts and propagation, followed by final manual correction step to “lock in” the label prior to saving. This workflow aligns with a propagation framework that operates primarily through prompt-based interactions, ensuring consistency and reproducibility.
6. **Quantitative export and visualization:** Upon mask generation, the tool computes per-object volumetric measurements suitable for longitudinal tumor volume monitoring and provides 3D volume rendering capabilities for visual inspection of reconstructed anatomical structures. Saved masks preserve image geometry via SimpleITK [@lowekamp2013simpleitk].

# Implementation

The GUI is implemented in Python using Napari for multi-dimensional visualization [@sofroniew2022napari] and PyTorch for model execution [@paszke2019pytorch]. Medical-SAM2 [@zhu2024medical] provides SAM2-style memory-based propagation across slice sequences [@ravi2024sam2]. Image I/O, geometry preservation (spacing, origin, and direction), and mask saving are handled with SimpleITK [@lowekamp2013simpleitk]. Optional MRI preprocessing includes N4 bias-field correction [@tustison2010n4]. The software is designed exclusively for research annotation workflows and does not provide clinical decision support.

# Conflict of interest

The authors declare no competing interests.

# Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) through the Ministry of Science and ICT (No.RS-2025-00517614). We thank the developers of Napari [@sofroniew2022napari], SimpleITK [@lowekamp2013simpleitk], SAM [@kirillov2023sam], SAM2 [@ravi2024sam2], and Medical-SAM2 [@zhu2024medical] for releasing open-source software and models.

# References