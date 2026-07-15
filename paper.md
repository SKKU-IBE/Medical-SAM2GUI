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
    corresponding: true
  - name: "Yong Hwy Kim"
    affiliation: 2
    corresponding: true

affiliations:
  - name: "Department of Biomechatronic Engineering, Sungkyunkwan University, Suwon, Gyeonggi, Republic of Korea"
    index: 1
  - name: "Pituitary Center, Department of Neurosurgery, Seoul National University Hospital, Seoul National University College of Medicine, Seoul, Republic of Korea"
    index: 2
date: 14 July 2026
bibliography: paper.bib
---

# Summary

![Interactive Medical-SAM2 GUI during resumed manual annotation, showing a public FLAIR volume, imported multi-label mask, current workflow controls, and source-grid volumetry. The displayed case is from the TCIA UCSD-PTGBM BraTS-GLI 2024 Test Data [@gagnon2026ucsdptgbm]. \label{fig:manual-workflow}](./images/manual-mask-resume.png)

Interactive Medical-SAM2 GUI is an open-source desktop application for semi-automatic annotation of 3D medical image volumes (\autoref{fig:manual-workflow}). Built on the Napari multi-dimensional viewer [@sofroniew2022napari], it integrates box/point prompting with SAM2-style propagation (treating a 3D scan as a “video” of slices) using Medical-SAM2 [@zhu2024medical] on top of SAM2 [@ravi2024sam2]. The tool is designed for clinician-friendly workflows: users can place DICOM series and/or NIfTI volumes under a single root folder and annotate automatically discovered cases sequentially, choosing to proceed or skip each case without repeatedly browsing individual patient files. Existing multi-label masks can be reloaded for continued annotation, and manual corrections are synchronized to the original image grid before export. During editing and saving, the tool reports per-object volumetry and provides optional 3D volume rendering to support rapid inspection and quantitative tracking (e.g., tumor burden).

# Statement of need

Voxel-level annotation is essential for developing and validating medical imaging algorithms, yet manual labeling is slow and expensive, especially for 3D scans with hundreds of slices. Expert-friendly platforms such as ITK-SNAP [@yushkevich2006itksnap], 3D Slicer [@fedorov2012slicer], and MITK [@wolf2005mitk] provide robust visualization and classical semi-automatic segmentation tools, but producing consistent 3D labels at cohort scale still requires substantial manual work and careful data handling.

AI-assisted labeling frameworks have improved throughput by integrating model inference and active learning into annotation workflows. MONAI Label supports both local (3D Slicer) and web frontends and provides a framework for deploying labeling applications around AI models [@diazpinto2024monailabel]. While web-based labeling can be attractive for accessibility, clinical deployment is often constrained by institutional data governance and privacy requirements unless de-identification and secure hosting are rigorously validated, motivating local-first workflows for routine annotation. Interactive refinement methods such as DeepEdit aim to reduce the number of user interactions needed to reach high-quality 3D segmentations by learning from simulated edits [@diazpinto2023deepedit].

Promptable foundation models have recently lowered the barrier to interactive segmentation. Segment Anything (SAM) [@kirillov2023sam] and medical adaptations such as MedSAM [@ma2024medsam] have motivated integrations into common annotation environments, including 3D Slicer extensions (e.g., MedSAMSlicer [@medsamslicer2023]) and Napari plugins (e.g., napari-sam [@naparisam2023]). Medical-SAM2 extends SAM2’s memory-based video segmentation paradigm by treating medical volumes as slice sequences, enabling propagation from sparse prompts across slices [@zhu2024medical; @ravi2024sam2]. However, many existing integrations emphasize per-slice interaction and do not provide a unified, cohort-oriented workflow that combines navigation, propagation, final correction, and quantitative export in a single local pipeline.

Interactive Medical-SAM2 GUI targets this practical gap by packaging Medical-SAM2 propagation into a local-first Napari workflow designed for efficient 3D annotation across many patient studies using only DICOM or NIfTI inputs.

# State of the field and differentiation

**General medical imaging workbenches.** 
3D Slicer and MITK offer broad ecosystems of modules for segmentation, registration, and visualization [@fedorov2012slicer; @wolf2005mitk]. ITK-SNAP remains widely used for interactive 3D segmentation with user-guided active contour methods [@yushkevich2006itksnap]. These environments are powerful, but repetitive annotation may still need additional tooling to standardize navigation, prompt-based propagation, correction, and quantitative export across many cases.

**Interactive ML labeling tools and general annotators.** 
ilastik provides interactive machine-learning workflows (segmentation/classification/tracking) that adapt to a task using sparse user annotations and can process up to 5D data [@berg2019ilastik]. In digital pathology, QuPath supports efficient annotation and scripting for large whole-slide images [@bankhead2017qupath]. Generic data-labeling platforms (e.g., CVAT [@cvat] and Label Studio [@labelstudio]) provide flexible web-based segmentation interfaces, but typically require additional engineering to handle DICOM/NIfTI conventions, geometry preservation, and radiology-style workflows.

**Promptable foundation-model integrations.** 
Community integrations such as MedSAMSlicer [@medsamslicer2023] and napari-sam [@naparisam2023] demonstrate strong demand for prompt-based labeling inside established viewers. Interactive Medical-SAM2 GUI differentiates itself by focusing on a single, clinician-oriented pipeline for **navigation → prompting/propagation → final correction → quantitative export**:

1. **Cohort navigation:** users provide one root path containing patient studies and annotate cases sequentially with explicit actions to proceed or skip, reducing manual file handling during routine labeling. Generated masks and NIfTI label maps are excluded from the patient queue.
2. **Box-first prompting and propagation:** box prompts are the primary interaction for initializing objects. For single-object annotation, the user can place box prompts on the first and last slices where the object appears and run propagation to generate masks for intermediate slices using Medical-SAM2.
3. **Multi-object support with explicit control:** multiple objects can be annotated within the same volume. For multi-object scenarios, prompts can be provided on relevant slices for each object to maintain user control in complex cases.
4. **Point prompts for refinement:** point prompts can be added to refine predictions on a slice; in the current workflow, a box prompt defines the object on that slice and points provide additional guidance for small additions or corrections.
5. **Prompt-first correction and resumable annotation:** users typically obtain the best possible segmentation from prompts and propagation, and then perform a final manual correction step to “lock in” the label before saving. Previously saved or externally produced label maps can be reloaded, aligned to the source geometry, and edited without discarding unaffected slices or object IDs.
6. **Quantitative export and visualization:** the tool maintains a multi-label mask on the original image grid, displays per-object and total volumes during editing, and offers 3D volume rendering to visually inspect the reconstructed shape. Combined and object-wise masks, together with a voxel-count volume report, preserve the source geometry via SimpleITK [@lowekamp2013simpleitk].

# Software design

The GUI is implemented in Python using Napari for multi-dimensional visualization [@sofroniew2022napari] and PyTorch for model execution [@paszke2019pytorch]. Medical-SAM2 [@zhu2024medical] provides SAM2-style memory-based propagation across slice sequences [@ravi2024sam2]. Source volumes are discovered as DICOM series or NIfTI files, while generated mask directories and label-map NIfTI files are omitted from the patient queue. DICOM loading first uses SimpleITK [@lowekamp2013simpleitk]; when malformed or zero-valued slice-spacing metadata prevents loading, pydicom [@mason2011pydicom] recovers slice order and spacing from physical positions, valid spacing tags, slice thickness, or an explicit folder-name fallback.

The editable Napari mask is a display representation, whereas the canonical multi-label mask remains on the original image grid. Imported NIfTI, NRRD, and MetaImage label maps can be replaced or merged, and geometry mismatches are resampled with nearest-neighbor interpolation only after user confirmation. SimpleITK handles general medical-image I/O and geometry, with NiBabel [@brett2024nibabel] providing a Unicode-safe NIfTI path on Windows. Volumes are computed from source-grid voxel counts multiplied by source voxel volume, and the same canonical data are used for the live overlay, combined and object-wise NIfTI outputs, and the text report. PyVista supports optional 3D inspection [@sullivan2019pyvista]. Optional MRI preprocessing includes N4 bias-field correction [@tustison2010n4]. Automated tests cover DICOM spacing recovery, mask import and resampling, source-grid volumetry, Unicode paths, saving, and manual-edit synchronization. The source code, documentation, and tests are maintained in the public repository [@medicalsam2gui2026].

The software is intended for research annotation workflows and does not provide clinical decision support.

# Research impact statement

The project brings together engineering and neurosurgical collaborators at Sungkyunkwan University and Seoul National University Hospital. Its immediate research role is the creation and revision of source-aligned 3D labels for longitudinal medical imaging studies, where segmentation continuity and volume changes must be reviewed across examinations. The combined multi-label mask, object-wise masks, and volume report can be used for dataset curation, inter-reader quality assurance, model training and validation, and longitudinal burden analyses. The local-first workflow also allows protected imaging data to remain within institutional computing environments while retaining reproducible geometry and object identifiers.

The tool's development was presented at the 20th Korean Brain Tumor Society Winter Meeting [@hong2026kbts], and an early usability evaluation was presented at the 44th Annual Spring Meeting of the Korean Neurosurgical Society [@hong2026kns]. These conference presentations are cited as separate research outputs; no participant-level or quantitative usability results are reported in this software paper.

# AI usage disclosure

OpenAI Codex (GPT-5; the exact model snapshot was not exposed) and GitHub Copilot (the underlying model and version were not exposed) were used to assist with portions of software implementation, debugging, automated test generation, and revision of documentation and paper text. All generated suggestions were reviewed, modified where necessary, and validated by the authors through code review, automated tests, and manual GUI evaluation. The authors retained responsibility for the scientific, architectural, and interface-design decisions. Generative AI was not used to generate or analyze study data or scientific results.

# Conflict of interest

The authors declare no competing interests.

# Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) through the Ministry of Science and ICT (No. RS-2025-00517614). We thank the developers of Napari [@sofroniew2022napari], SimpleITK [@lowekamp2013simpleitk], pydicom [@mason2011pydicom], NiBabel [@brett2024nibabel], PyVista [@sullivan2019pyvista], SAM [@kirillov2023sam], SAM2 [@ravi2024sam2], and Medical-SAM2 [@zhu2024medical] for releasing open-source software and models.

# References
