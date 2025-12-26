# Interactive Medical SAM2 GUI

Napari-based GUI that wraps the open Medical-SAM2 model for clinical-style workflows: load DICOM or NIfTI, prompt with points/boxes, propagate masks across slices, manually refine, render multi-object 3D volumes, measure per-object volumes, and save masks aligned to the source geometry.

The typical workflow: add point/box prompts, run propagation, inspect, and iteratively adjust prompts or switch to brush-based manual edits when needed. Boxes generally provide higher fidelity than points; draw them tightly around the target object. Raw MRI can optionally undergo N4 bias field correction and intensity normalization before prompting.
You can point the GUI at a root folder containing DICOM series or NIfTI files; patients are discovered and processed sequentially, and you can choose preprocessing and prompting method per patient.

## Features
- Point/box prompting with propagate and undo/redo history.
- Manual edit mode for label painting and box editing inside Napari; full brush-based mask creation is supported when model outputs need replacement.
- Multi-object tracking, per-object volume computation, and 3D volume rendering.
- DICOM/NIfTI loading with geometry preservation; mask export matches source spacing/origin/direction.
- Works on Linux and Windows with CUDA-enabled or CPU-only PyTorch builds.
- Optional MRI preprocessing: N4 bias field correction and intensity normalization when raw inputs require harmonization.

## Installation (conda)
1) Create environment with Qt handled by the solver:
```bash
conda create -y -n medsam -c conda-forge python=3.10 pyqt=5.15.* pyopengl pip
conda activate medsam
```

2) Install PyTorch matching your GPU/driver. The commands below are examples; check your NVIDIA driver’s supported CUDA runtime and pick a matching torch build (or CPU-only):
```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3) Install project dependencies:
```bash
pip install -r requirements_medsam2_gui.txt
```

Windows: identical steps in an Anaconda/Miniconda/Miniforge PowerShell prompt after activating the env. Linux users should ensure the installed NVIDIA driver supports the chosen CUDA runtime (`nvidia-smi`).

## Quick start
```bash
conda activate medsam
python medsam_gui_v5_multi.py
```

During setup, pick mode (auto/manual), method, and data path. For auto mode, the pipeline builds prompts from detections; for manual mode, you add prompts and propagate.

Planned extension: inserting a detection/segmentation model upfront to auto-generate prompts, then refining and brushing within the same GUI workflow.

## Usage

Navigation dialogs guide you through dataset selection and per-patient options.

- Main dialog: choose `manual` (user-supplied prompts) or `auto` (future version will accept auto-generated prompts from upstream detection/segmentation models). `Method` is only relevant for `auto`. Enable “Perform preprocessing” for raw MRI (N4 + intensity normalization). Set `Data path` to a folder containing NIfTI files or DICOM series.
- Patient queue: patients in the data path are processed in order. For each patient you can toggle preprocessing and mode. `Double viewer` lets you view paired series (e.g., T1/T2) together.
- Workspace: left panel lists layers (image, box/mask layers, user points/boxes); right panel holds prompt and edit controls.

Prompting and propagation
- `Add Box`: click top-left then bottom-right to place a box prompt (works on box/mask layers). One box prompt per object per slice; multiple point prompts are allowed (positive/negative). Boxes usually outperform points—fit them tightly around the object.
- `Add + / - Point`: place positive/negative clicks on the visible slice.
- `Propagate`: define boxes (and optional points) on the first/last slice where the object appears, then propagate to run Medical-SAM2 and get masks. Add prompts slice-by-slice as needed and re-propagate.

Under the hood
- Prompts are stored as `pos_points`, `neg_points`, and `box_prompts`; `Undo/Redo` replays this history.
- Boxes sync from the “User Boxes correction layer” and are auto-rectified to valid rectangles; edits keep shapes stable.
- Points sync from the “User Points correction layer” and preserve original object IDs when edited.
- Propagation builds a sub-volume between the min/max prompted slices, pushes boxes then points into Medical-SAM2 (`train_add_new_bbox` → `train_add_new_points`), and collects logits via `propagate_in_video` to form per-slice label masks.

Editing and QA
- Manual edit mode: brush/erase labels, adjust brush size, change object ID colors, and tweak mask opacity.
- `Edit Points`: toggle editability of user point layer to move/delete points.
- `Edit Boxes`: toggle editability of user box layer; rectangles keep shape during edits.
- `Manual Edit`: enable/disable napari painting/rectangle drawing (clears box-drag callbacks when enabled).
- `Clear All`: remove prompts/masks and reset history.
- `Hide/Show Object IDs`: toggle object ID overlays.
- `Undo`/`Redo`: prompt history management.
- `3D Volume Render`: PyVista view to inspect object location and per-object volumes (voxel volume × count).

Saving
- `Save Mask`: choose an output folder; object-wise masks and the combined mask are saved as `.nii.gz` with preserved geometry.

Layers (left panel)
- `image, point layer`: base RGB image per slice.
- `mask, box layer`: labels/masks and box drawing surface.
- `User Points correction layer`: editable user points (positive/negative) for prompts.
- `User Boxes correction layer`: editable rectangles for box prompts.
- `Object IDs`: optional overlay for object ID display.


## Data and intensity handling
- Accepts DICOM folders or NIfTI files. Geometry (spacing/origin/direction) is preserved on save.
- Per-slice preprocessing for display/model input: percentile clip (0.5/99.5), normalize per slice, scale to 0–255 `uint8`. The tensors are cast to `float32` but retain the 0–255 range before entering the model.

## Outputs
- Masks are saved with the original geometry. Per-object volume summaries and 3D renderings are available from the GUI.

## Tests / Sanity checks
- Import check (headless): `python - <<'PY'
import medsam_gui_dataloader_v2, gui.navigation, gui.segmentation
print('Imports OK')
PY`
- GPU/driver check: `nvidia-smi` (Linux) or NVIDIA-SMI in PowerShell (Windows), then run the PyTorch CUDA snippet in the README install section.
- GUI smoke test: `python medsam_gui_v5_multi.py`, browse a sample DICOM/NIfTI folder, and ensure images and prompt layers render without errors.

## How to cite
Until a JOSS DOI is issued, please cite the repository: https://github.com/SKKU-IBE/SNU_MedSAM2_GUI. The Medical-SAM2 model and weights are from https://github.com/ImprintLab/Medical-SAM2; cite their work per their license.

## License
Apache License 2.0 (see `LICENSE`).

Model weights: downloaded from https://github.com/ImprintLab/Medical-SAM2 and subject to that project’s license.*** End Patch