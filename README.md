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

## Installation (micromamba/conda)
1) Create environment with Qt handled by the solver:
```bash
micromamba create -y -n medsam -c conda-forge python=3.10 pyqt=5.15.* pyopengl pip
micromamba activate medsam
```
If you prefer conda, the same command works with `conda` instead of `micromamba`.

2) Install PyTorch matching your GPU/driver (choose one):
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
micromamba activate medsam  # or conda activate medsam
python medsam_gui_v5_multi.py
```

During setup, pick mode (auto/manual), method, and data path. For auto mode, the pipeline builds prompts from detections; for manual mode, you add prompts and propagate.

Planned extension: inserting a detection/segmentation model upfront to auto-generate prompts, then refining and brushing within the same GUI workflow.

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