# UCSD-PTGBM Demonstration Data

This directory contains one public FLAIR volume and its corresponding BraTS
tumor-segmentation label map for reproducing the documented Medical-SAM2 GUI
workflow. The files are redistributed without intentional modification after
download from the official TCIA access package.

## Files

| File | Description | Shape | Data type | SHA-256 |
|---|---|---:|---|---|
| `UCSD-PTGBM-0001_01_FLAIR.nii.gz` | FLAIR MRI | 256 x 256 x 256 | float32 | `629822ee3b6ae5ea61e3b20ae56787acb2bcfcdda43185c75c02b7fc1c5b987d` |
| `UCSD-PTGBM-0001_01_BraTS_tumor_seg.nii.gz` | BraTS tumor label map | 256 x 256 x 256 | uint16 | `5cf25273aecbfcb5cc7d531475b9cfb76c9294d7ba6087df681a8419c7bab2e9` |

Both files use 1 mm isotropic voxel spacing. To verify the downloaded Git
checkout on PowerShell, run:

```powershell
Get-FileHash test_data\*.nii.gz -Algorithm SHA256
```

## Source And Attribution

- Collection: UCSD-PTGBM, Version 3
- Access package: BraTS-GLI 2024 Test Data
  (`UCSD-PTGBM-BraTS-2024-test-set`)
- Public case identifier: `UCSD-PTGBM-0001_01`
- Repository files: FLAIR MRI and BraTS tumor segmentation
- Dataset DOI: <https://doi.org/10.7937/fwv2-dt74>
- Dataset page: <https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/>
- TCIA data-usage policy: <https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/>
- License: [Creative Commons Attribution 4.0 International](LICENSE.md)

Required attribution:

Gagnon, L., Gupta, D., Nguyen, U., Correia de Verdier, M., Saluja, R.,
Mastorakos, G., White, N., Goodwill, V., McDonald, C. R., Beaumont, T., Conlin,
C., Seibert, T. M., Hattangadi-Gluth, J., Kesari, S., Schulte, J. D., Piccioni,
D., Schmainda, K. M., Farid, N., Dale, A. M., and Rudie, J. D. (2026). *The
University of California San Diego Post-Treatment Glioblastoma (UCSD-PTGBM)
Annotated Multimodal MRI Dataset*, Version 3. The Cancer Imaging Archive.
<https://doi.org/10.7937/fwv2-dt74>.

## Use In The GUI

1. Select this `test_data/` directory as the cohort root.
2. Open `UCSD-PTGBM-0001_01_FLAIR.nii.gz` from the patient list.
3. In Manual mode, select `Load Masks` and choose
   `UCSD-PTGBM-0001_01_BraTS_tumor_seg.nii.gz`.

Study discovery should identify the integer segmentation as a label map and
exclude it from the patient list. The source data are not required for normal
operation or automated tests and are not installed as runtime package data.

## Data-Use Conditions

The NIfTI files are data, not GPL-licensed project code. They remain available
under CC BY 4.0. Users must preserve attribution, link to the license, indicate
changes, and comply with the TCIA data-usage policy, including the prohibition
on attempting to identify or contact participants. No endorsement by TCIA or
the dataset authors is implied.
