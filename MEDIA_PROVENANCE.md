# Media Provenance

The repository redistributes one public FLAIR volume and its corresponding
tumor label map from the dataset below as a compact, reproducible demonstration
case. The source NIfTI files and the derived README and paper media are provided
under the Creative Commons Attribution 4.0 International license.

## TCIA UCSD-PTGBM demonstration

| Field | Value |
|---|---|
| Collection | UCSD-PTGBM |
| Collection release | Version 3, updated 2026-03-13 |
| Access package | BraTS-GLI 2024 Test Data (`UCSD-PTGBM-BraTS-2024-test-set`) |
| TCIA package release | 2026-01-23 |
| Case | `UCSD-PTGBM-0001_01` |
| Image modality | FLAIR MRI |
| Annotation | BraTS tumor segmentation label map |
| Dataset DOI | <https://doi.org/10.7937/fwv2-dt74> |
| Dataset page | <https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/> |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

### Derived files

- `images/Figure1.png`
- `images/Figure2.png`
- `images/Image-1.png`
- `images/Image-2.png`
- `images/Image-3.png`
- `images/Image-4.png`
- `images/Image-5.png`
- `images/Image-6.png`

### Bundled source files

- `test_data/UCSD-PTGBM-0001_01_FLAIR.nii.gz`
  - SHA-256: `629822ee3b6ae5ea61e3b20ae56787acb2bcfcdda43185c75c02b7fc1c5b987d`
- `test_data/UCSD-PTGBM-0001_01_BraTS_tumor_seg.nii.gz`
  - SHA-256: `5cf25273aecbfcb5cc7d531475b9cfb76c9294d7ba6087df681a8419c7bab2e9`

These two files are redistributed without intentional modification after
download. They retain the dataset license rather than the repository's
GPL-3.0-only software license. See `test_data/README.md` and
`test_data/LICENSE.md` for the data-specific notice.

### Processing

The FLAIR volume and corresponding `BraTS_tumor_seg` label map were downloaded
through the official TCIA Aspera access package. Interactive
Medical-SAM2 GUI v1.1.0 loaded the FLAIR volume through its standard NIfTI
dataset path and imported the label map through the Manual-mode `Load Masks`
workflow. The GUI generated its 1024 x 1024 display representation while
retaining the label map on the original source grid for volume calculation.
The screenshots document the initial setup, patient navigation, resumed
multi-label annotation, box and point prompting, color-matched source-grid
volumetry, and optional 3D rendering. `Figure1.png` presents the main viewer,
while `Figure2.png` combines the setup, navigation, and 3D-rendering captures.
The application windows were resized for legibility; no anatomical content or
labels were added, removed, or generated in the bundled source files.

Only the public case identifier is shown with a generic local demonstration
path (`C:/Users/user/test_data`) in the setup capture. Acquisition dates,
institutional identifiers, and private cohort paths are not present in the
derived media.

### Attribution

Data used in the demonstration: Gagnon, L., Gupta, D., Nguyen, U., Correia de
Verdier, M., Saluja, R., Mastorakos, G., White, N., Goodwill, V., McDonald,
C. R., Beaumont, T., Conlin, C., Seibert, T. M., Hattangadi-Gluth, J., Kesari,
S., Schulte, J. D., Piccioni, D., Schmainda, K. M., Farid, N., Dale, A. M., and
Rudie, J. D. (2026). *The University of California San Diego Post-Treatment
Glioblastoma (UCSD-PTGBM) Annotated Multimodal MRI Dataset*, Version 3. The
Cancer Imaging Archive. <https://doi.org/10.7937/fwv2-dt74>. Licensed under CC
BY 4.0.
