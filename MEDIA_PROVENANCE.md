# Media Provenance

The repository does not redistribute source medical-image volumes. The public
README and paper media are derived from the dataset below and are distributed
under the same Creative Commons Attribution 4.0 International license.

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

- `images/manual-mask-resume.png`
- `images/manual-mask-resume.gif`

### Processing

The FLAIR volume and corresponding `BraTS_tumor_seg` label map were downloaded
temporarily through the official TCIA Aspera access package. Interactive
Medical-SAM2 GUI v1.1.0 loaded the FLAIR volume through its standard NIfTI
dataset path and imported the label map through the Manual-mode `Load Masks`
workflow. The GUI generated its 1024 x 1024 display representation while
retaining the label map on the original source grid for volume calculation.
The screenshot was captured at a slice with tumor labels, and the GIF advances
through neighboring slices. The application window was resized for legibility;
no anatomical content or labels were added, removed, or generated. The source
NIfTI files are not committed to this repository.

Only the public case identifier is shown. Personal paths, acquisition dates,
institutional identifiers, and local usernames are not present in the derived
media.

### Attribution

Data used in the demonstration: Gagnon, L., Gupta, D., Nguyen, U., Correia de
Verdier, M., Saluja, R., Mastorakos, G., White, N., Goodwill, V., McDonald,
C. R., Beaumont, T., Conlin, C., Seibert, T. M., Hattangadi-Gluth, J., Kesari,
S., Schulte, J. D., Piccioni, D., Schmainda, K. M., Farid, N., Dale, A. M., and
Rudie, J. D. (2026). *The University of California San Diego annotated
post-treatment high-grade glioma multimodal MRI dataset (UCSD-PTGBM)*, Version
3. The Cancer Imaging Archive. <https://doi.org/10.7937/fwv2-dt74>. Licensed
under CC BY 4.0.
