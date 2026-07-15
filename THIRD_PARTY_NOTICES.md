# Third-Party Notices

Interactive Medical-SAM2 GUI is distributed under the GNU General Public
License v3.0 only. The repository also contains files derived from upstream
projects whose original notices and licenses are preserved below. Inclusion in
this GPL-licensed distribution does not replace those upstream notices.

## Medical-SAM2 and Meta SAM 2

The `medical_sam2_train/` implementation is copied from or adapted from:

- Medical-SAM2: <https://github.com/ImprintLab/Medical-SAM2>
- Segment Anything Model 2 (SAM 2): <https://github.com/facebookresearch/sam2>

Those upstream code portions are provided under the Apache License 2.0. A copy
is stored at `medical_sam2_train/LICENSE`.

## cc_torch connected components implementation

`medical_sam2_train/csrc/connected_components.cu` is derived from the
connected-components implementation included by SAM 2. Its BSD 3-Clause
license is stored at `medical_sam2_train/LICENSE_cctorch`.

## Medical-SAM2 checkpoint

`Medical_SAM2_pretrain.pth` is not redistributed by this repository or its
Python packages. The downloader retrieves it directly from the upstream
Medical-SAM2 Hugging Face repository and verifies the published artifact by
SHA-256. Users must review and comply with the upstream model terms:

<https://huggingface.co/jiayuanz3/MedSAM2_pretrain>

## Example medical imaging data and media

The repository includes one FLAIR NIfTI volume and one tumor-segmentation NIfTI
label map from the public TCIA UCSD-PTGBM BraTS-GLI 2024 Test Data package.
Those files and their derived screenshots are licensed under CC BY 4.0 and are
not covered by the repository's GPL-3.0-only software license. Their filenames,
checksums, citation, processing record, and attribution are documented in
`test_data/README.md` and `MEDIA_PROVENANCE.md`.
