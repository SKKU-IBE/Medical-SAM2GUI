# Changelog

All notable changes to this project are documented here.

## [1.1.0] - 2026-07-14

### Added

- Verified Medical-SAM2 checkpoint downloader and console entry points.
- Mixed DICOM/NIfTI cohort discovery with conservative NIfTI label-map exclusion.
- Manual-mode import of NIfTI, NRRD, and MetaImage label maps for resumed editing.
- Replace/merge import modes, geometry validation, and nearest-neighbor resampling.
- Source-grid canonical multi-label masks and live per-object/total volume overlay.
- Unicode-safe NIfTI read/write fallback on Windows.
- DICOM slice-spacing recovery from physical positions, valid tags, thickness, or folder names.
- Ubuntu/Python 3.10 and Windows/Python 3.12 CI, packaging checks, and JOSS draft workflow.

### Changed

- Unified package, citation, and documentation version metadata at 1.1.0.
- Saved full masks now preserve multi-label source-grid data and geometry.
- Propagation preserves other Object IDs and slices outside the prompted range.
- Mask save dialogs start beside the current source study.
- README and paper figures now use an attributed TCIA UCSD-PTGBM case.

### Fixed

- Corrected volume calculation that previously used the display grid rather than source voxels.
- Synchronized the final paint, erase, or fill event before save.
- Prevented long Napari strokes from losing their final mouse-move segments.
- Prevented stale generated object masks from remaining after an Object ID is deleted.
- Recovered DICOM series rejected because `SpacingBetweenSlices` was zero.
- Preserved an existing checkpoint when a forced replacement download fails.

### Security and rights

- Replaced the damaged root license with canonical `GPL-3.0-only` text.
- Added Apache-2.0 and BSD-3-Clause upstream license copies and third-party notices.
- Removed checkpoint binaries and non-redistributable medical media from Git history.
- Stopped redistributing the checkpoint; releases now use the verified downloader.

## [1.0.1] - 2025-05-04

- Previous maintenance release.

## [1.0.0] - 2025-04-26

- Initial public release.

[1.1.0]: https://github.com/SKKU-IBE/Medical-SAM2GUI/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/SKKU-IBE/Medical-SAM2GUI/releases/tag/v1.0.1
[1.0.0]: https://github.com/SKKU-IBE/Medical-SAM2GUI/releases/tag/v1.0.0
