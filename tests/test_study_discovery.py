from pathlib import Path

import nibabel as nib
import numpy as np

from dataloader import discover_studies, nifti_is_label_map


def _write_nifti(path, data, intent=None):
    image = nib.Nifti1Image(np.asarray(data), np.eye(4))
    if intent is not None:
        image.header.set_intent(intent)
    nib.save(image, str(path))


def test_binary_label_nifti_is_not_added_as_patient(tmp_path):
    path = tmp_path / "neutral_volume_name.nii.gz"
    data = np.zeros((8, 8, 3), dtype=np.uint8)
    data[2:6, 2:6, 1] = 1
    _write_nifti(path, data)

    assert nifti_is_label_map(path) is True
    assert discover_studies(str(tmp_path)) == []


def test_sparse_multilabel_nifti_is_not_added_as_patient(tmp_path):
    path = tmp_path / "external_result.nii.gz"
    data = np.zeros((8, 8, 3), dtype=np.int16)
    data[1:3, 1:3, 0] = 7
    data[4:6, 4:6, 2] = 1000
    _write_nifti(path, data)

    assert discover_studies(str(tmp_path)) == []


def test_nifti_label_intent_is_excluded_even_with_many_values(tmp_path):
    path = tmp_path / "atlas.nii.gz"
    data = np.arange(5 * 5 * 4, dtype=np.int16).reshape(5, 5, 4)
    _write_nifti(path, data, intent="label")

    assert discover_studies(str(tmp_path)) == []


def test_integer_intensity_nifti_with_many_values_remains_a_patient(tmp_path):
    path = tmp_path / "ct_image.nii.gz"
    data = np.arange(5 * 5 * 4, dtype=np.int16).reshape(5, 5, 4)
    _write_nifti(path, data)

    assert discover_studies(str(tmp_path)) == [(str(path), "ct_image")]


def test_float_intensity_nifti_remains_a_patient(tmp_path):
    path = tmp_path / "mr_image.nii.gz"
    data = np.linspace(0.0, 1.0, 8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
    _write_nifti(path, data)

    assert discover_studies(str(tmp_path)) == [(str(path), "mr_image")]


def test_masks_and_preprocessed_directories_are_pruned(tmp_path):
    dicom_dir = tmp_path / "study_3mm"
    dicom_dir.mkdir()
    (dicom_dir / "slice001.dcm").write_bytes(b"not needed for discovery")
    masks_dir = dicom_dir / "study_3mm_masks"
    masks_dir.mkdir()
    (masks_dir / "unreadable_mask.nii.gz").write_bytes(b"invalid")
    preprocessed_dir = tmp_path / "Preprocessed"
    preprocessed_dir.mkdir()
    (preprocessed_dir / "cached.nii.gz").write_bytes(b"invalid")

    assert discover_studies(str(tmp_path)) == [(str(dicom_dir), "study_3mm")]


def test_unreadable_nifti_is_kept_for_normal_load_error_handling(tmp_path, capsys):
    path = tmp_path / "possibly_image.nii.gz"
    path.write_bytes(b"invalid")

    studies = discover_studies(str(tmp_path))

    assert studies == [(str(path), "possibly_image")]
    assert "keeping it in the patient list" in capsys.readouterr().out


def test_direct_label_file_path_returns_no_studies(tmp_path):
    path = tmp_path / "한글 label.nii.gz"
    _write_nifti(path, np.zeros((3, 3, 3), dtype=np.uint8))

    assert discover_studies(str(path)) == []
