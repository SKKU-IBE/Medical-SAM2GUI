from types import SimpleNamespace

import pytest

from dataloader import _infer_dicom_slice_order_and_spacing


def _ds(
    instance,
    position=None,
    orientation=None,
    spacing_between=None,
    thickness=None,
):
    return SimpleNamespace(
        InstanceNumber=instance,
        ImagePositionPatient=position,
        ImageOrientationPatient=orientation,
        SpacingBetweenSlices=spacing_between,
        SliceThickness=thickness,
    )


def test_position_spacing_wins_over_zero_spacing_between_slices():
    orientation = [1, 0, 0, 0, 0, -1]
    items = [
        ("slice_003.dcm", _ds(3, [-1, 0, 5], orientation, spacing_between=0, thickness=9)),
        ("slice_001.dcm", _ds(1, [-1, 6, 5], orientation, spacing_between=0, thickness=9)),
        ("slice_002.dcm", _ds(2, [-1, 3, 5], orientation, spacing_between=0, thickness=9)),
    ]

    ordered, spacing, source, origin, _ = _infer_dicom_slice_order_and_spacing(
        items, "case_3mm"
    )

    assert [path for path, _ in ordered] == ["slice_003.dcm", "slice_002.dcm", "slice_001.dcm"]
    assert spacing == pytest.approx(3.0)
    assert source == "dicom_position"
    assert origin == (-1.0, 0.0, 5.0)


def test_positive_spacing_between_slices_is_used_when_positions_are_missing():
    items = [
        ("slice_001.dcm", _ds(1, spacing_between=0, thickness=7)),
        ("slice_002.dcm", _ds(2, spacing_between=4.0, thickness=7)),
        ("slice_003.dcm", _ds(3, spacing_between=4.5, thickness=7)),
    ]

    _, spacing, source, _, _ = _infer_dicom_slice_order_and_spacing(items, "case_3mm")

    assert spacing == pytest.approx(4.25)
    assert source == "spacing_between_slices"


def test_slice_thickness_is_used_before_folder_name():
    items = [
        ("slice_001.dcm", _ds(1, spacing_between=0, thickness=2.5)),
        ("slice_002.dcm", _ds(2, spacing_between=None, thickness=3.5)),
    ]

    _, spacing, source, _, _ = _infer_dicom_slice_order_and_spacing(items, "case_9mm")

    assert spacing == pytest.approx(3.0)
    assert source == "slice_thickness"


def test_folder_name_spacing_is_last_resort():
    items = [
        ("slice_001.dcm", _ds(1)),
        ("slice_002.dcm", _ds(2)),
    ]

    _, spacing, source, _, _ = _infer_dicom_slice_order_and_spacing(items, "study_3.3mm")

    assert spacing == pytest.approx(3.3)
    assert source == "folder_name"


def test_duplicate_slice_positions_raise_clear_error():
    orientation = [1, 0, 0, 0, 0, -1]
    items = [
        ("slice_001.dcm", _ds(1, [-1, 0, 5], orientation, spacing_between=3)),
        ("slice_002.dcm", _ds(2, [-1, 0, 5], orientation, spacing_between=3)),
    ]

    with pytest.raises(RuntimeError, match="Duplicate DICOM slice positions"):
        _infer_dicom_slice_order_and_spacing(items, "case_3mm")
