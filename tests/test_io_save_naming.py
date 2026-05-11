from types import SimpleNamespace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


class DummyTensor:
    pass


sys.modules.setdefault("torch", SimpleNamespace(Tensor=DummyTensor))
_REPO_ROOT = Path(__file__).resolve().parents[1]
_IO_SPEC = importlib.util.spec_from_file_location("gui_io", _REPO_ROOT / "gui" / "io.py")
io = importlib.util.module_from_spec(_IO_SPEC)
_IO_SPEC.loader.exec_module(io)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("case01.nii.gz", "case01"),
        ("case01.nii", "case01"),
        (r"C:\data\case01", "case01"),
        ("group/aorta:case?.nii.gz", "aorta_case_"),
        ("Patient 2: GKRS0001_T1_image", "GKRS0001_T1_image"),
        ("", "Unknown"),
        (None, "Unknown"),
    ],
)
def test_safe_source_stem(raw, expected):
    assert io._safe_source_stem(raw) == expected


def test_get_save_patient_stem_prefers_meta_patient():
    gui = SimpleNamespace(meta={"patient": "meta_case.nii.gz"}, patient_id="display_case")

    assert io._get_save_patient_stem(gui) == "meta_case"


def test_get_save_patient_stem_falls_back_to_patient_id():
    gui = SimpleNamespace(meta={"patient": None}, patient_id="Patient 1: case01")

    assert io._get_save_patient_stem(gui) == "case01"


def test_save_masks_manual_uses_source_stem_for_folder_and_files(tmp_path, monkeypatch):
    gui = SimpleNamespace(
        meta={
            "patient": "group/aorta:case?.nii.gz",
            "shape": (2, 3, 3),
            "spacing": (1.0, 1.0, 1.0),
            "origin": (0.0, 0.0, 0.0),
            "direction": tuple(np.eye(3).flatten()),
        },
        patient_id="Patient 1: ignored",
        mask_layer=SimpleNamespace(
            data=np.array(
                [
                    [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                ],
                dtype=np.uint8,
            )
        ),
        metrics=None,
    )
    monkeypatch.setattr(io.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(io.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(io.QMessageBox, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(io.QMessageBox, "critical", lambda *args, **kwargs: None)

    io.save_masks_manual(gui)

    patient_dir = tmp_path / "aorta_case__masks"
    assert patient_dir.is_dir()
    assert (patient_dir / "aorta_case__full_mask.nii.gz").is_file()
    assert (patient_dir / "aorta_case__mask_objectID_1.nii.gz").is_file()
    assert (patient_dir / "aorta_case__mask_objectID_2.nii.gz").is_file()


def test_save_masks_auto_uses_source_stem_for_folder_and_files(tmp_path, monkeypatch):
    gui = SimpleNamespace(
        meta={
            "patient": r"C:\data\case01.nii.gz",
            "spacing": (1.0, 1.0, 1.0),
            "origin": (0.0, 0.0, 0.0),
            "direction": tuple(np.eye(3).flatten()),
        },
        patient_id="Patient 2: ignored",
        mask_layer=SimpleNamespace(
            data=np.array(
                [
                    [[0, 1], [0, 0]],
                    [[0, 0], [0, 2]],
                ],
                dtype=np.uint8,
            )
        ),
        metrics=None,
    )
    monkeypatch.setattr(io.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(io.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(io.QMessageBox, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(io.QMessageBox, "critical", lambda *args, **kwargs: None)

    io.save_masks_auto(gui)

    patient_dir = tmp_path / "case01_masks"
    assert patient_dir.is_dir()
    assert (patient_dir / "case01_full_mask.nii.gz").is_file()
    assert (patient_dir / "case01_mask_label_1.nii.gz").is_file()
    assert (patient_dir / "case01_mask_label_2.nii.gz").is_file()
