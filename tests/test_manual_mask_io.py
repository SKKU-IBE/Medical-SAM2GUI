from types import SimpleNamespace
import importlib.util
from pathlib import Path
import warnings

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from gui.manual_gui import ManualPromptNapariGUI


_REPO_ROOT = Path(__file__).resolve().parents[1]
_IO_SPEC = importlib.util.spec_from_file_location(
    "manual_mask_io", _REPO_ROOT / "gui" / "io.py"
)
io = importlib.util.module_from_spec(_IO_SPEC)
_IO_SPEC.loader.exec_module(io)


def _meta(shape=(2, 4, 4), spacing=(0.5, 0.5, 2.0), patient="case01"):
    return {
        "patient": patient,
        "shape": shape,
        "spacing": spacing,
        "origin": (10.0, 20.0, 30.0),
        "direction": tuple(np.eye(3).reshape(-1)),
    }


def _write_label_image(path, array, meta):
    image = sitk.GetImageFromArray(np.asarray(array))
    reference = io.build_mask_reference_image(meta, array.shape)
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path))


def _write_nifti_with_nibabel(path, array, meta, affine=None):
    if affine is None:
        direction = np.asarray(meta["direction"], dtype=np.float64).reshape(3, 3)
        spacing = np.asarray(meta["spacing"], dtype=np.float64)
        lps_affine = np.eye(4, dtype=np.float64)
        lps_affine[:3, :3] = direction * spacing[np.newaxis, :]
        lps_affine[:3, 3] = np.asarray(meta["origin"], dtype=np.float64)
        affine = np.diag((-1.0, -1.0, 1.0, 1.0)) @ lps_affine
    data_xyz = np.transpose(np.asarray(array), (2, 1, 0))
    nib.save(nib.Nifti1Image(data_xyz, affine), str(path))


class _SpinBox:
    def __init__(self):
        self.minimum = 1
        self.maximum = 100
        self.value = 1

    def setRange(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    def setValue(self, value):
        self.value = value


class _ImportGui:
    def __init__(self, meta):
        self.meta = meta
        self.mask_layer = SimpleNamespace(
            data=np.zeros(meta["shape"], dtype=np.uint8)
        )
        self.source_mask_data = np.zeros(meta["shape"], dtype=np.int32)
        self.current_obj_id = 1
        self.oid_spin = _SpinBox()
        self.metrics = None
        self.history_calls = []

    def flush_source_mask_updates(self):
        return self.source_mask_data

    def set_source_mask_data(self, source_mask, record_history=True):
        self.source_mask_data = np.asarray(source_mask, dtype=np.int32).copy()
        self.mask_layer.data = io.source_mask_to_display(
            self.source_mask_data, self.mask_layer.data.shape
        )
        self.history_calls.append(record_history)

    def _update_object_id_text(self):
        pass


class _FakeVolumeTimer:
    def __init__(self):
        self.active = False
        self.start_count = 0

    def start(self, interval=None):
        self.active = True
        self.start_count += 1

    def stop(self):
        self.active = False

    def isActive(self):
        return self.active


class _ManualSyncState:
    _positive_label_counts = staticmethod(
        ManualPromptNapariGUI._positive_label_counts
    )
    _schedule_volume_update = ManualPromptNapariGUI._schedule_volume_update
    _sync_source_mask_from_display = (
        ManualPromptNapariGUI._sync_source_mask_from_display
    )
    _rebuild_volume_count_cache = ManualPromptNapariGUI._rebuild_volume_count_cache
    _update_volume_count_cache_slices = (
        ManualPromptNapariGUI._update_volume_count_cache_slices
    )
    _flush_volume_update = ManualPromptNapariGUI._flush_volume_update
    flush_source_mask_updates = ManualPromptNapariGUI.flush_source_mask_updates
    _on_manual_labels_update = ManualPromptNapariGUI._on_manual_labels_update
    _manual_edit_stroke_callback = (
        ManualPromptNapariGUI._manual_edit_stroke_callback
    )
    _ensure_manual_stroke_callback = (
        ManualPromptNapariGUI._ensure_manual_stroke_callback
    )

    def __init__(self, source_mask, display_mask, meta):
        self.meta = meta
        self.patient_id = "ignored"
        self.mask_layer = SimpleNamespace(
            data=np.asarray(display_mask).copy(), mouse_drag_callbacks=[]
        )
        self.source_mask_data = np.asarray(source_mask, dtype=np.int32).copy()
        self.frame_idx = 1
        self.manual_edit_enabled = True
        self._updating_layers = False
        self._manual_stroke_active = False
        self._manual_stroke_frame = None
        self._stroke_start_state = None
        self._last_mask_state = self.mask_layer.data.copy()
        self.mask_history = []
        self.mask_redo_stack = []
        self._pending_source_slices = set()
        self._pending_full_source_sync = False
        self._volume_timer = _FakeVolumeTimer()
        self._slice_label_counts = []
        self._volume_label_counts = {}
        self.overlay_update_count = 0
        self.metrics = None
        self._rebuild_volume_count_cache()

    def _push_mask_history(self, snapshot):
        self.mask_history.append(np.asarray(snapshot).copy())

    def _update_volume_overlay_now(self):
        self.overlay_update_count += 1


def test_display_round_trip_uses_source_grid_for_volume():
    source = np.zeros((2, 512, 512), dtype=np.uint8)
    source[:, 100:140, 200:240] = 1
    display = io.source_mask_to_display(source, (2, 1024, 1024))
    restored = io.display_mask_to_source(display, source.shape)

    assert np.count_nonzero(display) == np.count_nonzero(source) * 4
    assert np.array_equal(restored, source)
    entries = io.compute_label_volume_entries(restored, (0.3516, 0.3516, 3.0))
    assert entries[0][1] == 3200
    assert entries[0][2] == pytest.approx(1186.776576)


def test_known_voxel_count_volume_regression():
    source = np.zeros((1, 72, 72), dtype=np.uint8)
    source.reshape(-1)[:5137] = 1

    entries = io.compute_label_volume_entries(
        source, (0.3515999913215637, 0.3515999913215637, 3.0000038146972656)
    )

    assert entries[0][:2] == (1, 5137)
    assert entries[0][2] == pytest.approx(1905.1496006315658)


@pytest.mark.parametrize("suffix", [".nii.gz", ".nrrd", ".mha", ".mhd"])
def test_supported_mask_formats_are_read(tmp_path, suffix):
    meta = _meta()
    mask = np.zeros(meta["shape"], dtype=np.uint8)
    mask[:, 1:3, 1:3] = 1
    path = tmp_path / f"mask{suffix}"
    _write_label_image(path, mask, meta)

    loaded = io._read_import_mask(path)

    assert loaded.GetSize() == (4, 4, 2)


def test_geometry_resampling_uses_physical_coordinates():
    meta = _meta(shape=(2, 4, 4), spacing=(1.0, 1.0, 1.0))
    reference = io.build_mask_reference_image(meta, meta["shape"])
    coarse = np.ones((2, 2, 2), dtype=np.uint8)
    image = sitk.GetImageFromArray(coarse)
    image.SetSpacing((2.0, 2.0, 1.0))
    image.SetOrigin(meta["origin"])
    image.SetDirection(meta["direction"])

    prepared, changed = io._prepare_imported_mask(
        image, "external_mask.nrrd", reference, current_obj_id=3
    )

    assert changed is True
    assert prepared.shape == meta["shape"]
    assert set(np.unique(prepared)) <= {0, 3}
    assert np.count_nonzero(prepared) > 0


def test_non_overlapping_geometry_is_rejected():
    meta = _meta()
    reference = io.build_mask_reference_image(meta, meta["shape"])
    image = sitk.GetImageFromArray(np.ones(meta["shape"], dtype=np.uint8))
    image.SetSpacing(meta["spacing"])
    image.SetOrigin((1000.0, 1000.0, 1000.0))
    image.SetDirection(meta["direction"])

    with pytest.raises(ValueError, match="empty after alignment"):
        io._prepare_imported_mask(image, "far_away.nii.gz", reference, 1)


def test_probability_map_is_rejected():
    probability = np.zeros((2, 4, 4), dtype=np.float32)
    probability[0, 0, 0] = 0.4

    with pytest.raises(ValueError, match="probability map"):
        io._validate_label_array(probability, "probability.nii.gz")


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_integer_float_labels_do_not_emit_cast_warnings(dtype):
    labels = np.array([[[0.0, 1.0], [2.0, 0.0]]], dtype=dtype)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validated = io._validate_label_array(labels, "integer_labels.nii.gz")

    assert caught == []
    assert validated.dtype == np.int32
    assert np.array_equal(validated, labels)


def test_nibabel_fallback_reads_unicode_nifti_with_sitk_geometry(
    tmp_path, monkeypatch
):
    meta = _meta(shape=(3, 4, 5), spacing=(0.33, 0.44, 3.0))
    mask = np.zeros(meta["shape"], dtype=np.uint8)
    mask[1:, 1:3, 2:5] = 7
    unicode_directory = tmp_path / "한글 경로"
    unicode_directory.mkdir()
    path = unicode_directory / "외부 마스크.nii.gz"
    _write_nifti_with_nibabel(path, mask, meta)
    monkeypatch.setattr(
        io.sitk, "ReadImage", lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("SimpleITK unicode path failure")
        )
    )

    loaded = io._read_import_mask(path)

    assert np.array_equal(sitk.GetArrayFromImage(loaded), mask)
    assert loaded.GetSize() == (5, 4, 3)
    assert loaded.GetSpacing() == pytest.approx(meta["spacing"])
    assert loaded.GetOrigin() == pytest.approx(meta["origin"])
    assert loaded.GetDirection() == pytest.approx(meta["direction"])


def test_nibabel_fallback_rejects_shear_affine(tmp_path, monkeypatch):
    path = tmp_path / "sheared.nii.gz"
    affine = np.eye(4, dtype=np.float64)
    affine[0, 1] = 0.2
    _write_nifti_with_nibabel(
        path, np.ones((2, 3, 4), dtype=np.uint8), _meta(), affine=affine
    )
    monkeypatch.setattr(
        io.sitk, "ReadImage", lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("SimpleITK read failure")
        )
    )

    with pytest.raises(RuntimeError, match="nibabel fallback:.*shear"):
        io._read_import_mask(path)


def test_non_nifti_read_failure_does_not_use_nibabel(tmp_path, monkeypatch):
    path = tmp_path / "mask.nrrd"
    path.write_bytes(b"invalid")
    fallback_calls = []
    monkeypatch.setattr(
        io.sitk, "ReadImage", lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("SimpleITK read failure")
        )
    )
    monkeypatch.setattr(
        io,
        "_read_nifti_with_nibabel",
        lambda *args: fallback_calls.append(args),
    )

    with pytest.raises(RuntimeError, match="SimpleITK read failure"):
        io._read_import_mask(path)

    assert fallback_calls == []


def test_unicode_nifti_writer_preserves_data_dtype_and_geometry(tmp_path, monkeypatch):
    meta = _meta(shape=(3, 4, 5), spacing=(0.33, 0.44, 3.0))
    meta["direction"] = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0)
    mask = np.zeros(meta["shape"], dtype=np.int32)
    mask[1:, 1:3, 2:5] = 7
    image = sitk.GetImageFromArray(mask)
    image.CopyInformation(io.build_mask_reference_image(meta, mask.shape))
    output_directory = tmp_path / "한글 저장 경로"
    output_directory.mkdir()
    path = output_directory / "외부 mask.nii.gz"
    monkeypatch.setattr(io, "_windows_non_ascii_path", lambda path: True)
    monkeypatch.setattr(
        io.sitk,
        "WriteImage",
        lambda *args, **kwargs: pytest.fail(
            "SimpleITK must be skipped for Windows non-ASCII output paths"
        ),
    )

    backend = io._write_nifti_image(image, path)

    loaded = nib.load(str(path))
    data_zyx = np.transpose(np.asanyarray(loaded.dataobj), (2, 1, 0))
    lps_affine = np.diag((-1.0, -1.0, 1.0, 1.0)) @ loaded.affine
    spacing = np.linalg.norm(lps_affine[:3, :3], axis=0)
    direction = lps_affine[:3, :3] / spacing[np.newaxis, :]
    assert backend == "nibabel"
    assert np.array_equal(data_zyx, mask)
    assert data_zyx.dtype == np.dtype(np.int32)
    assert spacing == pytest.approx(meta["spacing"])
    assert lps_affine[:3, 3] == pytest.approx(meta["origin"])
    assert direction.reshape(-1) == pytest.approx(meta["direction"])
    assert loaded.header.get_xyzt_units()[0] == "mm"
    assert int(loaded.header["qform_code"]) == 1
    assert int(loaded.header["sform_code"]) == 1


def test_ascii_nifti_writer_falls_back_when_simpleitk_fails(tmp_path, monkeypatch):
    image = sitk.GetImageFromArray(np.ones((2, 3, 4), dtype=np.uint8))
    path = tmp_path / "mask.nii.gz"
    monkeypatch.setattr(io, "_windows_non_ascii_path", lambda path: False)
    monkeypatch.setattr(
        io.sitk,
        "WriteImage",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("SimpleITK write failure")
        ),
    )

    assert io._write_nifti_image(image, path) == "nibabel"
    assert path.is_file()


def test_nifti_writer_reports_both_backend_failures(tmp_path, monkeypatch):
    image = sitk.GetImageFromArray(np.ones((2, 3, 4), dtype=np.uint8))
    path = tmp_path / "mask.nii.gz"
    monkeypatch.setattr(io, "_windows_non_ascii_path", lambda path: False)
    monkeypatch.setattr(
        io.sitk,
        "WriteImage",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("SimpleITK write failure")
        ),
    )
    monkeypatch.setattr(
        io,
        "_write_nifti_with_nibabel",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("nibabel write failure")
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="SimpleITK write failure.*nibabel fallback: nibabel write failure",
    ):
        io._write_nifti_image(image, path)


def test_mask_load_directory_prefers_dicom_study_masks(tmp_path):
    study = tmp_path / "20141120_3mm"
    study.mkdir()
    masks = study / "20141120_3mm_masks"
    masks.mkdir()
    gui = SimpleNamespace(meta={"source_path": str(study)})

    assert Path(io.get_mask_load_initial_directory(gui)) == masks


def test_mask_load_directory_falls_back_to_dicom_study(tmp_path):
    study = tmp_path / "20141120_3mm"
    study.mkdir()
    gui = SimpleNamespace(meta={"source_path": str(study)})

    assert Path(io.get_mask_load_initial_directory(gui)) == study


def test_mask_load_directory_prefers_nifti_sibling_masks(tmp_path):
    source = tmp_path / "source_image.nii.gz"
    source.touch()
    masks = tmp_path / "source_image_masks"
    masks.mkdir()
    gui = SimpleNamespace(meta={"source_path": str(source)})

    assert Path(io.get_mask_load_initial_directory(gui)) == masks


def test_mask_load_directory_uses_navigation_path(tmp_path):
    study = tmp_path / "navigation_study"
    study.mkdir()
    navigation = SimpleNamespace(
        current_patient_idx=0,
        patient_list=[(str(study), "navigation_study")],
    )
    gui = SimpleNamespace(
        meta={"source_path": str(tmp_path / "missing")},
        navigation_manager=navigation,
    )

    assert Path(io.get_mask_load_initial_directory(gui)) == study


def test_mask_load_directory_uses_working_directory_when_sources_are_missing(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    gui = SimpleNamespace(meta={"source_path": str(tmp_path / "missing")})

    assert Path(io.get_mask_load_initial_directory(gui)) == tmp_path


def test_load_dialog_starts_in_resolved_mask_directory(tmp_path, monkeypatch):
    meta = _meta()
    study = tmp_path / "case01"
    study.mkdir()
    masks = study / "case01_masks"
    masks.mkdir()
    meta["source_path"] = str(study)
    gui = _ImportGui(meta)
    dialog_calls = []
    monkeypatch.setattr(
        io.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: (dialog_calls.append(args) or ([], "")),
    )

    assert io.load_masks_manual(gui) is False
    assert Path(dialog_calls[0][2]) == masks


def test_mask_save_directory_uses_dicom_study_not_existing_masks(tmp_path):
    study = tmp_path / "20141120_3mm"
    study.mkdir()
    (study / "20141120_3mm_masks").mkdir()
    gui = SimpleNamespace(meta={"source_path": str(study)})

    assert Path(io.get_mask_save_initial_directory(gui)) == study


def test_mask_save_directory_uses_nifti_parent(tmp_path):
    source = tmp_path / "source_image.nii.gz"
    source.touch()
    (tmp_path / "source_image_masks").mkdir()
    gui = SimpleNamespace(meta={"source_path": str(source)})

    assert Path(io.get_mask_save_initial_directory(gui)) == tmp_path


def test_mask_save_directory_uses_navigation_path(tmp_path):
    study = tmp_path / "navigation_study"
    study.mkdir()
    navigation = SimpleNamespace(
        current_patient_idx=0,
        patient_list=[(str(study), "navigation_study")],
    )
    gui = SimpleNamespace(
        meta={"source_path": str(tmp_path / "missing")},
        navigation_manager=navigation,
    )

    assert Path(io.get_mask_save_initial_directory(gui)) == study


@pytest.mark.parametrize("save_function", [io.save_masks_manual, io.save_masks_auto])
def test_save_dialog_starts_in_source_study_directory(
    tmp_path, monkeypatch, save_function
):
    study = tmp_path / "case01"
    study.mkdir()
    (study / "case01_masks").mkdir()
    gui = SimpleNamespace(meta={"source_path": str(study)})
    dialog_calls = []
    monkeypatch.setattr(
        io.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: (dialog_calls.append(args) or ""),
    )

    save_function(gui)

    assert Path(dialog_calls[0][2]) == study


def test_merge_preserves_current_labels_and_reports_conflicts():
    current = np.array([[[1, 0], [2, 0]]], dtype=np.int32)
    imported = np.array([[[1, 3], [4, 0]]], dtype=np.int32)

    merged, conflict_count = io.merge_source_masks_preserving_existing(
        current, imported
    )

    assert np.array_equal(merged, np.array([[[1, 3], [2, 0]]]))
    assert conflict_count == 1


def test_propagation_merge_preserves_other_objects_and_slices():
    current = np.zeros((3, 4, 4), dtype=np.uint8)
    current[0, 0, 0] = 1
    current[1, 0:2, 0:2] = 1
    current[1, 3, 3] = 2
    current[2, 1, 1] = 1
    propagated = np.zeros((4, 4), dtype=np.int32)
    propagated[1:3, 1:3] = 1

    merged = ManualPromptNapariGUI._merge_propagated_masks(
        current, {1: propagated}, {1}
    )

    assert np.array_equal(merged[0], current[0])
    assert np.array_equal(merged[2], current[2])
    assert merged[1, 3, 3] == 2
    assert np.array_equal(merged[1] == 1, propagated == 1)


def test_incremental_volume_cache_recounts_only_changed_slice():
    source = np.zeros((2, 4, 4), dtype=np.int32)
    source[0, 0:2, 0:2] = 1
    source[1, 1:3, 1:3] = 2
    state = SimpleNamespace(
        source_mask_data=source,
        _slice_label_counts=[{}, {}],
        _volume_label_counts={},
        _positive_label_counts=ManualPromptNapariGUI._positive_label_counts,
    )
    ManualPromptNapariGUI._rebuild_volume_count_cache(state)
    assert state._volume_label_counts == {1: 4, 2: 4}

    source[0] = 0
    source[0, 0, 0] = 2
    ManualPromptNapariGUI._update_volume_count_cache_slices(state, [0])

    assert state._volume_label_counts == {2: 5}


def test_manual_drag_callback_stays_active_until_release_and_records_once():
    meta = _meta(shape=(2, 2, 2))
    source = np.zeros(meta["shape"], dtype=np.int32)
    state = _ManualSyncState(
        source, io.source_mask_to_display(source, (2, 4, 4)), meta
    )
    native_callback = lambda *args: None
    state.mask_layer.mouse_drag_callbacks = [
        native_callback,
        state._manual_edit_stroke_callback,
    ]

    state._ensure_manual_stroke_callback()

    assert state.mask_layer.mouse_drag_callbacks[0] == (
        state._manual_edit_stroke_callback
    )
    assert state.mask_layer.mouse_drag_callbacks.count(
        state._manual_edit_stroke_callback
    ) == 1

    event = SimpleNamespace(type="mouse_press")
    callback = state._manual_edit_stroke_callback(state.mask_layer, event)
    next(callback)
    assert state._manual_stroke_active is True
    assert state._manual_stroke_frame == 1

    state.mask_layer.data[1, 0, 0] = 2
    event.type = "mouse_move"
    next(callback)
    assert state._manual_stroke_active is True
    state.mask_layer.data[1, 0, 1] = 2
    next(callback)
    assert state._manual_stroke_active is True

    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(callback)

    assert state._manual_stroke_active is False
    assert state._manual_stroke_frame is None
    assert len(state.mask_history) == 1
    assert not np.any(state.mask_history[0])
    assert state._pending_source_slices == {1}


def test_flush_source_mask_updates_includes_active_stroke_without_event():
    meta = _meta(shape=(2, 2, 2))
    source = np.zeros(meta["shape"], dtype=np.int32)
    state = _ManualSyncState(source, source.copy(), meta)
    state._manual_stroke_active = True
    state._manual_stroke_frame = 1
    state.mask_layer.data[1, 1, 1] = 4

    state.flush_source_mask_updates()

    assert state.source_mask_data[1, 1, 1] == 4
    assert state._pending_source_slices == set()


def test_long_stroke_tail_is_flushed_on_immediate_save_and_preserves_other_slice(
    tmp_path, monkeypatch
):
    meta = _meta(shape=(2, 2, 2), spacing=(0.5, 0.5, 2.0))
    original_source = np.zeros(meta["shape"], dtype=np.int32)
    original_source[0] = np.array([[7, 0], [0, 7]], dtype=np.int32)
    display = io.source_mask_to_display(original_source, (2, 4, 4))
    state = _ManualSyncState(original_source, display, meta)
    event = SimpleNamespace(type="mouse_press")
    callback = state._manual_edit_stroke_callback(state.mask_layer, event)
    next(callback)

    state.mask_layer.data[1, 0:2, 0:2] = 2
    state._on_manual_labels_update()
    event.type = "mouse_move"
    next(callback)
    assert state._pending_source_slices == {1}

    state._volume_timer.stop()
    state._flush_volume_update()
    assert state.source_mask_data[1, 0, 0] == 2
    assert state.source_mask_data[1, 1, 1] == 0
    assert state._pending_source_slices == set()

    state.mask_layer.data[1, 2:4, 2:4] = 2
    state._on_manual_labels_update()
    next(callback)
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(callback)
    assert state._pending_source_slices == {1}
    assert state._volume_timer.start_count == 3

    expected_source = io.sync_source_mask_slices(
        original_source, state.mask_layer.data, [1]
    )
    monkeypatch.setattr(
        io.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(io.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(io.QMessageBox, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(io.QMessageBox, "critical", lambda *args, **kwargs: None)

    io.save_masks_manual(state)

    patient_dir = tmp_path / "case01_masks"
    saved = sitk.GetArrayFromImage(
        sitk.ReadImage(str(patient_dir / "case01_full_mask.nii.gz"))
    )
    assert np.array_equal(saved, expected_source)
    assert np.array_equal(saved[0], original_source[0])
    assert saved[1, 0, 0] == 2
    assert saved[1, 1, 1] == 2
    assert state._volume_label_counts == {2: 2, 7: 2}
    assert (patient_dir / "volumes.txt").read_text().splitlines() == [
        "object_id\tvoxel_count\tvolume_mm3",
        "2\t2\t1.000",
        "7\t2\t1.000",
    ]


def test_multiple_binary_masks_restore_object_ids(tmp_path, monkeypatch):
    meta = _meta()
    first = np.zeros(meta["shape"], dtype=np.uint8)
    second = np.zeros(meta["shape"], dtype=np.uint8)
    first[:, 0:2, 0:2] = 1
    second[:, 2:4, 2:4] = 1
    first_path = tmp_path / "case_mask_objectID_2.nii.gz"
    second_path = tmp_path / "case_mask_label_7.nrrd"
    _write_label_image(first_path, first, meta)
    _write_label_image(second_path, second, meta)
    gui = _ImportGui(meta)

    monkeypatch.setattr(
        io.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(first_path), str(second_path)], ""),
    )
    monkeypatch.setattr(io.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(io.QMessageBox, "critical", lambda *args, **kwargs: None)

    assert io.load_masks_manual(gui) is True
    assert set(np.unique(gui.source_mask_data)) == {0, 2, 7}
    assert gui.oid_spin.value == 2
    assert gui.history_calls == [True]


def test_conflicting_selected_masks_cancel_import(tmp_path, monkeypatch):
    meta = _meta()
    mask = np.zeros(meta["shape"], dtype=np.uint8)
    mask[:, 1:3, 1:3] = 1
    first_path = tmp_path / "case_mask_objectID_2.nii.gz"
    second_path = tmp_path / "case_mask_objectID_7.nii.gz"
    _write_label_image(first_path, mask, meta)
    _write_label_image(second_path, mask, meta)
    gui = _ImportGui(meta)
    errors = []

    monkeypatch.setattr(
        io.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(first_path), str(second_path)], ""),
    )
    monkeypatch.setattr(io.QMessageBox, "critical", lambda *args: errors.append(args[-1]))

    assert io.load_masks_manual(gui) is False
    assert not np.any(gui.source_mask_data)
    assert "overlap" in errors[0]


def test_load_merge_keeps_current_conflicting_voxels(tmp_path, monkeypatch):
    meta = _meta()
    imported = np.zeros(meta["shape"], dtype=np.uint8)
    imported[:, 0:2, 0:2] = 1
    path = tmp_path / "case_mask_objectID_7.nii.gz"
    _write_label_image(path, imported, meta)
    gui = _ImportGui(meta)
    gui.source_mask_data[:, 0, 0] = 2
    gui.mask_layer.data[:, 0, 0] = 2

    monkeypatch.setattr(
        io.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(path)], ""),
    )
    monkeypatch.setattr(io, "_choose_import_mode", lambda gui: "merge")
    monkeypatch.setattr(io.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(io.QMessageBox, "critical", lambda *args, **kwargs: None)

    assert io.load_masks_manual(gui) is True
    assert np.all(gui.source_mask_data[:, 0, 0] == 2)
    assert np.any(gui.source_mask_data == 7)


def test_manual_save_preserves_labels_and_source_volume(tmp_path, monkeypatch):
    meta = _meta(shape=(2, 4, 4), spacing=(0.5, 0.5, 2.0))
    source = np.zeros(meta["shape"], dtype=np.int32)
    source[0, 0:2, 0:2] = 1
    source[1, 1:4, 1:4] = 3
    display = io.source_mask_to_display(source, (2, 8, 8))
    gui = SimpleNamespace(
        meta=meta,
        patient_id="ignored",
        mask_layer=SimpleNamespace(data=display),
        source_mask_data=source.copy(),
        flush_source_mask_updates=lambda: source,
        metrics=None,
    )
    patient_dir = tmp_path / "case01_masks"
    patient_dir.mkdir()
    stale = patient_dir / "case01_mask_objectID_9.nii.gz"
    stale.write_bytes(b"stale")

    monkeypatch.setattr(
        io.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(tmp_path)
    )
    monkeypatch.setattr(io.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(io.QMessageBox, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(io.QMessageBox, "critical", lambda *args, **kwargs: None)

    io.save_masks_manual(gui)

    full_path = patient_dir / "case01_full_mask.nii.gz"
    full_image = sitk.ReadImage(str(full_path))
    full_array = sitk.GetArrayFromImage(full_image)
    assert np.array_equal(full_array, source)
    assert full_image.GetSpacing() == pytest.approx(meta["spacing"])
    assert set(np.unique(full_array)) == {0, 1, 3}
    assert (patient_dir / "case01_mask_objectID_1.nii.gz").is_file()
    assert (patient_dir / "case01_mask_objectID_3.nii.gz").is_file()
    assert not stale.exists()
    assert (patient_dir / "volumes.txt").read_text().splitlines() == [
        "object_id\tvoxel_count\tvolume_mm3",
        "1\t4\t2.000",
        "3\t9\t4.500",
    ]


def test_manual_save_writes_full_and_object_masks_to_unicode_path(
    tmp_path, monkeypatch
):
    meta = _meta(shape=(2, 4, 4), spacing=(0.5, 0.75, 2.5))
    source = np.zeros(meta["shape"], dtype=np.int32)
    source[0, 0:2, 0:2] = 1
    source[1, 1:4, 1:4] = 3
    gui = SimpleNamespace(
        meta=meta,
        patient_id="ignored",
        mask_layer=SimpleNamespace(data=source.copy()),
        source_mask_data=source.copy(),
        flush_source_mask_updates=lambda: source,
        metrics=None,
    )
    save_root = tmp_path / "한글 저장 위치"
    save_root.mkdir()
    monkeypatch.setattr(
        io.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(save_root),
    )
    monkeypatch.setattr(io.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(io.QMessageBox, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(io.QMessageBox, "critical", lambda *args, **kwargs: None)
    monkeypatch.setattr(io, "_windows_non_ascii_path", lambda path: True)
    monkeypatch.setattr(
        io.sitk,
        "WriteImage",
        lambda *args, **kwargs: pytest.fail(
            "SimpleITK must be skipped for Windows non-ASCII output paths"
        ),
    )

    io.save_masks_manual(gui)

    patient_dir = save_root / "case01_masks"
    full_path = patient_dir / "case01_full_mask.nii.gz"
    full_image = nib.load(str(full_path))
    full_data = np.transpose(np.asanyarray(full_image.dataobj), (2, 1, 0))
    lps_affine = np.diag((-1.0, -1.0, 1.0, 1.0)) @ full_image.affine
    assert np.array_equal(full_data, source)
    assert full_data.dtype == np.dtype(np.int32)
    assert np.linalg.norm(lps_affine[:3, :3], axis=0) == pytest.approx(
        meta["spacing"]
    )
    assert lps_affine[:3, 3] == pytest.approx(meta["origin"])
    assert (patient_dir / "case01_mask_objectID_1.nii.gz").is_file()
    assert (patient_dir / "case01_mask_objectID_3.nii.gz").is_file()
    assert (patient_dir / "volumes.txt").is_file()


def test_auto_save_writes_masks_to_unicode_path(tmp_path, monkeypatch):
    meta = _meta(shape=(2, 4, 4))
    mask = np.zeros(meta["shape"], dtype=np.uint8)
    mask[0, 0:2, 0:2] = 1
    mask[1, 1:3, 1:3] = 2
    gui = SimpleNamespace(
        meta=meta,
        patient_id="ignored",
        mask_layer=SimpleNamespace(data=mask),
        metrics=None,
    )
    save_root = tmp_path / "자동 저장 한글"
    save_root.mkdir()
    monkeypatch.setattr(
        io.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(save_root),
    )
    monkeypatch.setattr(io.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(io.QMessageBox, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(io.QMessageBox, "critical", lambda *args, **kwargs: None)
    monkeypatch.setattr(io, "_windows_non_ascii_path", lambda path: True)
    monkeypatch.setattr(
        io.sitk,
        "WriteImage",
        lambda *args, **kwargs: pytest.fail(
            "SimpleITK must be skipped for Windows non-ASCII output paths"
        ),
    )

    io.save_masks_auto(gui)

    patient_dir = save_root / "case01_masks"
    assert (patient_dir / "case01_full_mask.nii.gz").is_file()
    assert (patient_dir / "case01_mask_label_1.nii.gz").is_file()
    assert (patient_dir / "case01_mask_label_2.nii.gz").is_file()
