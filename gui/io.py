"""I/O helpers for importing and saving masks."""
import os
import numpy as np
import traceback
import time
import re
import SimpleITK as sitk
from pathlib import Path
from PIL import Image
from qtpy.QtWidgets import QFileDialog, QMessageBox


_WINDOWS_FORBIDDEN_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_DISPLAY_PATIENT_PREFIX = re.compile(r'^\s*patient\s+\d+\s*:\s*', re.IGNORECASE)
_BINARY_OBJECT_ID = re.compile(r'(?i)_(?:mask_)?(?:objectid|label)_(\d+)(?:\D|$)')
_SUPPORTED_MASK_FILTER = (
    "Medical label images (*.nii *.nii.gz *.nrrd *.mha *.mhd);;"
    "NIfTI (*.nii *.nii.gz);;NRRD (*.nrrd);;MetaImage (*.mha *.mhd)"
)
_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


def _compute_voxel_volume(spacing):
    """Compute per-voxel physical volume from spacing (fallback to 1)."""
    if spacing is None or len(spacing) < 3:
        return 1.0
    try:
        vol = float(spacing[0]) * float(spacing[1]) * float(spacing[2])
        return vol if vol > 0 else 1.0
    except Exception:
        return 1.0


def _numpy_vector(value, default, length):
    if value is None:
        value = default
    if hasattr(value, 'cpu'):
        value = value.cpu().numpy()
    elif hasattr(value, 'numpy'):
        value = value.numpy()
    try:
        result = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        result = np.asarray(default, dtype=float).reshape(-1)
    if result.size < length:
        fallback = np.asarray(default, dtype=float).reshape(-1)
        result = np.concatenate((result, fallback[result.size:length]))
    return result[:length]


def get_source_mask_shape(meta, display_shape):
    """Return the original image array shape in z, y, x order."""
    fallback = tuple(int(v) for v in display_shape)
    shape = meta.get('shape', fallback) if isinstance(meta, dict) else fallback
    if hasattr(shape, 'cpu'):
        shape = shape.cpu().numpy()
    elif hasattr(shape, 'numpy'):
        shape = shape.numpy()
    try:
        shape = tuple(int(v) for v in np.asarray(shape).reshape(-1)[:3])
    except Exception:
        shape = fallback
    if len(shape) != 3 or any(v <= 0 for v in shape):
        return fallback
    return shape


def get_mask_geometry(meta, display_shape):
    shape = get_source_mask_shape(meta, display_shape)
    spacing = _numpy_vector(
        meta.get('spacing') if isinstance(meta, dict) else None,
        (1.0, 1.0, 1.0),
        3,
    )
    origin = _numpy_vector(
        meta.get('origin') if isinstance(meta, dict) else None,
        (0.0, 0.0, 0.0),
        3,
    )
    direction_value = meta.get('direction') if isinstance(meta, dict) else None
    if hasattr(direction_value, 'cpu'):
        direction_value = direction_value.cpu().numpy()
    elif hasattr(direction_value, 'numpy'):
        direction_value = direction_value.numpy()
    try:
        direction = np.asarray(direction_value, dtype=float).reshape(-1)
    except Exception:
        direction = np.asarray([], dtype=float)
    if direction.size != 9:
        direction = np.eye(3).reshape(-1)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Invalid source spacing: {tuple(spacing)}")
    if not np.all(np.isfinite(origin)):
        raise ValueError(f"Invalid source origin: {tuple(origin)}")
    if not np.all(np.isfinite(direction)):
        raise ValueError("Invalid source direction matrix.")
    return shape, spacing, origin, direction


def build_mask_reference_image(meta, display_shape):
    """Build an empty SimpleITK image matching the source image geometry."""
    shape, spacing, origin, direction = get_mask_geometry(meta, display_shape)
    reference = sitk.Image(
        int(shape[2]), int(shape[1]), int(shape[0]), sitk.sitkInt32
    )
    reference.SetSpacing(tuple(float(v) for v in spacing))
    reference.SetOrigin(tuple(float(v) for v in origin))
    reference.SetDirection(tuple(float(v) for v in direction))
    return reference


def _resize_label_slice(label_slice, target_y, target_x):
    source = np.asarray(label_slice, dtype=np.int32)
    if source.shape == (target_y, target_x):
        return source.copy()
    image = Image.fromarray(source, mode='I')
    return np.asarray(image.resize((target_x, target_y), _NEAREST), dtype=np.int32)


def _compact_label_dtype(array):
    array = np.asarray(array)
    maximum = int(array.max()) if array.size else 0
    if maximum <= np.iinfo(np.uint8).max:
        return array.astype(np.uint8, copy=False)
    if maximum <= np.iinfo(np.uint16).max:
        return array.astype(np.uint16, copy=False)
    return array.astype(np.int32, copy=False)


def resize_label_stack(label_stack, target_shape):
    """Nearest-neighbor resize of a z, y, x label stack."""
    source = np.asarray(label_stack)
    target_shape = tuple(int(v) for v in target_shape)
    if source.ndim != 3 or len(target_shape) != 3:
        raise ValueError("Label masks must be 3D arrays in z, y, x order.")
    if source.shape[0] != target_shape[0]:
        raise ValueError(
            f"Slice count mismatch: mask has {source.shape[0]}, expected {target_shape[0]}."
        )
    if source.shape == target_shape:
        return _compact_label_dtype(source.copy())
    resized = np.empty(target_shape, dtype=np.int32)
    for z_index in range(target_shape[0]):
        resized[z_index] = _resize_label_slice(
            source[z_index], target_shape[1], target_shape[2]
        )
    return _compact_label_dtype(resized)


def source_mask_to_display(source_mask, display_shape):
    return resize_label_stack(source_mask, display_shape)


def display_mask_to_source(display_mask, source_shape):
    return np.asarray(resize_label_stack(display_mask, source_shape), dtype=np.int32)


def sync_source_mask_slices(source_mask, display_mask, slice_indices):
    """Update selected source-grid slices from the editable display mask."""
    source = np.asarray(source_mask, dtype=np.int32).copy()
    display = np.asarray(display_mask)
    if source.ndim != 3 or display.ndim != 3 or source.shape[0] != display.shape[0]:
        raise ValueError("Source and display masks must have the same slice count.")
    for z_index in sorted({int(v) for v in slice_indices}):
        if 0 <= z_index < source.shape[0]:
            source[z_index] = _resize_label_slice(
                display[z_index], source.shape[1], source.shape[2]
            )
    return source


def compute_label_volume_entries(source_mask, spacing):
    """Return (label, voxel_count, volume_mm3) entries from a source-grid mask."""
    labels, counts = np.unique(np.asarray(source_mask), return_counts=True)
    voxel_volume = _compute_voxel_volume(spacing)
    return [
        (int(label), int(count), float(count) * voxel_volume)
        for label, count in zip(labels, counts)
        if int(label) > 0
    ]


def _validate_label_array(array, path):
    data = np.asarray(array)
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a 3D label image, got shape {data.shape}.")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: mask contains NaN or infinite values.")
    rounded = np.rint(data)
    if not np.allclose(data, rounded, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"{path}: mask contains non-integer values and appears to be a probability map."
        )
    minimum = float(np.min(rounded)) if rounded.size else 0.0
    maximum = float(np.max(rounded)) if rounded.size else 0.0
    if minimum < 0.0:
        raise ValueError(f"{path}: mask labels must be non-negative integers.")
    if maximum > float(np.iinfo(np.int32).max):
        raise ValueError(f"{path}: mask label exceeds the supported int32 range.")
    return rounded.astype(np.int32)


def _geometry_matches(image, reference):
    return (
        image.GetDimension() == 3
        and image.GetSize() == reference.GetSize()
        and np.allclose(image.GetSpacing(), reference.GetSpacing(), rtol=1e-5, atol=1e-5)
        and np.allclose(image.GetOrigin(), reference.GetOrigin(), rtol=0.0, atol=1e-3)
        and np.allclose(image.GetDirection(), reference.GetDirection(), rtol=0.0, atol=1e-5)
    )


def _geometry_description(image):
    return (
        f"size={image.GetSize()}, spacing={tuple(round(v, 6) for v in image.GetSpacing())}, "
        f"origin={tuple(round(v, 3) for v in image.GetOrigin())}, "
        f"direction={tuple(round(v, 4) for v in image.GetDirection())}"
    )


def _binary_object_id(path, current_obj_id):
    match = _BINARY_OBJECT_ID.search(Path(path).name)
    if match:
        return int(match.group(1))
    return int(current_obj_id)


def _is_nifti_path(path):
    lower = str(path).lower()
    return lower.endswith('.nii') or lower.endswith('.nii.gz')


def _compact_exception(error, limit=700):
    message = ' '.join(str(error).split()) or error.__class__.__name__
    if len(message) > limit:
        return message[: limit - 3] + '...'
    return message


def _read_nifti_with_nibabel(path):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "nibabel is not installed; install it to read NIfTI files from this path."
        ) from exc

    nib_image = nib.load(str(path))
    if len(nib_image.shape) != 3:
        raise ValueError(
            f"expected a 3D NIfTI label image, got shape {nib_image.shape}."
        )

    affine = np.asarray(nib_image.affine, dtype=np.float64)
    if (
        affine.shape != (4, 4)
        or not np.all(np.isfinite(affine))
        or not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-6)
    ):
        raise ValueError("NIfTI affine is missing or invalid.")

    # NIfTI uses RAS physical coordinates; SimpleITK uses LPS.
    ras_to_lps = np.diag((-1.0, -1.0, 1.0, 1.0))
    lps_affine = ras_to_lps @ affine
    axis_columns = lps_affine[:3, :3]
    spacing = np.linalg.norm(axis_columns, axis=0)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise ValueError("NIfTI affine contains invalid voxel spacing.")

    direction = axis_columns / spacing[np.newaxis, :]
    if not np.allclose(
        direction.T @ direction, np.eye(3), rtol=0.0, atol=1e-4
    ):
        raise ValueError("NIfTI affine contains shear, which is not supported.")
    if not np.isclose(abs(np.linalg.det(direction)), 1.0, rtol=0.0, atol=1e-4):
        raise ValueError("NIfTI affine contains an invalid direction matrix.")

    data_xyz = _validate_label_array(np.asanyarray(nib_image.dataobj), path)
    data_zyx = np.transpose(data_xyz, (2, 1, 0))
    image = sitk.GetImageFromArray(data_zyx)
    image.SetSpacing(tuple(float(value) for value in spacing))
    image.SetOrigin(tuple(float(value) for value in lps_affine[:3, 3]))
    image.SetDirection(tuple(float(value) for value in direction.reshape(-1)))
    return image


def _windows_non_ascii_path(path):
    if os.name != 'nt':
        return False
    try:
        os.fspath(path).encode('ascii')
        return False
    except UnicodeEncodeError:
        return True


def _write_nifti_with_nibabel(image, path):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "nibabel is not installed; install it to write NIfTI files to this path."
        ) from exc

    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Only scalar 3D images can be written as NIfTI masks.")

    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    if (
        spacing.shape != (3,)
        or origin.shape != (3,)
        or not np.all(np.isfinite(spacing))
        or not np.all(np.isfinite(origin))
        or not np.all(np.isfinite(direction))
        or np.any(spacing <= 0.0)
    ):
        raise ValueError("SimpleITK image geometry is invalid for NIfTI output.")
    if not np.allclose(
        direction.T @ direction, np.eye(3), rtol=0.0, atol=1e-4
    ):
        raise ValueError("SimpleITK image direction contains unsupported shear.")

    lps_affine = np.eye(4, dtype=np.float64)
    lps_affine[:3, :3] = direction * spacing[np.newaxis, :]
    lps_affine[:3, 3] = origin
    lps_to_ras = np.diag((-1.0, -1.0, 1.0, 1.0))
    ras_affine = lps_to_ras @ lps_affine

    data_zyx = sitk.GetArrayFromImage(image)
    data_xyz = np.transpose(data_zyx, (2, 1, 0))
    nib_image = nib.Nifti1Image(data_xyz, ras_affine)
    nib_image.header.set_xyzt_units('mm')
    nib_image.set_qform(ras_affine, code=1)
    nib_image.set_sform(ras_affine, code=1)
    nib.save(nib_image, str(path))


def _write_nifti_image(image, path):
    """Write NIfTI through SimpleITK, with a Unicode-safe nibabel fallback."""
    path = Path(path)
    if not _is_nifti_path(path):
        raise ValueError(f"Unsupported NIfTI output extension: {path.name}")

    sitk_error = None
    skip_simpleitk = _windows_non_ascii_path(path)
    if not skip_simpleitk:
        try:
            sitk.WriteImage(image, str(path))
            return 'simpleitk'
        except Exception as exc:
            sitk_error = exc

    try:
        _write_nifti_with_nibabel(image, path)
        return 'nibabel'
    except Exception as nibabel_error:
        if skip_simpleitk:
            sitk_message = "skipped for a Windows non-ASCII path"
        else:
            sitk_message = _compact_exception(sitk_error)
        raise RuntimeError(
            f"Could not write {path.name}. SimpleITK: {sitk_message}. "
            f"nibabel fallback: {_compact_exception(nibabel_error)}"
        ) from nibabel_error


def _read_import_mask(path):
    try:
        image = sitk.ReadImage(str(path))
    except Exception as sitk_error:
        if not _is_nifti_path(path):
            raise RuntimeError(
                f"Could not read {Path(path).name}. "
                f"SimpleITK: {_compact_exception(sitk_error)}"
            ) from sitk_error
        try:
            image = _read_nifti_with_nibabel(path)
        except Exception as nibabel_error:
            raise RuntimeError(
                f"Could not read {Path(path).name}. "
                f"SimpleITK: {_compact_exception(sitk_error)} "
                f"nibabel fallback: {_compact_exception(nibabel_error)}"
            ) from nibabel_error
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"{path}: only scalar 3D label images are supported.")
    _validate_label_array(sitk.GetArrayFromImage(image), path)
    return image


def _navigation_patient_path(gui):
    navigation = getattr(gui, 'navigation_manager', None)
    patient_list = getattr(navigation, 'patient_list', None)
    patient_index = getattr(navigation, 'current_patient_idx', None)
    if not isinstance(patient_index, int) or not patient_list:
        return None
    if patient_index < 0 or patient_index >= len(patient_list):
        return None
    patient = patient_list[patient_index]
    if isinstance(patient, (list, tuple)):
        return patient[0] if patient else None
    return patient


def _mask_directory_for_source(source_path):
    if not source_path:
        return None
    try:
        source = Path(str(source_path)).expanduser()
        if not source.exists():
            return None
        source = source.resolve()
    except (OSError, RuntimeError, ValueError):
        return None

    if source.is_dir():
        study_directory = source
        mask_directory = study_directory / f"{study_directory.name}_masks"
    elif _is_nifti_path(source):
        study_directory = source.parent
        mask_directory = study_directory / f"{_strip_nii_extension(source.name)}_masks"
    else:
        study_directory = source.parent
        mask_directory = study_directory / f"{study_directory.name}_masks"

    if mask_directory.is_dir():
        return mask_directory
    return study_directory if study_directory.is_dir() else None


def get_mask_load_initial_directory(gui):
    """Resolve the most relevant existing directory for the mask picker."""
    meta = getattr(gui, 'meta', {}) or {}
    for source_path in (meta.get('source_path'), _navigation_patient_path(gui)):
        directory = _mask_directory_for_source(source_path)
        if directory is not None:
            return str(directory)
    return os.getcwd()


def _source_container_directory(source_path):
    if not source_path:
        return None
    try:
        source = Path(str(source_path)).expanduser()
        if not source.exists():
            return None
        source = source.resolve()
    except (OSError, RuntimeError, ValueError):
        return None

    directory = source if source.is_dir() else source.parent
    return directory if directory.is_dir() else None


def get_mask_save_initial_directory(gui):
    """Resolve the source container used as the mask save dialog default."""
    meta = getattr(gui, 'meta', {}) or {}
    for source_path in (meta.get('source_path'), _navigation_patient_path(gui)):
        directory = _source_container_directory(source_path)
        if directory is not None:
            return str(directory)
    return os.getcwd()


def _prepare_imported_mask(image, path, reference, current_obj_id):
    geometry_changed = not _geometry_matches(image, reference)
    if geometry_changed:
        image = sitk.Resample(
            image,
            reference,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            image.GetPixelID(),
        )
    data = _validate_label_array(sitk.GetArrayFromImage(image), path)
    if not np.any(data):
        raise ValueError(
            f"{path}: mask is empty after alignment to the source image; the physical geometries may not overlap."
        )
    unique = np.unique(data)
    if np.array_equal(unique, [0, 1]) or np.array_equal(unique, [1]):
        object_id = _binary_object_id(path, current_obj_id)
        data = np.where(data > 0, object_id, 0).astype(np.int32)
    return data, geometry_changed


def _confirm_geometry_resampling(gui, mismatches, reference):
    details = [
        "The selected masks do not exactly match the source image geometry.",
        "They must be resampled to the source grid with nearest-neighbor interpolation.",
        "",
        f"Source: {_geometry_description(reference)}",
    ]
    for path, image in mismatches[:4]:
        details.append(f"{Path(path).name}: {_geometry_description(image)}")
    if len(mismatches) > 4:
        details.append(f"... and {len(mismatches) - 4} more file(s)")
    details.extend(("", "Continue with physical-coordinate resampling?"))
    answer = QMessageBox.question(
        gui,
        "Mask Geometry Mismatch",
        "\n".join(details),
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    return answer == QMessageBox.Yes


def _choose_import_mode(gui):
    dialog = QMessageBox(gui)
    dialog.setIcon(QMessageBox.Question)
    dialog.setWindowTitle("Existing Mask")
    dialog.setText("A mask is already loaded. Replace it or merge into empty voxels?")
    replace_button = dialog.addButton("Replace", QMessageBox.AcceptRole)
    merge_button = dialog.addButton("Merge", QMessageBox.ActionRole)
    dialog.addButton(QMessageBox.Cancel)
    if hasattr(dialog, 'exec'):
        dialog.exec()
    else:
        dialog.exec_()
    clicked = dialog.clickedButton()
    if clicked is replace_button:
        return 'replace'
    if clicked is merge_button:
        return 'merge'
    return 'cancel'


def merge_source_masks_preserving_existing(current, imported):
    """Merge imported labels into background voxels and report label conflicts."""
    current = np.asarray(current, dtype=np.int32)
    imported = np.asarray(imported, dtype=np.int32)
    if current.shape != imported.shape:
        raise ValueError(f"Mask shape mismatch: {current.shape} != {imported.shape}")
    conflict = (current > 0) & (imported > 0) & (current != imported)
    combined = current.copy()
    fill = (combined == 0) & (imported > 0)
    combined[fill] = imported[fill]
    return combined, int(np.count_nonzero(conflict))


def load_masks_manual(gui):
    """Load one or more source-geometry label images into the manual GUI."""
    initial_directory = get_mask_load_initial_directory(gui)
    paths, _ = QFileDialog.getOpenFileNames(
        gui, "Load Mask Images", initial_directory, _SUPPORTED_MASK_FILTER
    )
    if not paths:
        return False

    try:
        display_shape = tuple(int(v) for v in gui.mask_layer.data.shape)
        reference = build_mask_reference_image(gui.meta, display_shape)
        images = [(path, _read_import_mask(path)) for path in paths]
        mismatches = [
            (path, image)
            for path, image in images
            if not _geometry_matches(image, reference)
        ]
        if mismatches and not _confirm_geometry_resampling(gui, mismatches, reference):
            return False

        imported = np.zeros(tuple(reversed(reference.GetSize())), dtype=np.int32)
        resampled_count = 0
        for path, image in images:
            prepared, geometry_changed = _prepare_imported_mask(
                image, path, reference, getattr(gui, 'current_obj_id', 1)
            )
            conflict = (imported > 0) & (prepared > 0) & (imported != prepared)
            if np.any(conflict):
                raise ValueError(
                    f"{Path(path).name}: {int(np.count_nonzero(conflict))} voxels overlap "
                    "labels from another selected file. Import was cancelled."
                )
            imported[prepared > 0] = prepared[prepared > 0]
            resampled_count += int(geometry_changed)

        if hasattr(gui, 'flush_source_mask_updates'):
            gui.flush_source_mask_updates()
        current = getattr(gui, 'source_mask_data', None)
        if current is None or np.asarray(current).shape != imported.shape:
            current = display_mask_to_source(gui.mask_layer.data, imported.shape)
        else:
            current = np.asarray(current, dtype=np.int32)

        import_mode = 'replace'
        ignored_conflicts = 0
        if np.any(current):
            import_mode = _choose_import_mode(gui)
            if import_mode == 'cancel':
                return False
        if import_mode == 'merge':
            combined, ignored_conflicts = merge_source_masks_preserving_existing(
                current, imported
            )
        else:
            combined = imported

        if hasattr(gui, 'set_source_mask_data'):
            gui.set_source_mask_data(combined, record_history=True)
        else:
            gui.source_mask_data = combined.copy()
            gui._set_mask_data(source_mask_to_display(combined, display_shape))

        labels = np.unique(combined)
        labels = labels[labels > 0]
        if labels.size:
            maximum = max(100, int(labels.max()))
            gui.oid_spin.setRange(1, maximum)
            gui.oid_spin.setValue(int(labels.min()))
        if hasattr(gui, '_update_object_id_text'):
            gui._update_object_id_text()

        message = (
            f"Loaded {len(paths)} mask file(s).\n"
            f"Labels: {', '.join(str(int(v)) for v in labels)}\n"
            f"Source shape: {combined.shape}"
        )
        if resampled_count:
            message += f"\nResampled to source geometry: {resampled_count} file(s)"
        if ignored_conflicts:
            message += f"\nMerge conflicts kept from current mask: {ignored_conflicts} voxels"
        QMessageBox.information(gui, "Masks Loaded", message)
        if hasattr(gui, 'metrics') and gui.metrics and gui.metrics.is_active():
            gui.metrics.add_event(
                'load_masks_manual',
                file_count=len(paths),
                label_count=int(labels.size),
                resampled_count=resampled_count,
                mode=import_mode,
            )
        return True
    except Exception as exc:
        print(f"Mask load failed: {_compact_exception(exc)}")
        QMessageBox.critical(gui, "Mask Load Failed", str(exc))
        return False


def _save_volume_report(patient_dir: Path, entries):
    """Write a simple text report of voxel counts and volumes."""
    if not entries:
        return
    report_path = patient_dir / "volumes.txt"
    lines = ["object_id\tvoxel_count\tvolume_mm3\n"]
    lines += [f"{obj_id}\t{vox}\t{vol:.3f}\n" for obj_id, vox, vol in entries]
    report_path.write_text("".join(lines))


def _strip_nii_extension(name):
    lower = name.lower()
    if lower.endswith('.nii.gz'):
        return name[:-7]
    if lower.endswith('.nii'):
        return name[:-4]
    return name


def _safe_source_stem(value):
    try:
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
    except Exception:
        value = None
    if value is None:
        return "Unknown"

    text = str(value).strip()
    if not text:
        return "Unknown"
    text = _DISPLAY_PATIENT_PREFIX.sub('', text).strip()
    if not text:
        return "Unknown"

    normalized = text.replace('\\', '/').rstrip('/ ')
    basename = normalized.rsplit('/', 1)[-1] if normalized else text
    stem = _strip_nii_extension(basename)
    stem = _WINDOWS_FORBIDDEN_CHARS.sub('_', stem).strip(' .')
    return stem or "Unknown"


def _get_save_patient_stem(gui):
    meta = getattr(gui, 'meta', None)
    patient = None
    if isinstance(meta, dict):
        patient = meta.get('patient')
    patient_stem = _safe_source_stem(patient)
    if patient_stem != "Unknown":
        return patient_stem
    return _safe_source_stem(getattr(gui, 'patient_id', None))


def save_masks_auto(gui):
    """Save masks from auto GUI."""
    try:
        initial_directory = get_mask_save_initial_directory(gui)
        save_dir = QFileDialog.getExistingDirectory(
            gui, "Select Mask Save Folder", initial_directory
        )
        if not save_dir:
            return
        save_dir = Path(save_dir)
        patient_name = _get_save_patient_stem(gui)
        patient_dir = save_dir / f"{patient_name}_masks"
        patient_dir.mkdir(exist_ok=True)

        mask_data = gui.mask_layer.data
        if mask_data.sum() == 0:
            QMessageBox.warning(gui, "Save Failed", "No masks to save.")
            if hasattr(gui, 'metrics') and gui.metrics and gui.metrics.is_active():
                gui.metrics.add_event('save_skipped', reason='no_masks')
            return

        save_start = time.time()

        spacing = gui.meta.get('spacing', [1.0, 1.0, 1.0])
        origin = gui.meta.get('origin', [0.0, 0.0, 0.0])
        direction = gui.meta.get('direction', np.eye(3).flatten())

        if hasattr(spacing, 'cpu'):
            spacing = spacing.cpu().numpy()
        if hasattr(origin, 'cpu'):
            origin = origin.cpu().numpy()
        if hasattr(direction, 'cpu'):
            direction = direction.cpu().numpy()

        spacing = np.array(spacing).flatten().astype(float)
        origin = np.array(origin).flatten().astype(float)
        direction = np.array(direction).flatten().astype(float)

        if mask_data.dtype != np.uint8:
            mask_data = (mask_data * 255).astype(np.uint8) if mask_data.max() <= 1 else mask_data.astype(np.uint8)

        mask_sitk = sitk.GetImageFromArray(mask_data)
        try:
            if len(spacing) >= 3:
                mask_sitk.SetSpacing(tuple(spacing[:3]))
            elif len(spacing) == 2:
                mask_sitk.SetSpacing((float(spacing[0]), float(spacing[1]), 1.0))
            else:
                mask_sitk.SetSpacing((1.0, 1.0, 1.0))

            if len(origin) >= 3:
                mask_sitk.SetOrigin(tuple(origin[:3]))
            elif len(origin) == 2:
                mask_sitk.SetOrigin((float(origin[0]), float(origin[1]), 0.0))
            else:
                mask_sitk.SetOrigin((0.0, 0.0, 0.0))

            if len(direction) == 9:
                mask_sitk.SetDirection(tuple(direction))
            elif len(direction) == 4:
                mask_sitk.SetDirection(tuple(direction))
            else:
                mask_sitk.SetDirection(tuple(np.eye(3).flatten()))
        except Exception as e:
            print(f"❌ Error setting geometric information: {e}")
            mask_sitk.SetSpacing((1.0, 1.0, 1.0))
            mask_sitk.SetOrigin((0.0, 0.0, 0.0))
            mask_sitk.SetDirection(tuple(np.eye(3).flatten()))

        mask_path = patient_dir / f"{patient_name}_full_mask.nii.gz"
        _write_nifti_image(mask_sitk, mask_path)

        unique_labels = np.unique(mask_data)
        unique_labels = unique_labels[unique_labels > 0]
        saved_count = 0
        volume_entries = []
        voxel_volume = _compute_voxel_volume(spacing)
        for label in unique_labels:
            single_mask = (mask_data == label).astype(np.uint8)
            if single_mask.sum() == 0:
                continue
            single_mask_sitk = sitk.GetImageFromArray(single_mask)
            try:
                if len(spacing) >= 3:
                    single_mask_sitk.SetSpacing(tuple(spacing[:3]))
                else:
                    single_mask_sitk.SetSpacing((1.0, 1.0, 1.0))
                if len(origin) >= 3:
                    single_mask_sitk.SetOrigin(tuple(origin[:3]))
                else:
                    single_mask_sitk.SetOrigin((0.0, 0.0, 0.0))
                if len(direction) == 9:
                    single_mask_sitk.SetDirection(tuple(direction))
                elif len(direction) == 4:
                    single_mask_sitk.SetDirection(tuple(direction))
                else:
                    single_mask_sitk.SetDirection(tuple(np.eye(3).flatten()))
            except Exception as e:
                print(f"Label {label} geometric information setting error: {e}")
                single_mask_sitk.SetSpacing((1.0, 1.0, 1.0))
                single_mask_sitk.SetOrigin((0.0, 0.0, 0.0))
                single_mask_sitk.SetDirection(tuple(np.eye(3).flatten()))

            single_mask_path = patient_dir / f"{patient_name}_mask_label_{label}.nii.gz"
            _write_nifti_image(single_mask_sitk, single_mask_path)
            voxels = int(single_mask.sum())
            volume_entries.append((label, voxels, voxels * voxel_volume))
            saved_count += 1

        _save_volume_report(patient_dir, volume_entries)

        success_msg = (
            f"Mask save completed!\n\n"
            f"Save location: {patient_dir}\n"
            f"Total masks: {mask_path.name}\n"
            f"Individual masks: {saved_count} labels\n"
            f"Volumes: volumes.txt (object_id, voxel_count, volume_mm3)\n"
            f"Geometric information:\n"
            f"  Spacing: {spacing[:3] if len(spacing) >= 3 else spacing}\n"
            f"  Origin: {origin[:3] if len(origin) >= 3 else origin}\n"
            f"  Direction: {'3x3 Matrix applied' if len(direction)==9 else 'Applied'}\n"
        )
        QMessageBox.information(gui, "Save Completed", success_msg)
        if hasattr(gui, 'metrics') and gui.metrics and gui.metrics.is_active():
            gui.metrics.record_stage(
                'save_masks_auto',
                save_start,
                time.time(),
                patient_id=str(patient_name),
                labels_saved=int(saved_count),
                slice_count=int(mask_data.shape[0]) if hasattr(mask_data, 'shape') else None,
                save_dir=str(patient_dir),
            )
            gui.metrics.finalize({'mode': 'auto', 'save_dir': str(patient_dir)})
    except Exception as e:
        error_msg = f"Error occurred while saving mask:\n{str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        QMessageBox.critical(gui, "Save Failed", error_msg)
        if hasattr(gui, 'metrics') and gui.metrics and gui.metrics.is_active():
            gui.metrics.add_event('save_error', message=str(e))


def save_masks_manual(gui):
    """Save the canonical source-grid mask from the manual GUI."""
    try:
        initial_directory = get_mask_save_initial_directory(gui)
        save_dir = QFileDialog.getExistingDirectory(
            gui, "Select Mask Save Folder", initial_directory
        )
        if not save_dir:
            return
        save_dir = Path(save_dir)
        patient_name = _get_save_patient_stem(gui)
        patient_dir = save_dir / f"{patient_name}_masks"
        patient_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(gui, 'flush_source_mask_updates'):
            gui.flush_source_mask_updates()
        display_mask = np.asarray(gui.mask_layer.data)
        source_shape, spacing, _, _ = get_mask_geometry(gui.meta, display_mask.shape)
        source_mask = getattr(gui, 'source_mask_data', None)
        if source_mask is None or np.asarray(source_mask).shape != source_shape:
            source_mask = display_mask_to_source(display_mask, source_shape)
        source_mask = _validate_label_array(source_mask, "Current mask")
        if not np.any(source_mask):
            QMessageBox.warning(gui, "Save Failed", "No masks to save.")
            if hasattr(gui, 'metrics') and gui.metrics and gui.metrics.is_active():
                gui.metrics.add_event('save_skipped', reason='no_masks')
            return

        save_start = time.time()
        reference = build_mask_reference_image(gui.meta, display_mask.shape)
        full_mask_sitk = sitk.GetImageFromArray(source_mask.astype(np.int32))
        full_mask_sitk.CopyInformation(reference)
        full_mask_path = patient_dir / f"{patient_name}_full_mask.nii.gz"
        _write_nifti_image(full_mask_sitk, full_mask_path)

        volume_entries = compute_label_volume_entries(source_mask, spacing)
        expected_object_paths = set()
        for object_id, _, _ in volume_entries:
            object_mask = (source_mask == object_id).astype(np.uint8)
            object_mask_sitk = sitk.GetImageFromArray(object_mask)
            object_mask_sitk.CopyInformation(reference)
            object_mask_path = patient_dir / f"{patient_name}_mask_objectID_{object_id}.nii.gz"
            _write_nifti_image(object_mask_sitk, object_mask_path)
            expected_object_paths.add(object_mask_path.resolve())

        stale_pattern = f"{patient_name}_mask_objectID_*.nii.gz"
        for stale_path in patient_dir.glob(stale_pattern):
            if stale_path.resolve() not in expected_object_paths:
                stale_path.unlink()

        _save_volume_report(patient_dir, volume_entries)
        message = (
            f"Masks have been successfully saved!\n"
            f"Save location: {patient_dir}\n"
            f"Full mask: {full_mask_path.name}\n"
            f"Individual masks for each Object ID: {len(volume_entries)} files\n"
            f"Volumes: volumes.txt (object_id, voxel_count, volume_mm3)\n"
            f"Source grid: {source_mask.shape}"
        )
        QMessageBox.information(gui, "Save Complete", message)
        if hasattr(gui, 'metrics') and gui.metrics and gui.metrics.is_active():
            gui.metrics.record_stage(
                'save_masks_manual',
                save_start,
                time.time(),
                patient_id=str(patient_name),
                labels_saved=len(volume_entries),
                slice_count=int(source_mask.shape[0]),
                save_dir=str(patient_dir),
            )
            gui.metrics.finalize({'mode': 'manual', 'save_dir': str(patient_dir)})
    except Exception as e:
        QMessageBox.critical(gui, "Save Failed", f"An error occurred while saving masks:\n{str(e)}")
        traceback.print_exc()
        if hasattr(gui, 'metrics') and gui.metrics and gui.metrics.is_active():
            gui.metrics.add_event('save_error', message=str(e))
