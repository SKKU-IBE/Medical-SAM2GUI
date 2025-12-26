"""I/O helpers for saving masks."""
import numpy as np
import torch
import traceback
import SimpleITK as sitk
from pathlib import Path
from PIL import Image
from PyQt5.QtWidgets import QFileDialog, QMessageBox


def _compute_voxel_volume(spacing):
    """Compute per-voxel physical volume from spacing (fallback to 1)."""
    if spacing is None or len(spacing) < 3:
        return 1.0
    try:
        vol = float(spacing[0]) * float(spacing[1]) * float(spacing[2])
        return vol if vol > 0 else 1.0
    except Exception:
        return 1.0


def _save_volume_report(patient_dir: Path, entries):
    """Write a simple text report of voxel counts and volumes."""
    if not entries:
        return
    report_path = patient_dir / "volumes.txt"
    lines = ["object_id\tvoxel_count\tvolume_mm3\n"]
    lines += [f"{obj_id}\t{vox}\t{vol:.3f}\n" for obj_id, vox, vol in entries]
    report_path.write_text("".join(lines))


def save_masks_auto(gui):
    """Save masks from auto GUI."""
    try:
        save_dir = QFileDialog.getExistingDirectory(gui, "Select Mask Save Folder", ".")
        if not save_dir:
            return
        save_dir = Path(save_dir)
        patient_dir = save_dir / f"{gui.patient_id}_masks"
        patient_dir.mkdir(exist_ok=True)

        mask_data = gui.mask_layer.data
        if mask_data.sum() == 0:
            QMessageBox.warning(gui, "Save Failed", "No masks to save.")
            return

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

        mask_path = patient_dir / f"{gui.patient_id}_full_mask.nii.gz"
        sitk.WriteImage(mask_sitk, str(mask_path))

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

            single_mask_path = patient_dir / f"{gui.patient_id}_mask_label_{label}.nii.gz"
            sitk.WriteImage(single_mask_sitk, str(single_mask_path))
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
    except Exception as e:
        error_msg = f"Error occurred while saving mask:\n{str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        QMessageBox.critical(gui, "Save Failed", error_msg)


def save_masks_manual(gui):
    """Save masks from manual GUI (per object IDs)."""
    try:
        save_dir = QFileDialog.getExistingDirectory(gui, "Select Mask Save Folder", ".")
        if not save_dir:
            return
        save_dir = Path(save_dir)
        patient_name = gui._get_patient_display_name()
        patient_dir = save_dir / f"{patient_name}_masks"
        patient_dir.mkdir(exist_ok=True)

        mask3d_resized = gui.mask_layer.data
        if mask3d_resized.sum() == 0:
            QMessageBox.warning(gui, "Save Failed", "No masks to save.")
            return

        ori_shape = gui.meta.get('shape', mask3d_resized.shape)
        if hasattr(ori_shape, 'numpy'):
            ori_shape = ori_shape.numpy()
        elif isinstance(ori_shape, torch.Tensor):
            ori_shape = ori_shape.cpu().numpy()
        else:
            ori_shape = np.array(ori_shape)
        ori_shape = np.array(ori_shape).flatten()

        resize_needed = False
        if len(ori_shape) >= 3:
            ori_z, ori_y, ori_x = int(ori_shape[0]), int(ori_shape[1]), int(ori_shape[2])
            if (ori_z, ori_y, ori_x) != mask3d_resized.shape:
                resize_needed = True
        else:
            ori_z, ori_y, ori_x = mask3d_resized.shape

        spacing_raw = gui.meta.get('spacing', [1.0, 1.0, 1.0])
        origin_raw = gui.meta.get('origin', [0.0, 0.0, 0.0])
        direction_raw = gui.meta.get('direction', np.eye(3).flatten())

        spacing = spacing_raw.cpu().numpy() if hasattr(spacing_raw, 'cpu') else np.array(spacing_raw)
        origin = origin_raw.cpu().numpy() if hasattr(origin_raw, 'cpu') else np.array(origin_raw)
        direction = direction_raw.cpu().numpy() if hasattr(direction_raw, 'cpu') else np.array(direction_raw)

        spacing = np.array(spacing).flatten().astype(float)
        origin = np.array(origin).flatten().astype(float)
        direction = np.array(direction).flatten().astype(float)

        full_mask = (mask3d_resized > 0).astype(np.uint8)
        if resize_needed:
            full_mask_original = np.zeros((ori_z, ori_y, ori_x), dtype=np.uint8)
            for z in range(min(full_mask.shape[0], ori_z)):
                if full_mask[z].sum() > 0:
                    mask_slice_pil = Image.fromarray(full_mask[z].astype(np.uint8))
                    mask_slice_resized = mask_slice_pil.resize((ori_x, ori_y), Image.NEAREST)
                    full_mask_original[z] = np.array(mask_slice_resized)
        else:
            full_mask_original = full_mask

        full_mask_sitk = sitk.GetImageFromArray(full_mask_original)
        try:
            if len(spacing) >= 3:
                full_mask_sitk.SetSpacing(tuple(spacing[:3]))
            if len(origin) >= 3:
                full_mask_sitk.SetOrigin(tuple(origin[:3]))
            if len(direction) == 9:
                full_mask_sitk.SetDirection(tuple(direction))
            elif len(direction) == 4:
                full_mask_sitk.SetDirection(tuple(direction))
            else:
                full_mask_sitk.SetDirection(tuple(np.eye(3).flatten()))
        except Exception as e:
            print(f"❌ Full mask geometric information setting error: {e}")
            full_mask_sitk.SetSpacing((1.0, 1.0, 1.0))
            full_mask_sitk.SetOrigin((0.0, 0.0, 0.0))
            full_mask_sitk.SetDirection(tuple(np.eye(3).flatten()))

        full_mask_path = patient_dir / f"{patient_name}_full_mask.nii.gz"
        sitk.WriteImage(full_mask_sitk, str(full_mask_path))

        saved_full_mask = sitk.ReadImage(str(full_mask_path))
        saved_direction = np.array(saved_full_mask.GetDirection())
        if len(saved_direction) == 9 and len(direction) == 9:
            diff = np.abs(saved_direction - direction).max()
            if diff >= 1e-6:
                print("⚠️  Full mask Direction slight difference detected")

        unique_labels = np.unique(mask3d_resized)
        unique_labels = unique_labels[unique_labels > 0]
        saved_count = 0
        volume_entries = []
        voxel_volume = _compute_voxel_volume(spacing)
        for object_id in unique_labels:
            object_mask = (mask3d_resized == object_id).astype(np.uint8)
            if object_mask.sum() == 0:
                continue
            if resize_needed:
                object_mask_original = np.zeros((ori_z, ori_y, ori_x), dtype=np.uint8)
                for z in range(min(object_mask.shape[0], ori_z)):
                    if object_mask[z].sum() > 0:
                        mask_slice_pil = Image.fromarray(object_mask[z].astype(np.uint8))
                        mask_slice_resized = mask_slice_pil.resize((ori_x, ori_y), Image.NEAREST)
                        object_mask_original[z] = np.array(mask_slice_resized)
            else:
                object_mask_original = object_mask

            object_mask_sitk = sitk.GetImageFromArray(object_mask_original)
            try:
                if len(spacing) >= 3:
                    object_mask_sitk.SetSpacing(tuple(spacing[:3]))
                if len(origin) >= 3:
                    object_mask_sitk.SetOrigin(tuple(origin[:3]))
                if len(direction) == 9:
                    object_mask_sitk.SetDirection(tuple(direction))
                elif len(direction) == 4:
                    object_mask_sitk.SetDirection(tuple(direction))
                else:
                    object_mask_sitk.SetDirection(tuple(np.eye(3).flatten()))
            except Exception as e:
                print(f"❌ Object {object_id} geometric information setting error: {e}")
                object_mask_sitk.SetSpacing((1.0, 1.0, 1.0))
                object_mask_sitk.SetOrigin((0.0, 0.0, 0.0))
                object_mask_sitk.SetDirection(tuple(np.eye(3).flatten()))

            object_mask_path = patient_dir / f"{patient_name}_mask_objectID_{object_id}.nii.gz"
            sitk.WriteImage(object_mask_sitk, str(object_mask_path))
            voxels = int(object_mask_original.sum())
            volume_entries.append((object_id, voxels, voxels * voxel_volume))
            saved_count += 1

        _save_volume_report(patient_dir, volume_entries)

        message = (
            f"Masks have been successfully saved!\n"
            f"Save location: {patient_dir}\n"
            f"Full mask: {full_mask_path.name}\n"
            f"Individual masks for each Object ID: {saved_count} files\n"
            f"Volumes: volumes.txt (object_id, voxel_count, volume_mm3)"
        )
        if resize_needed:
            message += f"\nRestored to original size: {full_mask_original.shape}"

        QMessageBox.information(gui, "Save Complete", message)
    except Exception as e:
        QMessageBox.critical(gui, "Save Failed", f"An error occurred while saving masks:\n{str(e)}")
        traceback.print_exc()
