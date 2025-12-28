"""Rendering helpers extracted from GUI classes."""
import traceback
import numpy as np
from skimage.measure import marching_cubes
import pyvista as pv
from PyQt5.QtWidgets import QMessageBox


def render_auto_volume(gui):
    """Render 3D volume for auto GUI. Expects gui to have mask_layer, meta, patient_id."""
    mask3d = (gui.mask_layer.data > 0).astype(np.uint8)
    n_slices = mask3d.shape[0]

    print("=== Pure 3D Volume Rendering (No Interpolation/Smoothing) ===")
    print(f"Original mask3d shape: {mask3d.shape}")
    print(f"Original mask3d dtype: {mask3d.dtype}")
    print(f"Original mask3d range: {mask3d.min()} - {mask3d.max()}")
    print(f"Original non-zero voxels: {mask3d.sum()}")

    if n_slices < 2:
        QMessageBox.warning(gui, 'Insufficient Data', f'Need at least 2 slices for 3D rendering. Current: {n_slices}')
        return
    if mask3d.sum() == 0:
        QMessageBox.warning(gui, 'Empty Mask', 'No mask data to render!')
        return

    try:
        print(f"Using original {n_slices} slices (no interpolation/smoothing)")
        spacing_raw = gui.meta.get('spacing', [1.0, 1.0, 1.0])
        origin_raw = gui.meta.get('origin', [0.0, 0.0, 0.0])
        ori_shape = gui.meta.get('shape', mask3d.shape)
        slice_thickness = gui.meta.get('slice_thickness', None)

        def convert_to_numpy(data):
            if hasattr(data, 'numpy'):
                return data.numpy()
            if hasattr(data, 'cpu'):
                return data.cpu().numpy()
            return np.array(data)

        spacing_original = convert_to_numpy(spacing_raw).flatten()
        origin = convert_to_numpy(origin_raw).flatten()
        ori_shape = convert_to_numpy(ori_shape).flatten()

        if len(spacing_original) == 1:
            spacing_original = np.array([spacing_original[0]] * 3)
        elif len(spacing_original) == 2:
            spacing_original = np.array([spacing_original[0], spacing_original[1], spacing_original[0]])
        elif len(spacing_original) > 3:
            spacing_original = spacing_original[:3]

        if slice_thickness is not None:
            slice_thickness_val = float(convert_to_numpy(slice_thickness))
            z_spacing_original = spacing_original[2] if len(spacing_original) > 2 else spacing_original[0]
            if abs(z_spacing_original - slice_thickness_val) > 0.01:
                spacing_original[2] = slice_thickness_val

        current_shape = np.array(mask3d.shape)
        if len(ori_shape) >= 3:
            ori_z, ori_y, ori_x = ori_shape[:3]
        else:
            ori_z, ori_y, ori_x = current_shape
        curr_z, curr_y, curr_x = current_shape
        resize_factor_x = ori_x / curr_x if curr_x > 0 else 1.0
        resize_factor_y = ori_y / curr_y if curr_y > 0 else 1.0
        resize_factor_z = ori_z / curr_z if curr_z > 0 else 1.0

        spacing_current = np.array([
            spacing_original[0] * resize_factor_x,
            spacing_original[1] * resize_factor_y,
            spacing_original[2] * resize_factor_z,
        ])
        spacing_current_zyx = spacing_current[::-1]

        unique_ids = np.unique(mask3d)
        unique_ids = unique_ids[unique_ids > 0]
        if len(unique_ids) == 0:
            QMessageBox.warning(gui, 'No Objects', 'No valid objects found in mask!')
            return

        class_colors = {
            1: 'red', 2: 'blue', 3: 'green', 4: 'yellow',
            5: 'purple', 6: 'orange', 7: 'pink', 8: 'cyan', 9: 'brown', 0: 'gray'
        }

        all_meshes = []
        volume_results = {}
        for obj_id in unique_ids:
            obj_mask = (mask3d == obj_id).astype(np.uint8)
            if obj_mask.sum() == 0:
                continue
            obj_mask_float = obj_mask.astype(float)
            try:
                verts, faces, normals, values = marching_cubes(
                    obj_mask_float, level=0.5, spacing=spacing_current_zyx, gradient_direction='descent'
                )
                faces_vtk = np.column_stack([np.full(faces.shape[0], 3), faces]).flatten()
                mesh = pv.PolyData(verts, faces_vtk).clean()
                color = class_colors.get(obj_id, 'gray')
                all_meshes.append({'mesh': mesh, 'color': color, 'obj_id': obj_id})
            except Exception as e:
                print(f"  Failed to create mesh for object {obj_id}: {e}")
                continue

        print("=== Dual Volume Calculation ===")
        for obj_id in unique_ids:
            obj_mask_resized = (mask3d == obj_id).astype(np.uint8)
            voxel_count = int(obj_mask_resized.sum())
            if voxel_count == 0:
                continue
            voxel_volume = float(np.prod(spacing_current))
            volume = voxel_count * voxel_volume
            volume_results[obj_id] = {
                'voxel_count': voxel_count,
                'volume': volume,
                'voxel_size': voxel_volume,
            }

        if not all_meshes:
            QMessageBox.warning(gui, 'Mesh Generation Failed', 'Failed to generate any 3D meshes!')
            return

        z_max, y_max, x_max = mask3d.shape
        x_size = x_max * spacing_current_zyx[2]
        y_size = y_max * spacing_current_zyx[1]
        z_size = z_max * spacing_current_zyx[0]

        pl = pv.Plotter(title=f'Pure 3D Volume - {gui.patient_id}')
        grid_spacing = [max(10, x_size/20), max(10, y_size/20), max(10, z_size/20)]
        grid = pv.ImageData(
            dimensions=(int(x_size/grid_spacing[0])+1, int(y_size/grid_spacing[1])+1, int(z_size/grid_spacing[2])+1),
            spacing=grid_spacing,
            origin=(0, 0, 0),
        )
        pl.add_mesh(grid.outline(), color='lightgray', line_width=1, opacity=0.3)
        for mesh_info in all_meshes:
            pl.add_mesh(mesh_info['mesh'], color=mesh_info['color'], opacity=0.8, smooth_shading=True,
                        label=f"Object {mesh_info['obj_id']}")
        pl.add_axes(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')

        volume_lines = [f"Obj {oid}: {res['volume']:.2f} mm³" for oid, res in volume_results.items()]
        total_volume = sum(res['volume'] for res in volume_results.values())
        volume_text = (
            f"Volume Results:\n\n"
            f"Current Resolution (e.g., 1024×1024):\n{chr(10).join(volume_lines)}\n"
            f"Total: {total_volume:.2f} mm³\n"
        )
        pl.add_text(volume_text, position='upper_right', font_size=9, color='black')
        processing_info = [
            f'Slices: {n_slices} (original, no interpolation)',
            'Smoothing: None (pure binary)',
            'Threshold: 0.5 (exact)',
            f'Spacing verified: {slice_thickness is not None}',
        ]
        pl.add_text('\n'.join(processing_info), position='upper_left', font_size=9, color='darkblue')
        pl.show()
        QMessageBox.information(gui, 'Pure 3D Volume Results', volume_text)

    except ImportError as e:
        QMessageBox.critical(gui, 'Missing Dependencies',
                              f'Required packages missing: {e}\n\nPlease install: pip install scipy scikit-image pyvista')
    except Exception as e:
        print(f"3D reconstruction failed: {e}")
        traceback.print_exc()
        QMessageBox.critical(gui, 'Render Error', f'3D reconstruction failed: {e}')


def render_manual_volume(gui):
    """Render 3D volume for manual GUI. Expects gui with mask_layer, meta, patient_id."""
    mask3d = gui.mask_layer.data.copy()
    if mask3d.shape[0] < 2:
        QMessageBox.warning(gui, 'Insufficient Data',
                           f'Need at least 2 slices for 3D rendering. Current: {mask3d.shape[0]} slices')
        return
    if mask3d.sum() == 0:
        QMessageBox.warning(gui, 'Empty Mask', 'No mask data to render!')
        return

    print("\n=== Pure 3D Volume Rendering (Label-based) ===")
    print(f"mask3d shape: {mask3d.shape} (slices: {mask3d.shape[0]})")
    print(f"mask3d dtype: {mask3d.dtype}")
    print(f"mask3d range: {mask3d.min()} - {mask3d.max()}")
    print(f"non-zero voxels: {(mask3d > 0).sum()}")
    print(f"unique labels: {np.unique(mask3d[mask3d > 0])}")

    try:
        spacing_raw = gui.meta.get('spacing', [1.0, 1.0, 1.0])
        origin_raw = gui.meta.get('origin', [0.0, 0.0, 0.0])
        ori_shape = gui.meta.get('shape', mask3d.shape)

        def tensor_to_numpy(data):
            if hasattr(data, 'cpu'):
                return data.cpu().numpy()
            if hasattr(data, 'numpy'):
                return data.numpy()
            return np.array(data)

        spacing_original = tensor_to_numpy(spacing_raw).flatten()
        origin = tensor_to_numpy(origin_raw).flatten()
        ori_shape = tensor_to_numpy(ori_shape).flatten()

        if len(spacing_original) < 3:
            spacing_original = np.pad(spacing_original, (0, 3-len(spacing_original)), constant_values=1.0)
        elif len(spacing_original) > 3:
            spacing_original = spacing_original[:3]
        if len(origin) < 3:
            origin = np.pad(origin, (0, 3-len(origin)), constant_values=0.0)
        elif len(origin) > 3:
            origin = origin[:3]

        if len(ori_shape) >= 3:
            ori_z, ori_y, ori_x = int(ori_shape[0]), int(ori_shape[1]), int(ori_shape[2])
        else:
            ori_z, ori_y, ori_x = mask3d.shape
        curr_z, curr_y, curr_x = mask3d.shape

        resize_factor_x = ori_x / curr_x if curr_x > 0 else 1.0
        resize_factor_y = ori_y / curr_y if curr_y > 0 else 1.0
        resize_factor_z = ori_z / curr_z if curr_z > 0 else 1.0

        print(f"Resize factors (x,y,z): ({resize_factor_x:.3f}, {resize_factor_y:.3f}, {resize_factor_z:.3f})")

        current_voxel_volume = np.prod(spacing_original) * (resize_factor_x * resize_factor_y * resize_factor_z)
        current_resolution_volume = mask3d.sum() * current_voxel_volume

        print(f"1024x1024 Resolution Volume: {current_resolution_volume:.3f} mm³")

        spacing_adjusted = np.array([
            spacing_original[0] * resize_factor_x,
            spacing_original[1] * resize_factor_y,
            spacing_original[2] * resize_factor_z,
        ])
        spacing_zyx = np.array([spacing_adjusted[2], spacing_adjusted[1], spacing_adjusted[0]])
        origin_zyx = np.array([origin[2], origin[1], origin[0]])

        unique_labels = np.unique(mask3d)
        unique_labels = unique_labels[unique_labels > 0]
        print(f"Found {len(unique_labels)} unique labels: {unique_labels}")

        def get_napari_label_color(label_id):
            try:
                if hasattr(gui.mask_layer, 'get_color'):
                    color_rgba = gui.mask_layer.get_color(label_id)
                    return color_rgba[:3]
                colormap = gui.mask_layer.colormap
                if hasattr(colormap, 'map'):
                    color_rgba = colormap.map(label_id / 255.0)
                    return color_rgba[:3]
            except Exception as e:
                print(f"  Warning: Could not get color for label {label_id}, using default: {e}")
            default_colors = [
                [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.5, 0.0], [0.5, 0.0, 1.0],
            ]
            return default_colors[label_id % len(default_colors)]

        all_meshes = []
        label_volumes_current = {}
        for label_id in unique_labels:
            label_mask = (mask3d == label_id).astype(np.uint8)
            if label_mask.sum() == 0:
                continue
            try:
                verts, faces, normals, values = marching_cubes(
                    label_mask.astype(float), level=0.5, spacing=spacing_zyx, gradient_direction='descent'
                )
                verts = verts + origin_zyx.reshape(1, 3)
                verts_xyz = verts[:, [2, 1, 0]]
                faces_vtk = np.column_stack([np.full(faces.shape[0], 3), faces]).flatten()
                mesh = pv.PolyData(verts_xyz, faces_vtk)
                napari_color = get_napari_label_color(label_id)
                all_meshes.append({'mesh': mesh, 'color': napari_color, 'label_id': label_id})
                label_voxel_count = label_mask.sum()
                label_volumes_current[label_id] = label_voxel_count * current_voxel_volume
            except Exception as e:
                print(f"  Failed to generate mesh for label {label_id}: {e}")
                continue

        if not all_meshes:
            QMessageBox.warning(gui, 'No Meshes', 'Could not generate any 3D meshes!')
            return

        grid_x_size = curr_x * spacing_adjusted[0]
        grid_y_size = curr_y * spacing_adjusted[1]
        grid_z_size = curr_z * spacing_adjusted[2]
        pl = pv.Plotter(title=f'Pure 3D Volume Rendering - {getattr(gui, "patient_id", "Unknown")}')
        grid_bounds = [origin[0], origin[0] + grid_x_size, origin[1], origin[1] + grid_y_size,
                       origin[2], origin[2] + grid_z_size]
        grid_box = pv.Box(bounds=grid_bounds)
        pl.add_mesh(grid_box, style='wireframe', line_width=2, color='black', opacity=0.3, label='Volume Boundary')
        n_lines = min(8, max(curr_x, curr_y, curr_z) // 20)
        if n_lines > 0:
            x_step = max(1, curr_x // n_lines)
            for i in range(0, curr_x, x_step):
                x_pos = origin[0] + i * spacing_adjusted[0]
                line_points = np.array([[x_pos, origin[1], origin[2]], [x_pos, origin[1] + grid_y_size, origin[2] + grid_z_size]])
                line = pv.lines_from_points(line_points)
                pl.add_mesh(line, color='lightblue', line_width=1, opacity=0.2)

        for mesh_info in all_meshes:
            pl.add_mesh(mesh_info['mesh'], color=mesh_info['color'], opacity=0.8, label=f"Label {mesh_info['label_id']}")
        pl.add_axes(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)', line_width=5)
        pl.add_legend(bcolor='white', face='r')
        pl.camera_position = 'iso'

        total_current_volume = sum(label_volumes_current.values())
        grid_info = [
            'Pure 3D Rendering (Label-based, No Interpolation)',
            f'Slices: {curr_z} (minimum 2 required)',
            f'Current Grid: {curr_x} × {curr_y} × {curr_z} voxels',
            f'Physical Size: {grid_x_size:.1f} × {grid_y_size:.1f} × {grid_z_size:.1f} mm³',
            f'Slice Thickness: {spacing_adjusted[2]:.3f} mm',
        ]
        volume_info = [
            f'Label {label_id}: {label_volumes_current[label_id]:.2f} mm³'
            for label_id in sorted(label_volumes_current.keys())
        ]
        pl.add_text('\n'.join(grid_info), position='upper_left', font_size=9, color='black')
        if volume_info:
            pl.add_text('\n'.join(volume_info), position='upper_right', font_size=9, color='darkblue')
        pl.add_text(
            f'1024×1024 Total: {total_current_volume:.2f} mm³',
            position='lower_right', font_size=11, color='darkred'
        )
        pl.show()

        summary_lines = grid_info + [''] + volume_info + [
            '',
            f'Total Volume (1024×1024): {total_current_volume:.2f} mm³',
        ]
        QMessageBox.information(gui, 'Pure 3D Volume Rendering Complete', '\n'.join(summary_lines))

    except ImportError as e:
        QMessageBox.critical(gui, 'Missing Dependencies', f'Required packages missing: {e}\n\nPlease install: pip install scikit-image pyvista')
    except Exception as e:
        print(f"3D reconstruction failed: {e}")
        traceback.print_exc()
        QMessageBox.critical(gui, 'Render Error', f'3D reconstruction failed: {e}')
