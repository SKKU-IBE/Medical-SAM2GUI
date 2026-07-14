import os
import re
from functools import lru_cache
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from PIL import Image
import cv2
from scipy.optimize import linear_sum_assignment


def _strip_nii_ext(name: str):
    name = os.path.basename(name)
    return name[:-7] if name.lower().endswith('.nii.gz') else name[:-4] if name.lower().endswith('.nii') else name


@lru_cache(maxsize=4096)
def _cached_nifti_is_label_map(path, file_size, modified_ns, max_unique_labels):
    """Inspect a stable NIfTI file revision and conservatively detect label maps."""
    del file_size, modified_ns
    try:
        import nibabel as nib

        image = nib.load(path)
        if len(image.shape) != 3:
            return False

        intent_name = str(image.header.get_intent()[0]).lower()
        if intent_name == 'label':
            return True

        data = np.asanyarray(image.dataobj)
        if data.size == 0 or not np.issubdtype(data.dtype, np.number):
            return False
        if not np.all(np.isfinite(data)):
            return False

        minimum = float(np.min(data))
        if minimum < 0.0:
            return False
        if np.issubdtype(data.dtype, np.floating):
            rounded = np.rint(data)
            if not np.allclose(data, rounded, rtol=0.0, atol=1e-6):
                return False

        return int(np.unique(data).size) <= int(max_unique_labels)
    except Exception as exc:
        print(
            f"Warning: Could not inspect NIfTI for label data; keeping it in the "
            f"patient list: {path} ({exc})"
        )
        return False


def nifti_is_label_map(path, max_unique_labels=64):
    """Return True when a NIfTI file has the value characteristics of a label map."""
    try:
        absolute_path = os.path.abspath(os.fspath(path))
        stat = os.stat(absolute_path)
    except (OSError, TypeError, ValueError) as exc:
        print(
            f"Warning: Could not stat NIfTI for label detection; keeping it in the "
            f"patient list: {path} ({exc})"
        )
        return False
    return _cached_nifti_is_label_map(
        absolute_path,
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(max_unique_labels),
    )


def discover_studies(data_root, exclude_dirs=('preprocessed',)):
    """Discover NIfTI files and DICOM series directories under a root path."""
    studies = []

    def add_nifti(path, name):
        if nifti_is_label_map(path):
            print(f"Skipping NIfTI label map from patient list: {path}")
            return
        studies.append((path, name))

    if os.path.isfile(data_root):
        lower = data_root.lower()
        if lower.endswith('.nii') or lower.endswith('.nii.gz'):
            add_nifti(data_root, _strip_nii_ext(data_root))
            return studies
        if lower.endswith('.dcm'):
            parent = os.path.dirname(data_root)
            studies.append((parent, os.path.basename(parent)))
            return studies

    excluded_names = {str(name).casefold() for name in exclude_dirs}
    for root, dirs, files in os.walk(data_root):
        dirs[:] = [
            directory
            for directory in dirs
            if directory.casefold() not in excluded_names
            and not directory.casefold().endswith('_masks')
        ]

        for fname in sorted(files):
            lower = fname.lower()
            if lower.endswith('.nii') or lower.endswith('.nii.gz'):
                fpath = os.path.join(root, fname)
                add_nifti(fpath, _strip_nii_ext(fname))

        if any(f.lower().endswith('.dcm') for f in files):
            studies.append((root, os.path.basename(root)))

    studies.sort(key=lambda x: x[0])
    return studies

def n4_correction(image_sitk):
    # Convert to 32-bit float to avoid pixel type issues
    image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)
    
    # Apply Otsu threshold for mask creation
    maskImage = sitk.OtsuThreshold(image_sitk, 0, 1, 200)
    
    # Set up N4 bias field correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([4]*4)
    
    try:
        output = corrector.Execute(image_sitk, maskImage)
        return output
    except RuntimeError as e:
        print(f"N4 correction failed: {e}")
        print("Returning original image without correction")
        return image_sitk

def intensity_clipping(arr, low=0.5, high=99.5):
    v_min, v_max = np.percentile(arr, [low, high])
    arr = np.clip(arr, v_min, v_max)
    return arr

def intensity_normalization(arr):
    arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
    return arr

def normalize_for_display(arr2d):
    """Normalize image to 0-1 range for display."""
    arr_min, arr_max = arr2d.min(), arr2d.max()
    if arr_max > arr_min:
        return (arr2d - arr_min) / (arr_max - arr_min)
    else:
        return np.zeros_like(arr2d)

def safe_convert_to_uint8(arr2d):
    """Safely convert array to uint8."""
    # 1) Remove outliers using 0.5/99.5 percentiles
    p_low, p_high = np.percentile(arr2d, [0.5, 99.5])
    arr_clipped = np.clip(arr2d, p_low, p_high)
    
    # 2) Normalize to 0-1
    if p_high > p_low:
        arr_norm = (arr_clipped - p_low) / (p_high - p_low)
    else:
        arr_norm = np.zeros_like(arr_clipped)
    
    # 3) Convert to uint8
    return (arr_norm * 255).astype(np.uint8)

def analyze_dicom_properties(folder):
    """Debug helper to inspect DICOM folder properties including orientation."""
    try:
        import pydicom
        import glob
        
        dicom_files = glob.glob(os.path.join(folder, "*.dcm"))
        if not dicom_files:
            dicom_files = [f for f in os.listdir(folder) if not f.startswith('.')]
            if dicom_files:
                dicom_files = [os.path.join(folder, f) for f in dicom_files[:1]]
        
        if dicom_files:
            try:
                ds = pydicom.dcmread(dicom_files[0])
                
                print("=== DICOM Properties ===")
                print(f"Patient Name: {getattr(ds, 'PatientName', 'Unknown')}")
                print(f"Study Description: {getattr(ds, 'StudyDescription', 'Unknown')}")
                print(f"Series Description: {getattr(ds, 'SeriesDescription', 'Unknown')}")
                print(f"Modality: {getattr(ds, 'Modality', 'Unknown')}")
                print(f"Photometric Interpretation: {getattr(ds, 'PhotometricInterpretation', 'Unknown')}")
                print(f"Rescale Slope: {getattr(ds, 'RescaleSlope', 'None')}")
                print(f"Rescale Intercept: {getattr(ds, 'RescaleIntercept', 'None')}")
                print(f"Window Center: {getattr(ds, 'WindowCenter', 'None')}")
                print(f"Window Width: {getattr(ds, 'WindowWidth', 'None')}")
                print(f"Pixel Representation: {getattr(ds, 'PixelRepresentation', 'None')}")
                print(f"Bits Stored: {getattr(ds, 'BitsStored', 'None')}")
                
                # Orientation details
                print(f"\n=== Geometric Information ===")
                print(f"Image Orientation Patient: {getattr(ds, 'ImageOrientationPatient', 'None')}")
                print(f"Image Position Patient: {getattr(ds, 'ImagePositionPatient', 'None')}")
                print(f"Slice Thickness: {getattr(ds, 'SliceThickness', 'None')}")
                print(f"Pixel Spacing: {getattr(ds, 'PixelSpacing', 'None')}")
                print(f"Rows x Columns: {getattr(ds, 'Rows', 'None')} x {getattr(ds, 'Columns', 'None')}")
                
                # Interpret Image Orientation Patient
                if hasattr(ds, 'ImageOrientationPatient'):
                    iop = np.array(ds.ImageOrientationPatient, dtype=float)
                    if len(iop) == 6:
                        row_cosines = iop[:3]
                        col_cosines = iop[3:]
                        
                        print(f"\n=== Orientation Analysis ===")
                        print(f"Row direction cosines: {row_cosines}")
                        print(f"Column direction cosines: {col_cosines}")
                        
                        # Compute slice direction via cross product
                        slice_cosines = np.cross(row_cosines, col_cosines)
                        print(f"Slice direction cosines: {slice_cosines}")
                        
                        # Determine primary axes
                        def get_primary_axis(cosines):
                            abs_cosines = np.abs(cosines)
                            max_idx = np.argmax(abs_cosines)
                            axes = ['L/R (Left/Right)', 'A/P (Anterior/Posterior)', 'I/S (Inferior/Superior)']
                            direction = 'positive' if cosines[max_idx] > 0 else 'negative'
                            return f"{axes[max_idx]} ({direction})"
                        
                        print(f"Row primary axis: {get_primary_axis(row_cosines)}")
                        print(f"Column primary axis: {get_primary_axis(col_cosines)}")
                        print(f"Slice primary axis: {get_primary_axis(slice_cosines)}")
                
                if hasattr(ds, 'pixel_array'):
                    pixel_array = ds.pixel_array
                    print(f"\nRaw pixel array - min: {pixel_array.min()}, max: {pixel_array.max()}, dtype: {pixel_array.dtype}")
                
                print("=" * 40)
                
            except Exception as e:
                print(f"Failed to analyze DICOM: {e}")
    except ImportError:
        print("pydicom not available for detailed DICOM analysis")

def save_nifti(arr, ref_sitk, save_path, preserve_dtype=True, verbose=False):
    """Save array to NIfTI while preserving geometry from a reference image.

    Args:
        arr: array to save (numpy array)
        ref_sitk: reference ITK image providing geometry
        save_path: output path
        preserve_dtype: keep original data type if True
        verbose: print debug details if True
    """
    if verbose:
        print(f"\n=== NIfTI Save Debug ===")
        print(f"Array shape: {arr.shape}, dtype: {arr.dtype}")
        print(f"Array range: {arr.min()} ~ {arr.max()}")
    
    # Handle data type
    if preserve_dtype:
        # Inspect reference image dtype
        ref_array = sitk.GetArrayFromImage(ref_sitk)
        if verbose:
            print(f"Reference image dtype: {ref_array.dtype}")
        
        # Masks are typically integer types
        if arr.dtype == bool or (arr.max() <= 255 and arr.min() >= 0):
            save_arr = arr.astype(np.uint8)
        else:
            save_arr = arr.astype(ref_array.dtype)
    else:
        save_arr = arr.astype(np.float32)
    
    # Build ITK image
    img = sitk.GetImageFromArray(save_arr)
    
    # Copy geometry exactly
    img.SetSpacing(ref_sitk.GetSpacing())
    img.SetDirection(ref_sitk.GetDirection())
    img.SetOrigin(ref_sitk.GetOrigin())
    
    if verbose:
        print(f"Reference image:")
        print(f"  Size: {ref_sitk.GetSize()}")
        print(f"  Spacing: {ref_sitk.GetSpacing()}")
        print(f"  Origin: {ref_sitk.GetOrigin()}")
        print(f"  Direction: {ref_sitk.GetDirection()}")
        
        print(f"Image to save:")
        print(f"  Size: {img.GetSize()}")
        print(f"  Spacing: {img.GetSpacing()}")
        print(f"  Origin: {img.GetOrigin()}")
        print(f"  Direction: {img.GetDirection()}")
    
    # Save
    sitk.WriteImage(img, save_path)
    
    if verbose:
        # Validate after saving
        saved_img = sitk.ReadImage(save_path)
        print(f"Post-save validation:")
        print(f"  Geometry match: {check_geometry_match(ref_sitk, saved_img)}")
        print(f"  Saved path: {save_path}")
        print("=" * 40)

def check_geometry_match(img1, img2, tolerance=1e-6):
    """Check if two ITK images share the same geometry."""
    try:
        size_match = img1.GetSize() == img2.GetSize()
        spacing_match = np.allclose(img1.GetSpacing(), img2.GetSpacing(), rtol=tolerance)
        origin_match = np.allclose(img1.GetOrigin(), img2.GetOrigin(), rtol=tolerance)
        direction_match = np.allclose(img1.GetDirection(), img2.GetDirection(), rtol=tolerance)
        
        return size_match and spacing_match and origin_match and direction_match
    except:
        return False

def load_nii_image(image_path):
    img_sitk = sitk.ReadImage(image_path)
    arr = sitk.GetArrayFromImage(img_sitk)
    return arr, img_sitk


DICOM_SPACING_SOURCE_META_KEY = "dicom_spacing_source"
DICOM_SELECTED_SERIES_META_KEY = "dicom_selected_series_uid"
DICOM_SLICE_SPACING_META_KEY = "dicom_slice_spacing"


def _dicom_float_values(value):
    """Convert common DICOM scalar/multivalue fields to finite floats."""
    if value is None:
        return []
    if isinstance(value, bytes):
        value = value.decode(errors="ignore")
    if isinstance(value, str):
        parts = value.split("\\")
    else:
        try:
            parts = list(value)
        except TypeError:
            parts = [value]

    values = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if not text:
            continue
        try:
            number = float(text)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            values.append(number)
    return values


def _median_positive_dicom_value(values):
    numbers = []
    for value in values:
        for number in _dicom_float_values(value):
            if number > 0:
                numbers.append(number)
    if not numbers:
        return None
    return float(np.median(numbers))


def _infer_spacing_from_folder_name(folder):
    folder_name = os.path.basename(os.path.normpath(folder))
    match = re.search(r"(?i)(?:^|[_\-\s])(\d+(?:\.\d+)?)\s*mm$", folder_name)
    if not match:
        return None
    try:
        spacing = float(match.group(1))
    except ValueError:
        return None
    return spacing if spacing > 0 else None


def _dicom_instance_number(ds):
    values = _dicom_float_values(getattr(ds, "InstanceNumber", None))
    return int(values[0]) if values else None


def _sort_dicom_items_by_instance_or_name(items):
    def sort_key(item):
        path, ds = item
        instance_number = _dicom_instance_number(ds)
        if instance_number is not None:
            return (0, instance_number, os.path.basename(path))
        return (1, os.path.basename(path))

    return sorted(items, key=sort_key)


def _dicom_orientation_vectors(items):
    for _, ds in items:
        orientation = _dicom_float_values(getattr(ds, "ImageOrientationPatient", None))
        if len(orientation) < 6:
            continue
        row = np.asarray(orientation[:3], dtype=float)
        col = np.asarray(orientation[3:6], dtype=float)
        row_norm = np.linalg.norm(row)
        col_norm = np.linalg.norm(col)
        if row_norm <= 0 or col_norm <= 0:
            continue
        row = row / row_norm
        col = col / col_norm
        normal = np.cross(row, col)
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= 0:
            continue
        normal = normal / normal_norm
        return row, col, normal
    return None


def _direction_from_orientation(row, col, normal):
    direction_matrix = np.column_stack((row, col, normal))
    return tuple(float(v) for v in direction_matrix.flatten())


def _origin_from_dicom_item(item):
    _, ds = item
    position = _dicom_float_values(getattr(ds, "ImagePositionPatient", None))
    if len(position) >= 3:
        return tuple(float(v) for v in position[:3])
    return (0.0, 0.0, 0.0)


def _infer_dicom_slice_order_and_spacing(items, folder, duplicate_tolerance=1e-4):
    """
    Return ordered DICOM items, z spacing, spacing source, origin, and direction.

    Spacing priority:
    1. ImagePositionPatient projected onto the slice normal
    2. positive SpacingBetweenSlices values
    3. positive SliceThickness values
    4. folder-name suffix like *_3mm
    """
    orientation = _dicom_orientation_vectors(items)
    if orientation is not None:
        row, col, normal = orientation
        direction = _direction_from_orientation(row, col, normal)
        position_records = []
        for item in items:
            _, ds = item
            position = _dicom_float_values(getattr(ds, "ImagePositionPatient", None))
            if len(position) < 3:
                continue
            position_arr = np.asarray(position[:3], dtype=float)
            projection = float(np.dot(position_arr, normal))
            position_records.append((projection, item))

        if len(position_records) == len(items):
            position_records.sort(key=lambda record: record[0])
            projections = np.asarray([record[0] for record in position_records], dtype=float)
            diffs = np.diff(projections)
            duplicate_diffs = [d for d in diffs if abs(float(d)) <= duplicate_tolerance]
            if duplicate_diffs:
                raise RuntimeError(
                    "Duplicate DICOM slice positions detected after applying "
                    "ImagePositionPatient/ImageOrientationPatient. Cannot infer a "
                    "unique 3D slice order."
                )

            nonzero_diffs = [abs(float(d)) for d in diffs if abs(float(d)) > duplicate_tolerance]
            if nonzero_diffs:
                spacing = float(np.median(nonzero_diffs))
                ordered_items = [item for _, item in position_records]
                return ordered_items, spacing, "dicom_position", _origin_from_dicom_item(ordered_items[0]), direction
    else:
        direction = tuple(np.eye(3).flatten())

    ordered_items = _sort_dicom_items_by_instance_or_name(items)
    spacing = _median_positive_dicom_value(
        [getattr(ds, "SpacingBetweenSlices", None) for _, ds in items]
    )
    if spacing is not None:
        return ordered_items, spacing, "spacing_between_slices", _origin_from_dicom_item(ordered_items[0]), direction

    spacing = _median_positive_dicom_value(
        [getattr(ds, "SliceThickness", None) for _, ds in items]
    )
    if spacing is not None:
        return ordered_items, spacing, "slice_thickness", _origin_from_dicom_item(ordered_items[0]), direction

    spacing = _infer_spacing_from_folder_name(folder)
    if spacing is not None:
        return ordered_items, spacing, "folder_name", _origin_from_dicom_item(ordered_items[0]), direction

    raise RuntimeError(
        "Could not infer DICOM slice spacing from slice positions, "
        "SpacingBetweenSlices, SliceThickness, or a folder-name suffix like *_3mm."
    )


def _infer_dicom_pixel_spacing(items):
    row_spacings = []
    col_spacings = []
    for _, ds in items:
        pixel_spacing = _dicom_float_values(getattr(ds, "PixelSpacing", None))
        if len(pixel_spacing) >= 2 and pixel_spacing[0] > 0 and pixel_spacing[1] > 0:
            row_spacings.append(pixel_spacing[0])
            col_spacings.append(pixel_spacing[1])
    if not row_spacings or not col_spacings:
        raise RuntimeError("Could not infer in-plane DICOM PixelSpacing.")
    return float(np.median(col_spacings)), float(np.median(row_spacings))


def _collect_dicom_candidate_files(folder):
    search_dir = os.path.dirname(folder) if os.path.isfile(folder) else folder
    if not os.path.isdir(search_dir):
        raise RuntimeError(f"DICOM path is not a folder: {folder}")

    dcm_files = [
        os.path.join(search_dir, f)
        for f in os.listdir(search_dir)
        if f.lower().endswith(".dcm") and os.path.isfile(os.path.join(search_dir, f))
    ]
    if dcm_files:
        return sorted(dcm_files)

    return sorted(
        os.path.join(search_dir, f)
        for f in os.listdir(search_dir)
        if not f.startswith(".") and os.path.isfile(os.path.join(search_dir, f))
    )


def _dicom_single_float(ds, name, default):
    values = _dicom_float_values(getattr(ds, name, None))
    return float(values[0]) if values else default


def _stack_dicom_pixel_arrays(ordered_items):
    arrays = []
    first_shape = None
    for path, ds in ordered_items:
        try:
            arr = np.asarray(ds.pixel_array)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode DICOM pixel data from {path}: {exc}") from exc

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise RuntimeError(
                f"Unsupported DICOM pixel array shape {arr.shape} in {path}; expected one 2D slice per file."
            )
        if first_shape is None:
            first_shape = arr.shape
        elif arr.shape != first_shape:
            raise RuntimeError(
                f"Inconsistent DICOM slice shape in {path}: {arr.shape} != {first_shape}"
            )

        slope = _dicom_single_float(ds, "RescaleSlope", 1.0)
        intercept = _dicom_single_float(ds, "RescaleIntercept", 0.0)
        if slope != 1.0 or intercept != 0.0:
            arr = arr.astype(np.float32) * slope + intercept
        arrays.append(arr)

    if not arrays:
        raise RuntimeError("No DICOM pixel arrays could be decoded.")
    return np.stack(arrays, axis=0)


def _load_dicom_series_with_pydicom_spacing_fallback(folder, debug_geometry=False):
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("pydicom is required for DICOM spacing fallback.") from exc

    candidate_files = _collect_dicom_candidate_files(folder)
    if not candidate_files:
        raise RuntimeError(f"No DICOM files found in {folder}")

    series_groups = {}
    read_errors = []
    for path in candidate_files:
        try:
            ds = pydicom.dcmread(path, force=True)
        except Exception as exc:
            read_errors.append((path, exc))
            continue
        if not hasattr(ds, "PixelData"):
            continue
        series_uid = str(getattr(ds, "SeriesInstanceUID", "NO_SERIES_UID"))
        try:
            rows = int(getattr(ds, "Rows", 0) or 0)
            cols = int(getattr(ds, "Columns", 0) or 0)
        except (TypeError, ValueError):
            rows, cols = 0, 0
        group_key = (series_uid, rows, cols)
        series_groups.setdefault(group_key, []).append((path, ds))

    if not series_groups:
        detail = f" Read errors: {len(read_errors)}." if read_errors else ""
        raise RuntimeError(f"No readable DICOM pixel series found in {folder}.{detail}")

    selected_key, selected_items = max(
        series_groups.items(),
        key=lambda kv: (len(kv[1]), int(kv[0][1] or 0) * int(kv[0][2] or 0)),
    )
    selected_uid, selected_rows, selected_cols = selected_key
    if len(series_groups) > 1:
        print(
            f"Warning: found {len(series_groups)} DICOM series/shape groups in {folder}; "
            f"using series {selected_uid} shape {selected_rows}x{selected_cols} "
            f"with {len(selected_items)} files."
        )
    if read_errors and debug_geometry:
        print(f"Skipped {len(read_errors)} unreadable DICOM files during pydicom fallback.")

    ordered_items, z_spacing, spacing_source, origin, direction = _infer_dicom_slice_order_and_spacing(
        selected_items, folder
    )
    x_spacing, y_spacing = _infer_dicom_pixel_spacing(ordered_items)
    arr = _stack_dicom_pixel_arrays(ordered_items)

    img_sitk = sitk.GetImageFromArray(arr)
    img_sitk.SetSpacing((x_spacing, y_spacing, z_spacing))
    img_sitk.SetOrigin(origin)
    img_sitk.SetDirection(direction)
    img_sitk.SetMetaData(DICOM_SPACING_SOURCE_META_KEY, spacing_source)
    img_sitk.SetMetaData(DICOM_SELECTED_SERIES_META_KEY, selected_uid)
    img_sitk.SetMetaData(DICOM_SLICE_SPACING_META_KEY, f"{z_spacing:.12g}")

    if spacing_source == "folder_name":
        print(
            f"Warning: using DICOM slice spacing {z_spacing:g} mm inferred from "
            f"folder name '{os.path.basename(os.path.normpath(folder))}'."
        )

    if debug_geometry:
        print(f"\n=== DICOM pydicom Fallback Loading Results ===")
        print(f"Folder: {folder}")
        print(f"Series UID: {selected_uid}")
        print(f"DICOM files: {len(ordered_items)}")
        print(f"Image size: {img_sitk.GetSize()}")
        print(f"Array shape: {arr.shape}")
        print(f"Spacing: {img_sitk.GetSpacing()} (source: {spacing_source})")
        print(f"Origin: {img_sitk.GetOrigin()}")
        print(f"Direction: {img_sitk.GetDirection()}")
        print(f"Data range: {arr.min():.3f} ~ {arr.max():.3f}")
        print(f"Data dtype: {arr.dtype}")
        print("=" * 50)

    return arr, img_sitk


def load_dicom_series(folder, debug_geometry=False):
    """
    Load a DICOM series while preserving geometry.

    Args:
        folder: folder containing DICOM files
        debug_geometry: whether to print geometry debug info

    Returns:
        arr: numpy array
        img_sitk: SimpleITK image (with geometry)
    """
    reader = sitk.ImageSeriesReader()
    
    try:
        dicom_names = reader.GetGDCMSeriesFileNames(folder)
        if not dicom_names:
            raise RuntimeError(f"No DICOM files found in {folder}")
        
        reader.SetFileNames(dicom_names)
        
        # Enable metadata loading
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        
        img_sitk = reader.Execute()
        arr = sitk.GetArrayFromImage(img_sitk)
        
        if debug_geometry:
            print(f"\n=== DICOM Series Loading Results ===")
            print(f"Folder: {folder}")
            print(f"DICOM files: {len(dicom_names)}")
            print(f"Image size: {img_sitk.GetSize()}")
            print(f"Array shape: {arr.shape}")
            print(f"Spacing: {img_sitk.GetSpacing()}")
            print(f"Origin: {img_sitk.GetOrigin()}")
            print(f"Direction: {img_sitk.GetDirection()}")
            
            # Show direction matrix as 3x3
            direction = np.array(img_sitk.GetDirection())
            if len(direction) == 9:
                direction_matrix = direction.reshape(3, 3)
                print("Direction Matrix:")
                for i, row in enumerate(direction_matrix):
                    axis_names = ['X (L->R)', 'Y (P->A)', 'Z (I->S)']
                    print(f"  {axis_names[i]}: [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")
            
            # Data range info
            print(f"Data range: {arr.min():.3f} ~ {arr.max():.3f}")
            print(f"Data dtype: {arr.dtype}")
            print("=" * 50)
        
        return arr, img_sitk
        
    except Exception as e:
        print(f"Failed to load DICOM series with SimpleITK: {e}")
        print("Retrying with pydicom geometry/spacing fallback...")
        try:
            return _load_dicom_series_with_pydicom_spacing_fallback(
                folder, debug_geometry=debug_geometry
            )
        except Exception as fallback_error:
            raise RuntimeError(
                "Failed to load DICOM series with both SimpleITK and pydicom "
                f"spacing fallback. SimpleITK error: {e}. "
                f"Fallback error: {fallback_error}"
            ) from fallback_error

def generate_prompt_from_mask(mask_slice, prompt_type="point"):
    coords = np.argwhere(mask_slice > 0)
    if coords.shape[0] == 0:
        return None, None
    if prompt_type == "point":
        idx = np.random.choice(len(coords))
        pt = coords[idx]
        return 1, (int(pt[1]), int(pt[0]))
    else:
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        return [x1, y1, x2, y2]


def build_box_and_points_from_mask(comp_mask):
    """Create bbox + positive/negative points for SAM-style prompts from a mask."""
    yx = np.argwhere(comp_mask)
    if len(yx) == 0:
        return None, None, None
    y1, x1 = yx.min(axis=0)
    y2, x2 = yx.max(axis=0)
    bbox = [int(x1), int(y1), int(x2), int(y2)]

    cy, cx = center_of_mass(comp_mask)
    pos_pt = [int(round(cx)), int(round(cy)), 1]

    # Try to place a negative point inside the bbox but outside the mask
    candidate_pts = [
        (x1, y1), (x2, y1), (x2, y2), (x1, y2),
        (x1, int((y1 + y2) / 2)), (x2, int((y1 + y2) / 2)),
        (int((x1 + x2) / 2), y1), (int((x1 + x2) / 2), y2),
    ]
    neg_pt = None
    for x, y in candidate_pts:
        if 0 <= y < comp_mask.shape[0] and 0 <= x < comp_mask.shape[1]:
            if not comp_mask[y, x]:
                neg_pt = [int(x), int(y), 0]
                break
    if neg_pt is None:
        zeros = np.argwhere(comp_mask == 0)
        if len(zeros) > 0:
            y, x = zeros[len(zeros) // 2]
            neg_pt = [int(x), int(y), 0]
    return bbox, pos_pt, neg_pt

import numpy as np
from scipy.ndimage import label, center_of_mass

def get_connected_components(mask2d):
    # mask2d: 2D array with multiple classes possible (ex: [0, 0, 1, 2, 2])
    # returns: list of (component_mask, class_id)
    out = []
    class_ids = np.unique(mask2d/255)
    class_ids = class_ids[class_ids != 0]
    for cls in class_ids:
        binary = (mask2d == cls).astype(np.uint8)
        labeled, n_obj = label(binary)
        for comp_id in range(1, n_obj+1):
            comp_mask = (labeled == comp_id)
            out.append((comp_mask, cls))
    return out

def optimal_object_matching(prev_objects, current_objects, min_overlap_threshold=5):
    """
    Match objects across frames using the Hungarian algorithm.

    Args:
        prev_objects: {label: mask} from the previous frame
        current_objects: list of masks in the current frame
        min_overlap_threshold: minimum overlap pixels

    Returns:
        assignments: list of (prev_label, curr_idx)
        unmatched_current: list of current indices with no match
    """
    if not prev_objects or not current_objects:
        return [], list(range(len(current_objects)))
    
    prev_labels = list(prev_objects.keys())
    n_prev = len(prev_labels)
    n_curr = len(current_objects)
    
    # Build cost matrix (larger overlap -> lower cost)
    cost_matrix = np.full((n_prev, n_curr), 1000.0)  # high initial cost
    
    for i, prev_label in enumerate(prev_labels):
        prev_mask = prev_objects[prev_label]
        for j, curr_mask in enumerate(current_objects):
            overlap = np.logical_and(prev_mask, curr_mask).sum()
            if overlap >= min_overlap_threshold:
                # IoU-based cost (more overlap -> lower cost)
                union = np.logical_or(prev_mask, curr_mask).sum()
                iou = overlap / union if union > 0 else 0
                cost_matrix[i, j] = 1.0 - iou  # Higher IoU -> lower cost
    
    # Apply Hungarian algorithm
    try:
        prev_indices, curr_indices = linear_sum_assignment(cost_matrix)
    except:
        # On failure, return empty assignment
        return [], list(range(len(current_objects)))
    
    # Keep only valid matches
    valid_assignments = []
    matched_current = set()
    
    for p_idx, c_idx in zip(prev_indices, curr_indices):
        if cost_matrix[p_idx, c_idx] < 0.9:  # IoU > 0.1 only
            prev_label = prev_labels[p_idx]
            valid_assignments.append((prev_label, c_idx))
            matched_current.add(c_idx)
    
    # Unmatched current objects
    unmatched_current = [i for i in range(n_curr) if i not in matched_current]
    
    return valid_assignments, unmatched_current

class SNU3DMRI_MedSAM2Dataset(Dataset):
    """
    SNU 3D MRI Dataset (multi-patient, NIfTI/DICOM) with preprocessing and per-slice MedSAM2 inputs, includes improved object tracking.
    """
    def __init__(self, 
                 data_root,  # root containing patient folders or nii/dcm files
                 preprocess=True,
                 method='det', # 'det' (was 'cls-det'), 'seg'
                 mode='auto',      # 'auto', 'manual'
                 save_preproc_dir=None,
                 img_size=1024,
                 per_patient_settings=None):  # per-patient settings
        super().__init__()
        self.data_root = data_root
        self.preprocess = preprocess  # default preprocessing
        self.method = method
        self.mode = mode
        self.img_size = img_size
        self.save_preproc_dir = save_preproc_dir if save_preproc_dir else os.path.join(data_root, "preprocessed")
        self.per_patient_settings = per_patient_settings or {}  # per-patient settings

        # Collect patient paths and names
        self.patient_paths = []
        self.patient_names = []

        for path, name in discover_studies(data_root):
            self.patient_paths.append(path)
            self.patient_names.append(name)
    
    def update_patient_settings(self, patient_name, settings):
        """Update settings for a specific patient."""
        self.per_patient_settings[patient_name] = settings
    
    def get_patient_preprocess_setting(self, patient_name):
        """Get preprocess flag for a specific patient."""
        if patient_name in self.per_patient_settings:
            return self.per_patient_settings[patient_name].get('preprocess', self.preprocess)
        return self.preprocess
        
    def __len__(self):
        return len(self.patient_paths)

    def __getitem__(self, idx):
        path = self.patient_paths[idx]
        patient_name = self.patient_names[idx]
        save_name = f"{patient_name}_pre_image.nii.gz"
        save_path = os.path.join(self.save_preproc_dir, save_name)
        # print(path)
        # 1. Load data
        if path.endswith('.nii') or path.endswith('.nii.gz'):
            arr, img_sitk = load_nii_image(path)
            if self.method == 'test':
                if path.endswith('.nii'):
                    mask_path = path.replace('_image.nii', '_mask.nii')
                    mask_arr, _ = load_nii_image(mask_path)
                elif path.endswith('.nii.gz'):
                    mask_path = path.replace('_image.nii.gz', '_mask.nii.gz')
                    mask_arr, _ = load_nii_image(mask_path)
                # print(f'mask intessity: {np.unique(mask_arr)}') [0 1]
        elif path.endswith('.dcm') or os.path.isdir(path):
            # Optional DICOM property analysis
            if os.path.isdir(path):
                print(f"Analyzing DICOM folder: {path}")
                analyze_dicom_properties(path)
            
            # Load with geometry debug
            arr, img_sitk = load_dicom_series(path, debug_geometry=True)
            print(f"DICOM loaded - range: {arr.min():.3f} ~ {arr.max():.3f}")
        else:
            raise RuntimeError("Unsupported file/folder type.")

        # 2. Preprocess/save based on per-patient setting
        patient_preprocess = self.get_patient_preprocess_setting(patient_name)
        
        if patient_preprocess:
            os.makedirs(self.save_preproc_dir, exist_ok=True)
            
            # Use existing preprocessed file if present
            if os.path.exists(save_path):
                print(f"Loading existing preprocessed data for {patient_name}...")
                arr_3d, _ = load_nii_image(save_path)
            else:
                print(f"Preprocessing {patient_name}...")
                print(f"Source data stats:")
                print(f"  Range: {arr.min():.3f} ~ {arr.max():.3f}")
                print(f"  Mean: {arr.mean():.3f}, Std: {arr.std():.3f}")
                
                arr_proc = n4_correction(img_sitk)
                arr_proc = sitk.GetArrayFromImage(arr_proc)
                arr_proc = intensity_clipping(arr_proc)
                arr_proc = intensity_normalization(arr_proc)
                
                print(f"Post-preprocess stats:")
                print(f"  Range: {arr_proc.min():.3f} ~ {arr_proc.max():.3f}")
                print(f"  Mean: {arr_proc.mean():.3f}, Std: {arr_proc.std():.3f}")
                
                # Save with detailed geometry
                save_nifti(arr_proc, img_sitk, save_path, preserve_dtype=False, verbose=True)
                arr_3d = arr_proc
                print(f"Preprocessing completed for {patient_name}")
        else:
            print(f"Skipping preprocessing for {patient_name} (user setting)")
            arr_3d = arr

        # 3. Metadata
        shape = arr_3d.shape
        spacing = img_sitk.GetSpacing()
        origin = img_sitk.GetOrigin()
        direction = img_sitk.GetDirection()
        slice_thickness = spacing[2] if len(spacing) > 2 else None
        meta = {
            "shape": shape,
            "spacing": spacing,
            "origin": origin,
            "direction": direction,
            "slice_thickness": slice_thickness,
            "patient": patient_name,
            "source_path": os.path.abspath(path),
        }
        for dicom_meta_key in (
            DICOM_SPACING_SOURCE_META_KEY,
            DICOM_SELECTED_SERIES_META_KEY,
            DICOM_SLICE_SPACING_META_KEY,
        ):
            try:
                if img_sitk.HasMetaDataKey(dicom_meta_key):
                    meta[dicom_meta_key] = img_sitk.GetMetaData(dicom_meta_key)
            except Exception:
                pass

        # Precompute raw volume for nnUNet (no resizing)
        images_nnunet = torch.from_numpy(arr_3d.astype(np.float32))

        # 4. Branch by mode
        if self.mode == "manual":
            images = []
            
            # Debug info
            print("=== Debug Info ===")
            print(f"arr_3d range: {arr_3d.min():.3f} ~ {arr_3d.max():.3f}")
            print(f"arr_3d mean: {arr_3d.mean():.3f}, std: {arr_3d.std():.3f}")
            
            for s in range(arr_3d.shape[0]):  # for each frame/slice
                arr2d = arr_3d[s]
                
                # Safe uint8 conversion
                arr_uint8 = safe_convert_to_uint8(arr2d)
                img = Image.fromarray(arr_uint8).resize((self.img_size, self.img_size))
                img_tensor = torch.from_numpy(np.array(img)).unsqueeze(0).repeat(3, 1, 1)
                images.append(img_tensor)
                
            print(f"Images converted: {len(images)} slices")
            print("==================")
            
            return {
                "image_3d": torch.stack(images),
                "meta": meta
            }
        elif self.mode == "auto":
            images = []
            masks = []
            cls_labels = {}  # {frame_id: {object_id: class_id}}
            bboxes = {}      # {frame_id: {object_id: [x1, y1, x2, y2]}}
            seg_masks = {}
            prompts = {}     # {frame_id: {object_id: {"bboxes": [x1,y1,x2,y2], "points": [x, y]}}}
            object_id_tracker = {}  # {class_id: next_obj_id}
                

            if self.method in ["cls-det", "det"]:
                for s in range(arr_3d.shape[0]):  # for each frame/slice
                    arr2d = arr_3d[s]
                    
                    # Safe uint8 conversion
                    arr_uint8 = safe_convert_to_uint8(arr2d)
                    img = Image.fromarray(arr_uint8).resize((self.img_size, self.img_size))
                    img_tensor = torch.from_numpy(np.array(img)).unsqueeze(0).repeat(3, 1, 1)
                    images.append(img_tensor)
                    cls_labels[s] = {}
                    bboxes[s] = {}
                    prompts[s] = {}
                    seg_masks[s] = {}
                
                    # Example placeholder mask_pred (replace with model output)
                    mask_pred = np.zeros_like(arr2d)  # dummy

                    # Assume multiple detections
                    components = get_connected_components(mask_pred)
                    for comp_mask, cls in components:
                        # Manage object ids
                        if cls not in object_id_tracker:
                            object_id_tracker[cls] = cls * 10
                        object_id = object_id_tracker[cls]
                        object_id_tracker[cls] += 1

                        # Bounding box
                        yx = np.argwhere(comp_mask)
                        y1, x1 = yx.min(axis=0)
                        y2, x2 = yx.max(axis=0)
                        bbox = [int(x1), int(y1), int(x2), int(y2)]

                        cls_labels[s][object_id] = int(cls)
                        bboxes[s][object_id] = bbox
            elif self.method == "seg":
                for s in range(arr_3d.shape[0]):  # for each frame/slice
                    arr2d = arr_3d[s]
                    
                    # Safe uint8 conversion
                    arr_uint8 = safe_convert_to_uint8(arr2d)
                    img = Image.fromarray(arr_uint8).resize((self.img_size, self.img_size))
                    img_tensor = torch.from_numpy(np.array(img)).unsqueeze(0).repeat(3, 1, 1)
                    images.append(img_tensor)
                    cls_labels[s] = {}
                    bboxes[s] = {}
                    prompts[s] = {}
                    seg_masks[s] = {}
                
                    # Placeholder for segmentation mask prediction (replace with model output)
                    mask_pred = np.zeros_like(arr2d)

                    # Handle multiple detected objects
                    components = get_connected_components(mask_pred)
                    for comp_mask, cls in components:
                        if cls not in object_id_tracker:
                            object_id_tracker[cls] = cls * 10
                        object_id = object_id_tracker[cls]
                        object_id_tracker[cls] += 1

                        bbox, pos_pt, neg_pt = build_box_and_points_from_mask(comp_mask)
                        seg_masks[s][object_id] = torch.from_numpy(comp_mask.astype(np.uint8)[None, ...])
                        points = []
                        if pos_pt:
                            points.append(pos_pt)
                        if neg_pt:
                            points.append(neg_pt)
                        prompts[s][object_id] = {"bboxes": bbox, "points": points}
            
            elif self.method == "test":
                # Improved dynamic object tracking system
                prev_objects = {}       # {dynamic_label: mask} from previous frame
                next_new_label = {}     # {original_cls: next_available_label}
                
                print("=== Improved Object Tracking Started ===")
                
                for s in range(arr_3d.shape[0]):  # for each frame/slice
                    arr2d = arr_3d[s]
                    
                    # Safe uint8 conversion
                    arr_uint8 = safe_convert_to_uint8(arr2d)
                    img = Image.fromarray(arr_uint8).resize((self.img_size, self.img_size))
                    img_tensor = torch.from_numpy(np.array(img)).unsqueeze(0).repeat(3, 1, 1)
                    images.append(img_tensor)
                    cls_labels[s] = {}
                    bboxes[s] = {}
                    prompts[s] = {}
                    seg_masks[s] = {}
                    
                    mask2d = mask_arr[s]
                    mask = Image.fromarray((mask2d * 255).astype(np.uint8)).resize((self.img_size, self.img_size), resample=Image.NEAREST)
                    mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).repeat(3, 1, 1)
                    masks.append(mask_tensor)
                    
                    # Extract classes for current frame
                    classes = np.unique(np.array(mask) / 255)
                    classes = classes[classes != 0]
                    
                    current_objects = {}   # {dynamic_label: mask} current frame
                    
                    for cls in classes:
                        # Init next_new_label on first frame
                        if int(cls) not in next_new_label:
                            next_new_label[int(cls)] = int(cls) * 10
                        
                        # Binarize original class
                        binary = (np.array(mask) / 255 == cls).astype(np.uint8)
                        
                        # Connected components
                        num_comps, labels_cc = cv2.connectedComponents(binary, connectivity=8)
                        
                        if num_comps <= 1:  # only background
                            continue
                        
                        # Collect all components in current frame
                        current_components = []
                        for cid in range(1, num_comps):
                            comp_mask = (labels_cc == cid)
                            if comp_mask.sum() > 0:
                                current_components.append(comp_mask)
                        
                        if not current_components:
                            continue
                        
                        # First frame handling
                        if s == 0:
                            # Sort by size; largest keeps original class id
                            comp_sizes = [(i, comp.sum()) for i, comp in enumerate(current_components)]
                            comp_sizes.sort(key=lambda x: x[1], reverse=True)
                            
                            for rank, (comp_idx, size) in enumerate(comp_sizes):
                                if rank == 0:  # largest object
                                    dynamic_label = int(cls)
                                else:  # remaining objects
                                    dynamic_label = next_new_label[int(cls)]
                                    next_new_label[int(cls)] += 1
                                
                                current_objects[dynamic_label] = current_components[comp_idx]
                        
                        else:
                            # Match against previous frame (same class lineage)
                            relevant_prev = {
                                label: mask for label, mask in prev_objects.items()
                                if label == cls or label // 10 == cls
                            }
                            
                            # Compute optimal matching
                            assignments, unmatched_current = optimal_object_matching(
                                relevant_prev, current_components, min_overlap_threshold=5
                            )
                            
                            # Handle matched objects
                            for prev_label, curr_idx in assignments:
                                current_objects[prev_label] = current_components[curr_idx]
                                print(f"  Slice {s}: Matched object {prev_label} with component {curr_idx}")
                            
                            # Assign new labels to unmatched components
                            for curr_idx in unmatched_current:
                                new_label = next_new_label[int(cls)]
                                next_new_label[int(cls)] += 1
                                current_objects[new_label] = current_components[curr_idx]
                                print(f"  Slice {s}: New object {new_label} for unmatched component {curr_idx}")
                    
                    # Build prompts and masks
                    for dynamic_label, comp_mask in current_objects.items():
                        # Compute bounding box
                        yx = np.argwhere(comp_mask)
                        if len(yx) > 0:
                            y1, x1 = yx.min(axis=0)
                            y2, x2 = yx.max(axis=0)
                            bbox = [int(x1), int(y1), int(x2), int(y2)]
                            
                            # Center point
                            cy, cx = center_of_mass(comp_mask)
                            point = [int(round(cx)), int(round(cy)), 1]  # [x, y, label] (positive point)
                            
                            # Store into seg_masks and prompts
                            seg_masks[s][dynamic_label] = torch.from_numpy(comp_mask.astype(np.uint8)[None, ...])
                            prompts[s][dynamic_label] = {
                                "bboxes": bbox, 
                                "points": [point]  # Store as list (can hold multiple points)
                            }
                    
                    # Save for next frame
                    prev_objects = current_objects.copy()
                    
                    # Debug info
                    if len(current_objects) > 0:
                        print(f"  Slice {s}: Generated {len(current_objects)} prompts for objects {list(current_objects.keys())}")
                
                total_prompts = sum(len(objs) for objs in prompts.values())
                print(f"=== Object Tracking Completed: {total_prompts} total prompts ===")

                    
            output = {
                "meta": meta,
                "images": torch.stack(images),  # [num_slices, 3, H, W]
                "images_nnunet": images_nnunet,  # [num_slices, H, W]
            }
            if self.method in ["cls-det", "det"]:
                output.update({
                    "cls_labels": cls_labels,   # {frame_id: {object_id: class_id}}
                    "bboxes": bboxes            # {frame_id: {object_id: [x1,y1,x2,y2]}}
                })
            elif self.method == "seg":
                # prompts = {frame_id: {object_id: {"bboxes": , "points": }}}
                output.update({
                    "seg_masks": seg_masks,  # {frame_id: {object_id: mask}}
                    "prompts": prompts
                })
            elif self.method == "test":
                output.update({
                    "masks": torch.stack(masks),  # [num_slices, 3, H, W]
                    "prompts": prompts,  # {frame_id: {object_id: {"bboxes": , "points": }}}
                })
            return output
