#!/usr/bin/env python
"""
Example Usage of Interactive Medical-SAM2
==========================================

This script demonstrates basic usage patterns for the Interactive Medical-SAM2 GUI.
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from medsam_gui_v5_multi import (
    run_napari_gui_with_navigation,
    DotDict
)
from func_3d.utils import get_network


def example_automatic_segmentation():
    """
    Example 1: Automatic segmentation with pre-trained model
    
    This example shows how to run automatic segmentation on a dataset
    using the Medical SAM 2 model.
    """
    print("\n" + "="*70)
    print("Example 1: Automatic Segmentation")
    print("="*70)
    
    # Configuration
    data_path = "./data/sample_patients"  # Replace with your data path
    checkpoint_path = "./checkpoints/sam2_hiera_small.pt"
    
    # Setup arguments
    args = DotDict(
        gpu=True,
        gpu_device=0,
        dataset="custom",
        net="sam2",
        sam_ckpt=checkpoint_path,
        sam_config="sam2_hiera_s",
        image_size=1024,
        data_path=data_path,
        plane='axial',
        version='Medical_sam2',
    )
    
    # Load network
    print("Loading Medical SAM 2 model...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_network(args, device=device)
    print(f"Model loaded on device: {device}")
    
    # Run GUI with automatic mode
    print("Launching Napari GUI in automatic mode...")
    run_napari_gui_with_navigation(
        data_root=data_path,
        net=net,
        device=device,
        args=args,
        default_mode='auto',
        default_method='seg'
    )


def example_manual_annotation():
    """
    Example 2: Manual annotation mode
    
    This example demonstrates interactive manual annotation with
    point and box prompts.
    """
    print("\n" + "="*70)
    print("Example 2: Manual Annotation")
    print("="*70)
    
    data_path = "./data/sample_patients"
    checkpoint_path = "./checkpoints/sam2_hiera_small.pt"
    
    args = DotDict(
        gpu=True,
        gpu_device=0,
        dataset="custom",
        net="sam2",
        sam_ckpt=checkpoint_path,
        sam_config="sam2_hiera_s",
        image_size=1024,
        data_path=data_path,
        plane='axial',
        version='Medical_sam2',
    )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_network(args, device=device)
    
    print("Launching Napari GUI in manual mode...")
    print("\nManual Mode Instructions:")
    print("  1. Use 'Add + Point' to add positive prompts (include regions)")
    print("  2. Use 'Add - Point' to add negative prompts (exclude regions)")
    print("  3. Use 'Add Box' to draw bounding boxes")
    print("  4. Use 'Edit Points' or 'Edit Boxes' to modify prompts")
    print("  5. Click 'Propagate' to generate segmentation")
    print("  6. Use 'Save Masks' to export results")
    
    run_napari_gui_with_navigation(
        data_root=data_path,
        net=net,
        device=device,
        args=args,
        default_mode='manual'
    )


def example_with_preprocessing():
    """
    Example 3: Using preprocessing pipeline
    
    This example shows how to apply N4 bias correction and intensity
    normalization before segmentation.
    """
    print("\n" + "="*70)
    print("Example 3: With Preprocessing")
    print("="*70)
    
    data_path = "./data/sample_patients"
    checkpoint_path = "./checkpoints/sam2_hiera_small.pt"
    
    args = DotDict(
        gpu=True,
        gpu_device=0,
        dataset="custom",
        net="sam2",
        sam_ckpt=checkpoint_path,
        sam_config="sam2_hiera_s",
        image_size=1024,
        data_path=data_path,
        plane='axial',
        version='Medical_sam2',
        preprocess=True,  # Enable preprocessing
    )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_network(args, device=device)
    
    print("Running with preprocessing enabled...")
    print("  - N4 bias field correction")
    print("  - Intensity normalization")
    print("  - Results will be saved in 'preprocessed' folder")
    
    run_napari_gui_with_navigation(
        data_root=data_path,
        net=net,
        device=device,
        args=args,
        default_mode='auto',
        default_method='seg'
    )


def example_brain_tumor_segmentation():
    """
    Example 4: Brain tumor segmentation workflow
    
    Specific example for brain tumor (meningioma) segmentation.
    """
    print("\n" + "="*70)
    print("Example 4: Brain Tumor Segmentation")
    print("="*70)
    
    # Assuming data is organized as: data_path/patient_id/T1.nii.gz
    data_path = "./data/brain_tumors/T1"
    checkpoint_path = "./Medical_SAM2_pretrain.pth"  # Use pre-trained weights
    
    args = DotDict(
        gpu=True,
        gpu_device=0,
        dataset="BrainTumor",
        net="sam2",
        sam_ckpt=checkpoint_path,
        sam_config="sam2_hiera_s",
        image_size=1024,
        data_path=data_path,
        plane='axial',
        version='Medical_sam2',
        preprocess=True,
    )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_network(args, device=device)
    
    print("Brain tumor segmentation workflow:")
    print("  1. Data is automatically loaded from DICOM or NIfTI")
    print("  2. Preprocessing is applied")
    print("  3. Automatic segmentation generates initial masks")
    print("  4. Review and refine using interactive prompts")
    print("  5. Visualize in 3D using '3D Volume Render'")
    print("  6. Export masks with preserved medical metadata")
    
    run_napari_gui_with_navigation(
        data_root=data_path,
        net=net,
        device=device,
        args=args,
        default_mode='auto',
        default_method='seg'
    )


def example_batch_processing():
    """
    Example 5: Batch processing multiple patients
    
    Shows how to efficiently process multiple patients sequentially.
    """
    print("\n" + "="*70)
    print("Example 5: Batch Processing")
    print("="*70)
    
    data_path = "./data/multiple_patients"
    checkpoint_path = "./checkpoints/sam2_hiera_small.pt"
    
    args = DotDict(
        gpu=True,
        gpu_device=0,
        dataset="custom",
        net="sam2",
        sam_ckpt=checkpoint_path,
        sam_config="sam2_hiera_s",
        image_size=1024,
        data_path=data_path,
        plane='axial',
        version='Medical_sam2',
    )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_network(args, device=device)
    
    print("Batch processing workflow:")
    print("  1. All patients in data_path are detected automatically")
    print("  2. Navigate between patients using 'Next Patient' button")
    print("  3. Each patient's settings can be configured independently")
    print("  4. Masks are saved automatically when moving to next patient")
    print("  5. Progress is tracked across the entire dataset")
    
    run_napari_gui_with_navigation(
        data_root=data_path,
        net=net,
        device=device,
        args=args,
        default_mode='manual'
    )


def print_usage_menu():
    """Print menu for selecting examples"""
    print("\n" + "="*70)
    print("Interactive Medical-SAM2 - Example Usage")
    print("="*70)
    print("\nSelect an example to run:")
    print("  1. Automatic segmentation")
    print("  2. Manual annotation")
    print("  3. With preprocessing")
    print("  4. Brain tumor segmentation")
    print("  5. Batch processing")
    print("  0. Exit")
    print("="*70)


if __name__ == "__main__":
    examples = {
        '1': example_automatic_segmentation,
        '2': example_manual_annotation,
        '3': example_with_preprocessing,
        '4': example_brain_tumor_segmentation,
        '5': example_batch_processing,
    }
    
    # Check if running with command-line argument
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            examples[choice]()
        else:
            print(f"Invalid choice: {choice}")
            print("Usage: python example_usage.py [1-5]")
    else:
        # Interactive menu
        while True:
            print_usage_menu()
            choice = input("\nEnter your choice (0-5): ").strip()
            
            if choice == '0':
                print("Exiting...")
                break
            elif choice in examples:
                try:
                    examples[choice]()
                except KeyboardInterrupt:
                    print("\n\nExample interrupted by user.")
                except Exception as e:
                    print(f"\nError running example: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Invalid choice: {choice}. Please enter 0-5.")
