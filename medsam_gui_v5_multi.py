import sys
import torch
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox

from func_3d.utils import get_network
from gui.navigation import run_napari_gui_with_navigation
from gui.setup_dialogs import InitialSetupDialog


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":
    print("=" * 60)
    print("Interactive Medical SAM2 GUI")
    print("=" * 60)

    app = QApplication([])
    setup_dialog = InitialSetupDialog()

    if setup_dialog.exec_() != QDialog.Accepted:
        print("Setup was cancelled.")
        app.quit()
        sys.exit()

    settings = setup_dialog.get_settings()
    mode = settings['mode']
    method = settings['method']
    prep = settings['preprocess']
    data_path = settings['data_path']
    version = settings['version']

    print("Selected settings:")
    print(f"  - Mode: {mode}")
    print(f"  - Method: {method}")
    print(f"  - Preprocessing: {prep}")
    print(f"  - Data path: {data_path}")
    print(f"  - Model version: {version}")
    print("-" * 60)

    if version == 'Medical_sam2':
        exp_name = 'Medical_SAM2'
        checkpoint_path = "Medical_SAM2_pretrain.pth"
        sam2_config = 'sam2_hiera_t'
        image_size = 1024
    elif version == 'MedSAM2':
        exp_name = 'MedSAM2'
        checkpoint_path = "MedSAM2_latest.pt"
        sam2_config = 'sam2.1_hiera_t512'
        image_size = 512
    else:
        QMessageBox.critical(None, "Setup Error", f"Unsupported model version: {version}")
        sys.exit(1)

    args = DotDict(
        gpu=True,
        gpu_device=0,
        dataset="SNU_GaKn",
        net="sam2",
        exp_name=exp_name,
        sam_ckpt=checkpoint_path,
        sam_config=sam2_config,
        distributed=False,
        image_size=image_size,
        data_path=data_path,
        plane='axial',
        version=version,
    )

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=device, distribution=args.distributed)
        net.to(dtype=torch.float32)
        print(f"Model loaded on {device}.")
    except Exception as e:
        QMessageBox.critical(None, "Loading Error", f"Error occurred while loading model:\n{str(e)}")
        app.quit()
        sys.exit(1)

    if mode == 'auto':
        run_napari_gui_with_navigation(data_path, net, device, args, default_mode=mode, default_method=method)
    elif mode == 'manual':
        run_napari_gui_with_navigation(data_path, net, device, args, default_mode=mode, default_method=None)
    else:
        run_napari_gui_with_navigation(data_path, net, device, args, default_mode='manual', default_method=None)

    sys.exit()
