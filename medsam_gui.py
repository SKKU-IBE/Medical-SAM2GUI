"""Application entry point for Interactive Medical-SAM2 GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gui.checkpoints import (
    CHECKPOINT_URL,
    UPSTREAM_PAGE,
    CheckpointError,
    default_checkpoint_path,
    download_checkpoint,
    resolve_checkpoint,
)


APP_VERSION = "1.1.0"


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Interactive Medical-SAM2 GUI.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Path to a Medical-SAM2 checkpoint. Overrides "
            "MEDICAL_SAM2_CHECKPOINT, the legacy working-directory file, and the user cache."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    return parser


def _select_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _download_with_dialog(app, QMessageBox, QProgressDialog) -> Path | None:
    destination = default_checkpoint_path()
    prompt = (
        "The Medical-SAM2 checkpoint is not bundled with this software.\n\n"
        f"Source: {CHECKPOINT_URL}\n"
        f"Upstream terms: {UPSTREAM_PAGE}\n\n"
        f"Download and verify it now?\n{destination}"
    )
    answer = QMessageBox.question(
        None,
        "Checkpoint Required",
        prompt,
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    if answer != QMessageBox.Yes:
        return None

    progress_dialog = QProgressDialog("Downloading Medical-SAM2 checkpoint...", "", 0, 100)
    progress_dialog.setWindowTitle("Checkpoint Download")
    progress_dialog.setCancelButton(None)
    progress_dialog.setMinimumDuration(0)
    progress_dialog.setValue(0)

    def update_progress(downloaded: int, total: int | None) -> None:
        if total:
            progress_dialog.setValue(min(99, int(downloaded * 100 / total)))
        app.processEvents()

    try:
        checkpoint = download_checkpoint(destination, force=True, progress=update_progress)
    except CheckpointError as exc:
        progress_dialog.close()
        QMessageBox.critical(None, "Checkpoint Error", str(exc))
        return None

    progress_dialog.setValue(100)
    progress_dialog.close()
    return checkpoint


def _resolve_medical_sam2_checkpoint(
    explicit_path: Path | None,
    app,
    QMessageBox,
    QProgressDialog,
) -> Path | None:
    try:
        checkpoint = resolve_checkpoint(explicit_path)
    except CheckpointError as exc:
        if explicit_path is not None:
            QMessageBox.critical(None, "Checkpoint Error", str(exc))
            return None
        answer = QMessageBox.question(
            None,
            "Checkpoint Error",
            f"{exc}\n\nReplace the managed checkpoint from the official upstream source?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return None
        return _download_with_dialog(app, QMessageBox, QProgressDialog)

    if checkpoint is not None:
        return checkpoint
    return _download_with_dialog(app, QMessageBox, QProgressDialog)


def main(argv: list[str] | None = None) -> int:
    cli_args = build_parser().parse_args(argv)

    import torch
    from qtpy.QtWidgets import QApplication, QDialog, QMessageBox, QProgressDialog

    from func_3d.utils import get_network
    from gui.navigation import run_napari_gui_with_navigation
    from gui.setup_dialogs import InitialSetupDialog

    print("=" * 60)
    print("Interactive Medical-SAM2 GUI")
    print("=" * 60)

    app = QApplication.instance() or QApplication([sys.argv[0]])
    setup_dialog = InitialSetupDialog()

    if setup_dialog.exec_() != QDialog.Accepted:
        print("Setup was cancelled.")
        app.quit()
        return 0

    settings = setup_dialog.get_settings()
    mode = settings["mode"]
    method = settings["method"]
    prep = settings["preprocess"]
    data_path = settings["data_path"]
    model_version = settings["version"]

    print("Selected settings:")
    print(f"  - Mode: {mode}")
    print(f"  - Method: {method}")
    print(f"  - Preprocessing: {prep}")
    print(f"  - Data path: {data_path}")
    print(f"  - Model version: {model_version}")
    print("-" * 60)

    if model_version == "Medical_sam2":
        checkpoint_path = _resolve_medical_sam2_checkpoint(
            cli_args.checkpoint,
            app,
            QMessageBox,
            QProgressDialog,
        )
        if checkpoint_path is None:
            print("A checkpoint is required to start the application.")
            app.quit()
            return 1
        exp_name = "Medical_SAM2"
        sam2_config = "sam2_hiera_t"
        image_size = 1024
    elif model_version == "MedSAM2":
        exp_name = "MedSAM2"
        checkpoint_path = cli_args.checkpoint or Path("MedSAM2_latest.pt")
        if not checkpoint_path.is_file():
            QMessageBox.critical(
                None,
                "Checkpoint Error",
                f"MedSAM2 checkpoint does not exist: {checkpoint_path}",
            )
            app.quit()
            return 1
        sam2_config = "sam2.1_hiera_t512"
        image_size = 512
    else:
        QMessageBox.critical(None, "Setup Error", f"Unsupported model version: {model_version}")
        app.quit()
        return 1

    model_args = DotDict(
        gpu=True,
        gpu_device=0,
        dataset="SNU_GaKn",
        net="sam2",
        exp_name=exp_name,
        sam_ckpt=str(checkpoint_path),
        sam_config=sam2_config,
        distributed=False,
        image_size=image_size,
        data_path=data_path,
        plane="axial",
        version=model_version,
    )

    try:
        device = _select_device(torch)
        print(f"Selecting device: {device}")
        net = get_network(
            model_args,
            model_args.net,
            use_gpu=(device != "cpu"),
            gpu_device=device,
            distribution=model_args.distributed,
        )
        net.to(device=device, dtype=torch.float32)
        print(f"Model loaded on {device}.")
    except Exception as exc:
        QMessageBox.critical(None, "Loading Error", f"Error occurred while loading model:\n{exc}")
        app.quit()
        return 1

    if mode == "auto":
        run_napari_gui_with_navigation(
            data_path,
            net,
            device,
            model_args,
            default_mode=mode,
            default_method=method,
        )
    elif mode == "manual":
        run_napari_gui_with_navigation(
            data_path,
            net,
            device,
            model_args,
            default_mode=mode,
            default_method=None,
        )
    else:
        run_napari_gui_with_navigation(
            data_path,
            net,
            device,
            model_args,
            default_mode="manual",
            default_method=None,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
