#!/usr/bin/env python3
"""Validate the Stage 0 runtime and run a YOLO26n MPS smoke test."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

EXPECTED_PYTHON = (3, 11)
MODEL_NAME = "yolo26n.pt"
IMAGE_SIZE = 640
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "pretrained"


def print_check(label: str, value: object) -> None:
    print(f"{label}: {value}")


def main() -> int:
    failures: list[str] = []

    print("=== Stage 0 Environment Check ===")
    print_check("Python version", platform.python_version())
    print_check("Machine architecture", platform.machine())

    if sys.version_info[:2] != EXPECTED_PYTHON:
        failures.append(
            f"Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]} is required"
        )
    if platform.machine() != "arm64":
        failures.append("Apple Silicon arm64 architecture is required")

    try:
        import cv2
        import fastapi
        import numpy as np
        import pandas
        import streamlit
        import torch
        import ultralytics
        from ultralytics import YOLO
    except Exception as exc:
        print_check("Import check", f"FAIL ({type(exc).__name__}: {exc})")
        return 1

    print_check("PyTorch version", torch.__version__)
    print_check("torch.backends.mps.is_built()", torch.backends.mps.is_built())
    print_check("torch.backends.mps.is_available()", torch.backends.mps.is_available())
    print_check("OpenCV version", cv2.__version__)
    print_check("Ultralytics version", ultralytics.__version__)
    print_check("pandas import", f"PASS ({pandas.__version__})")
    print_check("FastAPI import", f"PASS ({fastapi.__version__})")
    print_check("Streamlit import", f"PASS ({streamlit.__version__})")
    print_check("PYTORCH_ENABLE_MPS_FALLBACK", os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "unset"))

    if not torch.backends.mps.is_built():
        failures.append("PyTorch was not built with MPS support")
    if not torch.backends.mps.is_available():
        failures.append("MPS is not available; CPU fallback is not accepted")

    if failures:
        for failure in failures:
            print_check("Failure", failure)
        print_check("Stage 0 environment check", "FAIL")
        return 1

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    try:
        # Ultralytics downloads a named official asset into the current directory.
        os.chdir(MODEL_DIR)
        model = YOLO(MODEL_NAME)
    except Exception as exc:
        print_check("YOLO26n load", f"FAIL ({type(exc).__name__}: {exc})")
        print_check("Stage 0 environment check", "FAIL")
        return 1
    finally:
        os.chdir(previous_cwd)

    model_path = MODEL_DIR / MODEL_NAME
    print_check("YOLO26n load", f"PASS ({model_path})")

    synthetic_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    cv2.rectangle(synthetic_image, (160, 160), (480, 480), (255, 255, 255), -1)

    try:
        results = model.predict(
            source=synthetic_image,
            imgsz=IMAGE_SIZE,
            device="mps",
            save=False,
            verbose=False,
        )
    except Exception as exc:
        print_check("YOLO26n MPS inference", f"FAIL ({type(exc).__name__}: {exc})")
        print_check("Stage 0 environment check", "FAIL")
        return 1

    if not results or results[0] is None:
        print_check("YOLO26n MPS inference", "FAIL (no result returned)")
        print_check("Stage 0 environment check", "FAIL")
        return 1

    print_check("Synthetic image", f"{IMAGE_SIZE}x{IMAGE_SIZE}")
    print_check("YOLO26n MPS inference", f"PASS ({len(results)} result returned)")
    print_check("Stage 0 environment check", "PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

