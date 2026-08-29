from __future__ import annotations

import platform
import sys
from pathlib import Path

import cv2
import fastapi
import pandas
import streamlit
import torch
import ultralytics

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_python_and_machine() -> None:
    assert sys.version_info[:2] == (3, 11)
    assert platform.machine() == "arm64"


def test_required_packages_imported() -> None:
    versions = (
        torch.__version__,
        ultralytics.__version__,
        cv2.__version__,
        pandas.__version__,
        fastapi.__version__,
        streamlit.__version__,
    )
    assert all(versions)


def test_mps_is_built_and_available() -> None:
    assert torch.backends.mps.is_built()
    assert torch.backends.mps.is_available()


def test_stage_zero_directories_exist() -> None:
    required = (
        "configs",
        "data/raw",
        "data/interim",
        "data/processed",
        "data/manifests",
        "models/pretrained",
        "models/finetuned",
        "outputs/videos",
        "outputs/detections",
        "outputs/tracks",
        "outputs/events",
        "outputs/analytics",
        "outputs/evidence",
        "src/vision_analytics/video",
        "src/vision_analytics/detection",
        "src/vision_analytics/tracking",
        "src/vision_analytics/spatial",
        "src/vision_analytics/events",
        "src/vision_analytics/analytics",
        "src/vision_analytics/services",
        "src/vision_analytics/utils",
        "scripts",
        "tests",
    )
    assert all((PROJECT_ROOT / path).is_dir() for path in required)

