"""Unit & Regression Tests for End-to-End Pipeline Scripts."""

import subprocess
import sys
import pytest


def run_pipeline_step(script_name: str):
    res = subprocess.run([sys.executable, f"pipeline/{script_name}"], capture_output=True, text=True)
    assert res.returncode == 0, f"{script_name} failed with stderr: {res.stderr}\nstdout: {res.stdout}"


def test_pipeline_step_01_ingestion():
    run_pipeline_step("01_ingestion.py")


def test_pipeline_step_02_cleaning():
    run_pipeline_step("02_cleaning.py")


def test_pipeline_step_03_preprocessing():
    run_pipeline_step("03_preprocessing.py")


def test_pipeline_step_04_feature_engineering():
    run_pipeline_step("04_feature_engineering.py")


def test_pipeline_step_05_data_splitting():
    run_pipeline_step("05_data_splitting.py")


def test_pipeline_step_06_training():
    run_pipeline_step("06_training.py")


def test_pipeline_step_07_evaluation():
    run_pipeline_step("07_evaluation.py")


def test_pipeline_step_08_prediction():
    run_pipeline_step("08_prediction.py")
