import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import face_landmarker as fl


def test_export_is_isolated_and_cached(tmp_path):
    def export(command, *, env, check):
        assert command[1:3] == ["-m", "face_landmarker_export"]
        assert env["CUDA_VISIBLE_DEVICES"] == "-1"
        assert check is True
        for name in ("blazeface.onnx", "facemesh_v2.onnx"):
            (Path(command[-1]) / name).touch()

    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), patch.object(
        fl.subprocess, "run", side_effect=export
    ) as run:
        paths = fl.ensure_models(tmp_path)
        assert all(path.exists() for path in paths)
        fl.ensure_models(tmp_path)
        run.assert_called_once()
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
        paths[0].unlink()
        fl.ensure_models(tmp_path)
        assert run.call_count == 2


def test_export_errors_propagate(tmp_path):
    with patch.object(fl.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "export")):
        with pytest.raises(subprocess.CalledProcessError):
            fl.ensure_models(tmp_path)


def test_auto_without_cuda_provider():
    with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
        with pytest.warns(RuntimeWarning, match="Using CPU"):
            assert fl._auto_device() == "cpu"


def test_auto_without_visible_gpu():
    cp = pytest.importorskip("cupy")
    with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider"]), \
            patch("onnxruntime.preload_dlls"), patch.object(cp.cuda.runtime, "getDeviceCount", return_value=0):
        with pytest.warns(RuntimeWarning, match="no visible CUDA"):
            assert fl._auto_device() == "cpu"


def test_explicit_cuda_without_provider():
    pytest.importorskip("cupy")
    with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
        with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
            fl.FaceLandmarker("missing.onnx", "missing.onnx", "cuda")


def test_auto_does_not_hide_model_errors():
    with patch.object(fl, "_auto_device", return_value="cpu"):
        with pytest.raises(Exception, match="missing.onnx"):
            fl.FaceLandmarker("missing.onnx", "missing.onnx", "auto")


@pytest.mark.integration
def test_auto_with_cpu_runtime():
    paths = tuple(Path(__file__).resolve().parents[1] / "models" / name
                  for name in ("blazeface.onnx", "facemesh_v2.onnx"))
    if not all(path.exists() for path in paths):
        pytest.skip("Exported models unavailable")
    with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
        with pytest.warns(RuntimeWarning, match="Using CPU"), fl.FaceLandmarker(*paths, device="auto") as pipeline:
            assert pipeline.device == "cpu"
            assert pipeline([np.zeros((8, 8, 3), np.uint8)])["landmarks"].shape == (1, 478, 3)


@pytest.mark.integration
def test_auto_when_ort_cannot_initialize_cuda():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA device unavailable")
    except RuntimeError:
        pytest.skip("CUDA device unavailable")
    import onnxruntime as ort

    paths = tuple(Path(__file__).resolve().parents[1] / "models" / name
                  for name in ("blazeface.onnx", "facemesh_v2.onnx"))
    if not all(path.exists() for path in paths):
        pytest.skip("Exported models unavailable")
    session = ort.InferenceSession
    with patch.object(fl, "_auto_device", return_value="cuda:0"), patch.object(
        ort, "InferenceSession", side_effect=lambda path, options, providers:
        session(path, options, providers=["CPUExecutionProvider"])
    ):
        with pytest.warns(RuntimeWarning, match="could not initialize"), fl.FaceLandmarker(*paths, "auto") as pipeline:
            assert pipeline.device == "cpu"
            assert np.isfinite(pipeline([np.zeros((8, 8, 3), np.uint8)])["landmarks"]).all()
