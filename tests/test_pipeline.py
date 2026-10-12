"""Run with: uv run --extra cuda --extra export --group dev pytest"""

import hashlib
import math
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image

from face_landmarker_export import anchors, detector_postprocess, download_bundle
from face_landmarker import FaceLandmarker, _project_cpu, _roi_cpu, _warp_cpu

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = (ROOT / "models/blazeface.onnx", ROOT / "models/facemesh_v2.onnx")


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    return ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])


def reference_detection(raw, logits, threshold=0.5):
    # Scalar, independent reference for one image; deliberately no ONNX helpers.
    centers = [(x + .5, y + .5, grid)
               for grid, count in ((16, 2), (8, 6))
               for y in range(grid) for x in range(grid) for _ in range(count)]
    decoded = np.empty((896, 16), np.float64)
    probabilities = 1 / (1 + np.exp(-np.clip(logits.reshape(-1).astype(np.float64), -100, 100)))
    for i, (ax, ay, grid) in enumerate(centers):
        cx, cy = raw[i, 0] / 128 + ax / grid, raw[i, 1] / 128 + ay / grid
        w, h = raw[i, 2] / 128, raw[i, 3] / 128
        decoded[i, :4] = cx-w/2, cy-h/2, cx+w/2, cy+h/2
        for k in range(6):
            decoded[i, 4+2*k] = raw[i, 4+2*k] / 128 + ax/grid
            decoded[i, 5+2*k] = raw[i, 5+2*k] / 128 + ay/grid
    best = int(np.argmax(probabilities))
    seed = decoded[best]
    candidates = []
    for i, candidate in enumerate(decoded):
        intersection = max(0, min(seed[2], candidate[2])-max(seed[0], candidate[0])) * max(
            0, min(seed[3], candidate[3])-max(seed[1], candidate[1]))
        union = (max(0, seed[2]-seed[0])*max(0, seed[3]-seed[1])
                 + max(0, candidate[2]-candidate[0])*max(0, candidate[3]-candidate[1]) - intersection)
        if i == best or (union > 0 and intersection / union > threshold):
            candidates.append(i)
    return np.average(decoded[candidates], axis=0, weights=probabilities[candidates]), probabilities[best]


def encoded_boxes(boxes, probabilities):
    raw = np.zeros((1, 896, 16), np.float32)
    logits = np.full((1, 896, 1), -100, np.float32)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        center = np.array([(x0+x1)/2, (y0+y1)/2], np.float32)
        raw[0, i, :2] = (center - anchors()[i]) * 128
        raw[0, i, 2:4] = np.array([x1-x0, y1-y0]) * 128
        raw[0, i, 4:] = np.tile((center - anchors()[i]) * 128, 6)
        logits[0, i, 0] = np.log(probabilities[i] / (1-probabilities[i]))
    return raw, logits


def test_anchor_order():
    a = anchors()
    assert a.shape == (896, 2)
    np.testing.assert_array_equal(a[:2], np.full((2, 2), 1/32, np.float32))
    np.testing.assert_array_equal(a[2], [3/32, 1/32])
    np.testing.assert_array_equal(a[510:512], np.full((2, 2), 31/32, np.float32))
    np.testing.assert_array_equal(a[512:518], np.full((6, 2), 1/16, np.float32))
    np.testing.assert_array_equal(a[518], [3/16, 1/16])


@pytest.mark.parametrize("boxes,probabilities,threshold", [
    ([[0, 0, 1, 1], [.1, .1, 1.1, 1.1], [2, 2, 3, 3]], [.9, .8, .7], .5),
    ([[0, 0, 3, 1], [1, 0, 4, 1]], [.9, .8], .5),  # Exactly on the IoU boundary.
    ([[0, 0, 1, 1], [2, 2, 3, 3]], [.8, .8], .5),  # First maximum wins.
    ([[0, 0, 1, 1], [.1, .1, 1.1, 1.1]], [.001, .002], .5),
    ([[0, 0, 0, 0]], [.001], .5),
    ([[0, 0, 1, 1], [0, 0, 1, 1]], [.9, .8], 1.0),
])
def test_weighted_nms(boxes, probabilities, threshold):
    raw, logits = encoded_boxes(boxes, probabilities)
    model = session(detector_postprocess(threshold).SerializeToString())
    actual = model.run(None, {"regressors": raw, "classificators": logits})
    expected, score = reference_detection(raw[0], logits[0], threshold)
    np.testing.assert_allclose(actual[0][0], expected, atol=1e-6)
    np.testing.assert_allclose(actual[1][0], score, atol=1e-7)


def test_nms_underflow_fallback():
    raw = np.zeros((2, 896, 16), np.float32)
    raw[:, :, 2:4] = 32
    logits = np.full((2, 896, 1), -1e10, np.float32)
    result = session(detector_postprocess().SerializeToString()).run(
        None, {"regressors": raw, "classificators": logits})
    assert all(np.isfinite(x).all() for x in result)
    # ORT's sigmoid may flush the clipped negative logits to exactly zero.
    if result[1][0] == 0:
        np.testing.assert_allclose(result[0][0, :4], [-.09375, -.09375, .15625, .15625])


def test_detector_padding_and_bilinear_sampling():
    image = np.full((2, 4, 3), 255, np.uint8)
    meta = np.array([[0, 4, 2]], np.int64)
    transform = np.array([[1, 0, 0, 0, 1, -1, 0]], np.float32)
    out = np.empty((1, 3, 4, 4), np.float32)
    _warp_cpu(image.ravel(), meta, transform, 4, False, out)
    np.testing.assert_allclose(out[:, :, [0, 3]], -1)
    np.testing.assert_allclose(out[:, :, [1, 2]], 1)
    image = np.arange(4*4*3, dtype=np.uint8).reshape(4, 4, 3)
    meta[0] = 0, 4, 4
    transform[0] = 0, 0, .25, 0, 0, .75, 0
    _warp_cpu(image.ravel(), meta, transform, 1, True, out[:, :, :1, :1])
    expected = ((image[0, 0]*.75+image[0, 1]*.25)*.25
                +(image[1, 0]*.75+image[1, 1]*.25)*.75)/255
    np.testing.assert_allclose(out[0, :, 0, 0], expected, atol=1e-7)
    transform[0, 2] = -50
    transform[0, 5] = 50
    _warp_cpu(image.ravel(), meta, transform, 1, True, out[:, :, :1, :1])
    np.testing.assert_allclose(out[0, :, 0, 0], image[-1, 0]/255, atol=1e-7)


@pytest.mark.parametrize("width,height,angle", [(640, 320, 0), (320, 640, .6), (640, 320, -.7)])
def test_roi_and_projection(width, height, angle):
    meta = np.array([[0, width, height]], np.int64)
    det = np.zeros((1, 16), np.float32)
    det[0, :4] = [.3, .35, .7, .65]
    det[0, 4:8] = [.4, .4, .4+.2*math.cos(angle), .4+.2*math.sin(angle)]
    transforms = np.empty((1, 7), np.float32)
    _roi_cpu(det, meta, transforms)
    co, si, side = math.cos(angle), math.sin(angle), max(width, height)
    rw, rh = .4*side*1.5, .3*side*1.5
    a, b, c, d = co*rw/256, -si*rh/256, si*rw/256, co*rh/256
    expected = [a, b, width/2-128*(a+b), c, d, height/2-128*(c+d),
                math.hypot(a/width, c/height)*width]
    np.testing.assert_allclose(transforms[0], expected, atol=1e-4, rtol=1e-6)
    landmarks = np.full((1, 478, 3), [128, 128, 10], np.float32)
    packed = np.empty((1, 1453), np.float32)
    score = np.array([.8], np.float32)
    for normalized in (False, True):
        _project_cpu(det, score, landmarks, score, score, meta, transforms, normalized, packed)
        xyz = np.array([width/2, height/2, expected[-1]*10])
        if normalized:
            xyz /= [width, height, width]
        np.testing.assert_allclose(packed[0, 17:20], xyz, atol=1e-4, rtol=1e-6)


def test_degenerate_roi():
    metadata = np.array([[0, 320, 240]], np.int64)
    transforms = np.empty((1, 7), np.float32)
    _roi_cpu(np.zeros((1, 16), np.float32), metadata, transforms)
    assert np.isfinite(transforms).all()
    np.testing.assert_allclose(transforms[0, [0, 4]], 1.5/256)
    np.testing.assert_allclose(transforms[0, [1, 3]], 0)


@pytest.fixture(scope="session")
def portrait():
    path = ROOT / ".cache/portrait.jpg"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(urllib.request.urlopen(
            "https://storage.googleapis.com/mediapipe-assets/portrait.jpg", timeout=30).read())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "a6f11efaa834706db23f275b6115058fa87fc7f14362681e6abe14e82749de3e"
    return Image.open(path).convert("RGB")


@pytest.fixture(scope="session")
def images(portrait):
    return [np.array(portrait.resize(size)) for size in ((320, 480), (512, 256), (257, 321))]


@pytest.fixture(scope="session")
def cpu():
    if not all(p.exists() for p in MODEL_PATHS):
        pytest.skip("Run face-landmarker-export first")
    with FaceLandmarker(*MODEL_PATHS, num_threads=2) as pipeline:
        yield pipeline


@pytest.fixture(scope="session")
def gpu():
    if not all(p.exists() for p in MODEL_PATHS):
        pytest.skip("Run face-landmarker-export first")
    cp = pytest.importorskip("cupy")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider unavailable")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA driver unavailable")
    if not count:
        pytest.skip("CUDA device unavailable")
    with FaceLandmarker(*MODEL_PATHS, device="cuda") as pipeline:
        yield pipeline


@pytest.mark.integration
@pytest.mark.parametrize("count", [1, 2, 7, 32])
def test_batch_matches_singles(cpu, images, count):
    batch = [images[i % len(images)] for i in range(count)]
    actual = cpu(batch)
    singles = [cpu([image]) for image in images]
    for key, value in actual.items():
        expected = np.concatenate([singles[i % len(images)][key] for i in range(count)])
        np.testing.assert_allclose(value, expected, atol=3e-3, rtol=3e-4)


@pytest.mark.integration
def test_coordinates_and_lifetime(cpu, images):
    pixels = cpu(images)
    saved = {key: value.copy() for key, value in pixels.items()}
    normalized = cpu(images, normalized=True)
    dims = np.array([[image.shape[1], image.shape[0]] for image in images])
    for key, value in pixels.items():
        np.testing.assert_array_equal(value, saved[key])
    np.testing.assert_allclose(pixels["boxes"], normalized["boxes"] * np.tile(dims, (1, 2)), atol=1e-4)
    np.testing.assert_allclose(pixels["keypoints"], normalized["keypoints"] * dims[:, None], atol=1e-4)
    xyz_dims = dims[:, [0, 1, 0]]
    np.testing.assert_allclose(pixels["landmarks"], normalized["landmarks"] * xyz_dims[:, None], atol=1e-4)
    assert pixels["landmarks"].shape == (3, 478, 3)
    assert all(value.dtype == np.float32 for value in pixels.values())
    assert all(np.isfinite(value).all() for value in pixels.values())


@pytest.mark.integration
def test_empty_blank_stacked_and_invalid_inputs(cpu, images):
    empty = cpu([])
    assert empty["boxes"].shape == (0, 4)
    assert empty["landmarks"].shape == (0, 478, 3)
    assert cpu(np.empty((0, 12, 12, 3), np.uint8))["landmarks"].shape == (0, 478, 3)
    blank = cpu([np.zeros((1, 1, 3), np.uint8), np.zeros((300, 500, 3), np.uint8)])
    assert blank["landmarks"].shape == (2, 478, 3)
    assert all(np.isfinite(x).all() for x in blank.values())
    stacked = cpu(np.stack([images[0], images[0]]))
    np.testing.assert_allclose(stacked["landmarks"][0], stacked["landmarks"][1])
    for bad in [np.zeros((12, 12), np.uint8), np.zeros((12, 12, 4), np.uint8),
                np.zeros((12, 12, 3), np.float32), np.zeros((0, 12, 3), np.uint8), "path.jpg"]:
        with pytest.raises(ValueError):
            cpu([bad])
    with pytest.raises(ValueError):
        cpu(images[0])
    with pytest.raises(ValueError):
        cpu([], return_gpu=True)


@pytest.mark.integration
def test_noncontiguous(cpu, images):
    sliced = images[0][::-2, ::-1]
    actual, expected = cpu([sliced]), cpu([np.ascontiguousarray(sliced)])
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.integration
def test_parallel_image_packing(cpu):
    images = [np.full((512, 512, 3), [i, i+64, i+128], np.uint8)[:, ::-1]
              for i in range(48)]
    pixels, metadata, _ = cpu._prepare(images)
    assert pixels.nbytes > 32 * 1024**2
    for image, (offset, width, height) in zip(images, metadata):
        restored = pixels[offset:offset+image.size].reshape(height, width, 3)
        np.testing.assert_array_equal(restored, image)


@pytest.mark.integration
def test_close(images):
    with FaceLandmarker(*MODEL_PATHS, num_threads=1) as pipeline:
        result = pipeline(images[:1])
    assert np.isfinite(result["landmarks"]).all()
    pipeline.close()
    with pytest.raises(RuntimeError, match="closed"):
        pipeline(images)


@pytest.mark.integration
@pytest.mark.cuda
def test_cuda(cpu, gpu, images):
    import cupy as cp

    expected = cpu(images, normalized=True)
    actual = gpu(images, normalized=True)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-3, rtol=2e-3)
    device = gpu(images, normalized=True, return_gpu=True)
    gpu([np.zeros((3, 4, 3), np.uint8)])
    for key, value in device.items():
        assert isinstance(value, cp.ndarray)
        np.testing.assert_allclose(value.get(), actual[key], atol=2e-4, rtol=2e-4)
    assert gpu([], return_gpu=True)["landmarks"].shape == (0, 478, 3)


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.parametrize("count", [1, 2, 7, 32])
def test_cuda_batch_matches_singles(gpu, images, count):
    singles = [gpu([image], normalized=True) for image in images]
    batch = gpu([images[i % 3] for i in range(count)], normalized=True)
    for key in batch:
        expected = np.concatenate([singles[i % 3][key] for i in range(count)])
        np.testing.assert_allclose(batch[key], expected, atol=2e-3, rtol=2e-3)


@pytest.mark.integration
@pytest.mark.cuda
def test_cuda_geometry(gpu, images):
    import cupy as cp

    # Identical inputs to each utility isolate geometry from neural differences.
    image = images[0]
    meta = np.array([[0, image.shape[1], image.shape[0]]], np.int64)
    det = np.array([[.3, .3, .7, .65, .4, .4, .6, .5] + [.5]*8], np.float32)
    cpu_tr = np.empty((1, 7), np.float32)
    _roi_cpu(det, meta, cpu_tr)
    with gpu._device, gpu._stream:
        gpu_tr, gpu_meta = cp.empty((1, 7), np.float32), cp.asarray(meta)
        gpu._roi(cp.asarray(det), gpu_meta, gpu_tr)
        actual = gpu._warp(cp.asarray(image.ravel()), gpu_meta, gpu_tr, 256).get()
        transform = gpu_tr.get()
    expected = np.empty((1, 3, 256, 256), np.float32)
    _warp_cpu(image.ravel(), meta, cpu_tr, 256, True, expected)
    np.testing.assert_allclose(transform, cpu_tr, atol=2e-5, rtol=1e-6)
    np.testing.assert_allclose(actual, expected, atol=5e-5)


@pytest.mark.integration
@pytest.mark.cuda
def test_cuda_shutdown_in_subprocess(gpu):
    # An async ORT run option previously deferred pinned-buffer cleanup until
    # destruction and aborted the process. Test both close and implicit exit.
    code = f'''
import numpy as np
from face_landmarker import FaceLandmarker
paths = {tuple(map(str, MODEL_PATHS))!r}
image = np.zeros((128, 128, 3), np.uint8)
for explicit in [True, False]:
    pipeline = FaceLandmarker(*paths, device="cuda")
    for n in [1, 8, 32, 8, 32, 1]:
        pipeline([image] * n)
    if explicit:
        pipeline.close()
'''
    process = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=False,
                             capture_output=True, text=True, timeout=90)
    assert process.returncode == 0, process.stderr


@pytest.mark.integration
def test_original_tflite_models(cpu, images):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    bundle = download_bundle(Path.home() / ".cache/face-landmark-onnx")
    with zipfile.ZipFile(bundle) as archive:
        detector = tf.lite.Interpreter(model_content=archive.read("face_detector.tflite"), num_threads=1)
        mesh = tf.lite.Interpreter(model_content=archive.read("face_landmarks_detector.tflite"), num_threads=1)
    for interpreter in (detector, mesh):
        interpreter.allocate_tensors()

    def invoke(interpreter, tensor):
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], tensor[None].transpose(0, 2, 3, 1))
        interpreter.invoke()
        return [interpreter.get_tensor(out["index"]) for out in interpreter.get_output_details()]

    pixels, metadata, transforms = cpu._prepare(images)
    detector_inputs = cpu._warp(pixels, metadata, transforms, 128)
    detections, scores = cpu._run(0, detector_inputs)
    for i, tensor in enumerate(detector_inputs):
        raw, logits = invoke(detector, tensor)
        expected, score = reference_detection(raw[0], logits[0])
        np.testing.assert_allclose(detections[i], expected, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(scores[i], score, atol=1e-6)
    cpu._roi(detections, metadata, transforms)
    mesh_inputs = cpu._warp(pixels, metadata, transforms, 256)
    actual = cpu._run(1, mesh_inputs)
    for i, tensor in enumerate(mesh_inputs):
        points, presence, tongue = invoke(mesh, tensor)
        np.testing.assert_allclose(actual[0][i], points.reshape(478, 3), atol=1e-3, rtol=1e-4)
        np.testing.assert_allclose(actual[1][i], 1/(1+np.exp(-presence.item())), atol=1e-5)
        np.testing.assert_allclose(actual[2][i], tongue.item(), atol=1e-5)
