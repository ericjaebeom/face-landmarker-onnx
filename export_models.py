"""Export the two batched models; run with UV's export dependency group."""
# ONNXScript string dimensions are not Python forward references.
# ruff: noqa: F821, UP037

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import compose, helper, numpy_helper
from onnxscript import FLOAT, script
from onnxscript import opset18 as op

BUNDLE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
BUNDLE_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODEL_FORMAT = "face-landmark-onnx/1"


def anchors() -> np.ndarray:
    # MediaPipe SsdAnchorsCalculator: strides [8,16,16,16], fixed_anchor_size.
    return np.concatenate([
        np.repeat(
            np.stack(np.meshgrid((np.arange(size) + 0.5) / size,
                                 (np.arange(size) + 0.5) / size), axis=-1)
            .reshape(-1, 2), repeats, axis=0,
        )
        for size, repeats in ((16, 2), (8, 6))
    ]).astype(np.float32)


@script()
def _decode_and_select(
    regressors: FLOAT["N", 896, 16],
    classificators: FLOAT["N", 896, 1],
    anchor_centers: FLOAT[896, 2],
    iou_threshold: FLOAT,
) -> tuple[FLOAT["N", 16], FLOAT["N"]]:
    centers = regressors[:, :, :2] / 128.0 + anchor_centers
    sizes = regressors[:, :, 2:4] / 128.0
    boxes = op.Concat(centers - sizes / 2.0, centers + sizes / 2.0, axis=2)
    points = op.Reshape(regressors[:, :, 4:], [-1, 896, 6, 2]) / 128.0
    points = points + op.Unsqueeze(anchor_centers, [0, 2])
    decoded = op.Concat(boxes, op.Reshape(points, [-1, 896, 12]), axis=2)
    scores = op.Squeeze(op.Sigmoid(op.Clip(classificators, -100.0, 100.0)), [2])
    best = op.ArgMax(scores, axis=1, keepdims=1, select_last_index=0)
    indices = op.Expand(op.Unsqueeze(best, [2]), op.Shape(decoded[:, :1, :]))
    seed = op.GatherElements(decoded, indices, axis=1)

    extent = op.Max(boxes[:, :, 2:4] - boxes[:, :, :2], 0.0)
    area = extent[:, :, 0] * extent[:, :, 1]
    seed_extent = op.Max(seed[:, :, 2:4] - seed[:, :, :2], 0.0)
    seed_area = seed_extent[:, :, 0] * seed_extent[:, :, 1]
    intersection_extent = op.Max(
        op.Min(boxes[:, :, 2:4], seed[:, :, 2:4])
        - op.Max(boxes[:, :, :2], seed[:, :, :2]), 0.0,
    )
    intersection = intersection_extent[:, :, 0] * intersection_extent[:, :, 1]
    union = area + seed_area - intersection
    iou = intersection / op.Where(union > 0.0, union, 1.0)
    overlap = op.Or(iou > iou_threshold, op.Equal(op.Range(0, 896, 1), best))
    weights = scores * op.Cast(overlap, to=1)
    total = op.ReduceSum(weights, [1], keepdims=1)
    weighted = op.ReduceSum(decoded * op.Unsqueeze(weights, [2]), [1], keepdims=0)
    detections = op.Where(
        total > 0.0, weighted / op.Where(total > 0.0, total, 1.0),
        op.Squeeze(seed, [1]),
    )
    detection_scores = op.Squeeze(op.GatherElements(scores, best, axis=1), [1])
    return detections, detection_scores


@script()
def _landmark_outputs(
    raw_landmarks: FLOAT["N", 1, 1, 1434],
    raw_presence: FLOAT["N", 1, 1, 1],
    raw_tongue: FLOAT["N", 1],
) -> tuple[FLOAT["N", 478, 3], FLOAT["N"], FLOAT["N"]]:
    landmarks = op.Reshape(raw_landmarks, [-1, 478, 3])
    presence_scores = op.Sigmoid(op.Reshape(raw_presence, [-1]))
    tongue_out = op.Reshape(raw_tongue, [-1])
    return landmarks, presence_scores, tongue_out


def detector_postprocess(iou_threshold: float = 0.5) -> onnx.ModelProto:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    model = _decode_and_select.to_model_proto()
    model.ir_version = 10
    model.graph.initializer.extend([
        numpy_helper.from_array(anchors(), "anchor_centers"),
        numpy_helper.from_array(np.array(iou_threshold, np.float32), "iou_threshold"),
    ])
    del model.graph.input[2:]
    return model


def download_bundle(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "face_landmarker_v1.task"
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == BUNDLE_SHA256:
        return path
    with urllib.request.urlopen(BUNDLE_URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != BUNDLE_SHA256:
        raise RuntimeError("The downloaded Face Landmarker bundle has an unexpected SHA-256")
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as output:
        output.write(data)
        temporary = Path(output.name)
    temporary.replace(path)
    return path


def _rename(model: onnx.ModelProto, names: dict[str, str]) -> None:
    for node in model.graph.node:
        for fields in (node.input, node.output):
            for i, name in enumerate(fields):
                fields[i] = names.get(name, name)
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info,
                  *model.graph.initializer):
        value.name = names.get(value.name, value.name)


def _dynamic_detector(model: onnx.ModelProto) -> None:
    # Patch only prediction heads, never arbitrary constants containing a 1.
    initializers = {x.name: x for x in model.graph.initializer}
    head_shapes = []
    for node in model.graph.node:
        if node.op_type != "Reshape" or node.input[1] not in initializers:
            continue
        shape = numpy_helper.to_array(initializers[node.input[1]])
        if shape.shape != (3,) or shape[0] != 1 or shape[2] not in (1, 16):
            continue
        # Zero copies the actual batch axis. The inferred anchor axis then
        # remains per image, including when the original shape is [1,-1,C].
        shape = shape.copy()
        shape[0] = 0
        name = node.name + "/batch_shape"
        model.graph.initializer.append(numpy_helper.from_array(shape, name))
        node.input[1] = name
        head_shapes.append(int(shape[2]))
    if sorted(head_shapes) != [1, 1, 16, 16]:
        raise RuntimeError(f"Unexpected BlazeFace prediction-head shapes: {head_shapes}")
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "N"
    for output in model.graph.output:
        output.type.tensor_type.shape.dim[0].dim_param = "N"
    del model.graph.value_info[:]
    # Remove constants no longer used by the repaired heads.
    used = {name for node in model.graph.node for name in node.input}
    kept = [x for x in model.graph.initializer if x.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def _join(base: onnx.ModelProto, post: onnx.ModelProto,
          connections: list[tuple[str, str]]) -> onnx.ModelProto:
    base.ir_version = post.ir_version = 10
    post = compose.add_prefix(post, "post/")
    model = compose.merge_models(base, post, [(a, "post/" + b) for a, b in connections])
    _rename(model, {
        model.graph.input[0].name: "images",
        **{output.name: output.name.removeprefix("post/") for output in model.graph.output},
    })
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    for value in (*model.graph.input, *model.graph.output):
        value.type.tensor_type.shape.dim[0].dim_param = "N"
    return model


def build_models(bundle: Path, work_dir: Path, iou_threshold: float = 0.5):
    # Export uses TensorFlow only on CPU; inference has no TensorFlow dependency.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    import tf2onnx

    with zipfile.ZipFile(bundle) as archive:
        for name in ("face_detector.tflite", "face_landmarks_detector.tflite"):
            (work_dir / name).write_bytes(archive.read(name))
    detector, _ = tf2onnx.convert.from_tflite(
        str(work_dir / "face_detector.tflite"), opset=18, inputs_as_nchw=["input"],
    )
    _dynamic_detector(detector)
    detector = _join(detector, detector_postprocess(iou_threshold), [
        ("regressors", "regressors"), ("classificators", "classificators"),
    ])
    mesh, _ = tf2onnx.convert.from_tflite(
        str(work_dir / "face_landmarks_detector.tflite"), opset=18,
        inputs_as_nchw=["input_12"], shape_override={"input_12": [-1, 256, 256, 3]},
    )
    mesh = _join(mesh, _landmark_outputs.to_model_proto(), [
        ("Identity", "raw_landmarks"), ("Identity_1", "raw_presence"),
        ("Identity_2", "raw_tongue"),
    ])
    for model, kind in ((detector, "detector"), (mesh, "landmarks")):
        helper.set_model_props(model, {
            "format": MODEL_FORMAT, "kind": kind,
            "bundle_url": BUNDLE_URL, "bundle_sha256": BUNDLE_SHA256,
            **({"iou_threshold": str(iou_threshold)} if kind == "detector" else {}),
        })
    return detector, mesh


def validate_model(model: onnx.ModelProto, size: int) -> None:
    import onnxruntime as ort

    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(model.SerializeToString(), options,
                                   providers=["CPUExecutionProvider"])
    images = np.random.default_rng(0).random((2, 3, size, size), dtype=np.float32)
    batch = session.run(None, {"images": images})
    singles = [session.run(None, {"images": image[None]}) for image in images]
    for index, output in enumerate(batch):
        if output.shape[0] != 2 or not np.isfinite(output).all():
            raise RuntimeError("Exported model failed its dynamic-batch smoke test")
        np.testing.assert_allclose(output, np.concatenate([s[index] for s in singles]),
                                   rtol=2e-4, atol=2e-4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.iou_threshold <= 1:
        parser.error("--iou-threshold must be between 0 and 1")
    bundle = download_bundle(Path.home() / ".cache" / "face-landmark-onnx")
    print("Converting the verified Face Landmarker v1 bundle...", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir) as temporary:
        work = Path(temporary)
        models = build_models(bundle, work, args.iou_threshold)
        for model, size, name in zip(models, (128, 256), ("blazeface.onnx", "facemesh_v2.onnx")):
            print(f"Validating {name}...", flush=True)
            validate_model(model, size)
            onnx.save(model, work / name)
        for name in ("blazeface.onnx", "facemesh_v2.onnx"):
            (work / name).replace(args.output_dir / name)
            print(args.output_dir / name)


if __name__ == "__main__":
    main()
