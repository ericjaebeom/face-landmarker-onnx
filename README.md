# Face Landmarker ONNX

Batched MediaPipe BlazeFace + FaceMesh-V2 inference on CPU or NVIDIA CUDA.
Every input image gets one prediction, including images with low confidence or
no face. Nothing is filtered by detection or presence confidence.

## Export models

Requires [UV](https://docs.astral.sh/uv/) and Python 3.12 (UV can install Python).

```bash
uv run --extra cpu --extra export face-landmarker-export --output-dir models
```

This downloads the official Face Landmarker v1 bundle and writes
`models/blazeface.onnx` and `models/facemesh_v2.onnx`. Both accept dynamic batches
of float32 NCHW tensors. The detector includes decoding and single-face weighted
NMS; FaceMesh returns reshaped landmarks and probabilities. The export command
can be rerun. Use `--iou-threshold 0.3` to change the default overlap threshold of
0.5. There is no confidence threshold.

## Install and run

Install from this checkout or from the public repository:

```bash
uv pip install "face-landmark-onnx[cuda,export] @ git+https://github.com/ericjaebeom/face-landmarker-onnx@1b35341b4b59f36aa89a459441b143cc763908c1"
```

Choose one runtime when working in the checkout:

```bash
uv sync --extra cpu
# Or, for NVIDIA CUDA (also supports CPU inference):
uv sync --extra cuda
```

The CUDA extra installs CUDA 12 libraries and cuDNN; a compatible NVIDIA driver
is required. Runtime inference uses neither TensorFlow nor MediaPipe.

```python
from face_landmarker import FaceLandmarker, ensure_models

# images: list of RGB uint8 NumPy arrays, each shaped (height, width, 3).
# Resolutions may differ. A stacked (N, H, W, 3) array also works.
with FaceLandmarker(
    *ensure_models("models"),  # requires the export extra on a fresh cache
    device="auto",  # or "cpu", "cuda", "cuda:0"
) as pipeline:
    result = pipeline(images)
    # result = pipeline(images, normalized=True)
    # result = pipeline(images, return_gpu=True)  # CUDA: CuPy arrays

    print(pipeline.device)  # selected "cpu" or "cuda:N"

landmarks = result["landmarks"]  # (N, 478, 3), in input order
```

Run your script in the selected environment with `uv run --no-sync python label.py`.
Load/decode images in your dataset code; convert BGR images to RGB before calling.

| Result | Shape | Contents |
|---|---|---|
| `boxes` | `(N, 4)` | Weighted detector box: xmin, ymin, xmax, ymax |
| `keypoints` | `(N, 6, 2)` | Right eye, left eye, nose tip, mouth center, right/left ear tragion (subject's sides) |
| `detection_scores` | `(N,)` | Highest detector confidence, retained after weighting |
| `landmarks` | `(N, 478, 3)` | FaceMesh x, y, z projected into the original image |
| `presence_scores` | `(N,)` | FaceMesh face-presence probability |
| `tongue_out` | `(N,)` | FaceMesh tongue-out probability |

All outputs are float32 NumPy arrays by default. Coordinates use original-image
pixels; `normalized=True` divides x and z by image width and y by image height.
Coordinates are not clipped to image bounds. z is relative depth, not distance
in physical units. `return_gpu=True` keeps results on CUDA as CuPy arrays.
Results remain valid after another call or after closing the pipeline.

Choose batch sizes that fit memory: a call processes the entire batch, retaining
original images and scratch buffers for reuse. Reuse a pipeline across batches;
the first call is slower, and changing CUDA batch size can trigger model retuning.
`num_threads=8` bounds CPU parallelism (default: up to eight threads). Use one
pipeline per calling thread, and close it with the context manager or `close()`.

`ensure_models(model_dir)` exports missing models in a subprocess, with CUDA
disabled only in that subprocess. TensorFlow is never imported by the runtime.
Cached models are checked for compatibility when the pipeline opens them. The
export extra installs TensorFlow, tf2onnx, ONNX, and ONNXScript; it is unnecessary
when supplying existing models directly.

`device="auto"` checks CUDA availability and reports a CPU fallback when no usable
GPU is found. Explicit CUDA requests fail when CUDA cannot initialize. Model and
inference errors propagate to the caller. One pipeline handles one GPU.

## Upstream notices

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for MediaPipe model and
processing attribution and the included Apache-2.0 license text.
