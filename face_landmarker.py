"""Batched, single-face MediaPipe inference using ONNX Runtime."""

from __future__ import annotations

import atexit
import math
import os
import subprocess
import sys
import warnings
import weakref
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
from pathlib import Path

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

_RESULT_WIDTH = 16 + 1 + 478 * 3 + 2


def ensure_models(model_dir: str | Path) -> tuple[Path, Path]:
    """Export missing models in a child process; requires the export extra.

    TensorFlow and its CPU-only export settings never enter the calling process.
    Existing models are checked for compatibility when FaceLandmarker loads them.
    """
    model_dir = Path(model_dir).resolve()
    paths = (model_dir / "blazeface.onnx", model_dir / "facemesh_v2.onnx")
    if not all(path.is_file() for path in paths):
        subprocess.run(
            [sys.executable, "-m", "face_landmarker_export", "--output-dir", str(model_dir)],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, check=True,
        )
    return paths


def _auto_device() -> str:
    import onnxruntime as ort

    reason = "CUDAExecutionProvider is not installed"
    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            import cupy as cp

            ort.preload_dlls(directory="")
            if cp.cuda.runtime.getDeviceCount():
                with cp.cuda.Device(0):
                    cp.empty(1)  # Check that the driver can actually initialize.
                return "cuda:0"
            reason = "no visible CUDA devices"
        except (ImportError, OSError, RuntimeError) as error:
            reason = str(error)
    warnings.warn(f"Using CPU: {reason}", RuntimeWarning, stacklevel=3)
    return "cpu"


@njit(cache=True, parallel=True)
def _warp_cpu(pixels, metadata, transforms, size, replicate, output):
    scale = np.float32(1 / 255 if replicate else 1 / 127.5)
    bias = np.float32(0 if replicate else -1)
    for row in prange(len(metadata) * size):
        n, y = row // size, row % size
        offset, width, height = metadata[n]
        a, b, tx, c, d, ty = transforms[n, :6]
        for x in range(size):
            sx = a * np.float32(x) + b * np.float32(y) + tx
            sy = c * np.float32(x) + d * np.float32(y) + ty
            if replicate:
                sx = min(max(sx, np.float32(0)), np.float32(width - 1))
                sy = min(max(sy, np.float32(0)), np.float32(height - 1))
            x0, y0 = math.floor(sx), math.floor(sy)
            fx, fy = sx - np.float32(x0), sy - np.float32(y0)
            p00 = offset + (y0 * width + x0) * 3 if 0 <= x0 < width and 0 <= y0 < height else -1
            p10 = offset + (y0 * width + x0 + 1) * 3 if 0 <= x0 + 1 < width and 0 <= y0 < height else -1
            p01 = offset + ((y0 + 1) * width + x0) * 3 if 0 <= x0 < width and 0 <= y0 + 1 < height else -1
            p11 = offset + ((y0 + 1) * width + x0 + 1) * 3 if 0 <= x0 + 1 < width and 0 <= y0 + 1 < height else -1
            for channel in range(3):
                v00 = np.float32(pixels[p00 + channel]) if p00 >= 0 else np.float32(0)
                v10 = np.float32(pixels[p10 + channel]) if p10 >= 0 else np.float32(0)
                v01 = np.float32(pixels[p01 + channel]) if p01 >= 0 else np.float32(0)
                v11 = np.float32(pixels[p11 + channel]) if p11 >= 0 else np.float32(0)
                top = v00 + (v10 - v00) * fx
                bottom = v01 + (v11 - v01) * fx
                output[n, channel, y, x] = (top + (bottom - top) * fy) * scale + bias


@njit(cache=True)
def _roi_cpu(detections, metadata, transforms):
    # MediaPipe image-mode ROI: expand both dimensions, without square_long.
    # The transform maps crop pixel coordinates directly into source pixels.
    for n in range(len(metadata)):
        width, height = np.float32(metadata[n, 1]), np.float32(metadata[n, 2])
        side = max(width, height)
        box = detections[n]
        cx = (box[0] + box[2]) * np.float32(0.5) * side + (width - side) * np.float32(0.5)
        cy = (box[1] + box[3]) * np.float32(0.5) * side + (height - side) * np.float32(0.5)
        rw = max((box[2] - box[0]) * side, np.float32(1)) * np.float32(1.5)
        rh = max((box[3] - box[1]) * side, np.float32(1)) * np.float32(1.5)
        angle = np.arctan2(box[7] - box[5], box[6] - box[4])
        co, si = np.cos(angle), np.sin(angle)
        a, b = co * rw / np.float32(256), -si * rh / np.float32(256)
        c, d = si * rw / np.float32(256), co * rh / np.float32(256)
        transforms[n, 0], transforms[n, 1] = a, b
        transforms[n, 2] = cx - np.float32(128) * (a + b)
        transforms[n, 3], transforms[n, 4] = c, d
        transforms[n, 5] = cy - np.float32(128) * (c + d)
        # MediaPipe scales z by the projected normalized ROI x-axis length.
        transforms[n, 6] = np.sqrt((a / width) ** 2 + (c / height) ** 2) * width


@njit(cache=True, parallel=True)
def _project_cpu(detections, scores, landmarks, presence, tongue,
                 metadata, transforms, normalized, output):
    for n in prange(len(metadata)):
        width, height = np.float32(metadata[n, 1]), np.float32(metadata[n, 2])
        side = max(width, height)
        xscale = np.float32(1) / width if normalized else np.float32(1)
        yscale = np.float32(1) / height if normalized else np.float32(1)
        for j in range(16):
            dimension = width if j % 2 == 0 else height
            scale = xscale if j % 2 == 0 else yscale
            output[n, j] = (detections[n, j] * side + (dimension - side) * np.float32(0.5)) * scale
        output[n, 16] = scores[n]
        for k in range(478):
            x, y, z = landmarks[n, k]
            a, b, tx, c, d, ty, zs = transforms[n]
            output[n, 17 + k * 3] = (a * x + b * y + tx) * xscale
            output[n, 18 + k * 3] = (c * x + d * y + ty) * yscale
            output[n, 19 + k * 3] = z * zs * xscale
        output[n, 1451] = presence[n]
        output[n, 1452] = tongue[n]


_CUDA_SOURCE = r'''
extern "C" __global__ void warp(
    const unsigned char* pixels, const long long* metadata, const float* tr,
    int count, int size, int replicate, float* output) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long plane = size * size;
    if (i >= count * plane) return;
    int n = i / plane, p = i % plane, x = p % size, y = p / size;
    long long offset = metadata[n*3], width = metadata[n*3+1], height = metadata[n*3+2];
    const float* t = tr + n*7;
    float sx = t[0]*x + t[1]*y + t[2], sy = t[3]*x + t[4]*y + t[5];
    if (replicate) {
        sx = fminf(fmaxf(sx, 0.f), width-1.f);
        sy = fminf(fmaxf(sy, 0.f), height-1.f);
    }
    long long x0 = (long long)floorf(sx), y0 = (long long)floorf(sy);
    float fx = sx-x0, fy = sy-y0;
    long long p00 = x0>=0 && x0<width && y0>=0 && y0<height ? offset+(y0*width+x0)*3 : -1;
    long long p10 = x0+1>=0 && x0+1<width && y0>=0 && y0<height ? offset+(y0*width+x0+1)*3 : -1;
    long long p01 = x0>=0 && x0<width && y0+1>=0 && y0+1<height ? offset+((y0+1)*width+x0)*3 : -1;
    long long p11 = x0+1>=0 && x0+1<width && y0+1>=0 && y0+1<height ? offset+((y0+1)*width+x0+1)*3 : -1;
    for (int c=0; c<3; ++c) {
        float v00 = p00<0 ? 0.f : pixels[p00+c], v10 = p10<0 ? 0.f : pixels[p10+c];
        float v01 = p01<0 ? 0.f : pixels[p01+c], v11 = p11<0 ? 0.f : pixels[p11+c];
        float top = v00+(v10-v00)*fx, bottom = v01+(v11-v01)*fx;
        output[(n*3+c)*plane+p] = (top+(bottom-top)*fy)*(replicate ? 1.f/255.f : 1.f/127.5f)
                                + (replicate ? 0.f : -1.f);
    }
}

extern "C" __global__ void roi(
    const float* detections, const long long* metadata, int count, float* tr) {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= count) return;
    float width = metadata[n*3+1], height = metadata[n*3+2], side = fmaxf(width, height);
    const float* box = detections + n*16;
    float cx = (box[0]+box[2])*0.5f*side+(width-side)*0.5f;
    float cy = (box[1]+box[3])*0.5f*side+(height-side)*0.5f;
    float rw = fmaxf((box[2]-box[0])*side, 1.f)*1.5f;
    float rh = fmaxf((box[3]-box[1])*side, 1.f)*1.5f;
    float angle = atan2f(box[7]-box[5], box[6]-box[4]);
    float co = cosf(angle), si = sinf(angle);
    float a = co*rw/256.f, b = -si*rh/256.f, c = si*rw/256.f, d = co*rh/256.f;
    float* t = tr + n*7;
    t[0]=a; t[1]=b; t[2]=cx-128.f*(a+b);
    t[3]=c; t[4]=d; t[5]=cy-128.f*(c+d);
    t[6]=sqrtf((a/width)*(a/width)+(c/height)*(c/height))*width;
}

extern "C" __global__ void project(
    const float* detections, const float* scores, const float* landmarks,
    const float* presence, const float* tongue, const long long* metadata,
    const float* tr, int count, int normalized, float* output) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count*478) return;
    int n = i/478, k = i%478;
    float width=metadata[n*3+1], height=metadata[n*3+2], side=fmaxf(width,height);
    float xs=normalized ? 1.f/width : 1.f, ys=normalized ? 1.f/height : 1.f;
    const float* t=tr+n*7;
    float x=landmarks[i*3], y=landmarks[i*3+1], z=landmarks[i*3+2];
    float* out=output+n*1453;
    out[17+k*3]=(t[0]*x+t[1]*y+t[2])*xs;
    out[18+k*3]=(t[3]*x+t[4]*y+t[5])*ys;
    out[19+k*3]=z*t[6]*xs;
    if (k<16) {
        float dim=k%2 ? height : width, scale=k%2 ? ys : xs;
        out[k]=(detections[n*16+k]*side+(dim-side)*0.5f)*scale;
    }
    if (k==0) { out[16]=scores[n]; out[1451]=presence[n]; out[1452]=tongue[n]; }
}
'''


def _result_views(packed):
    n = len(packed)
    return {
        "boxes": packed[:, :4],
        "keypoints": packed[:, 4:16].reshape(n, 6, 2),
        "detection_scores": packed[:, 16],
        "landmarks": packed[:, 17:1451].reshape(n, 478, 3),
        "presence_scores": packed[:, 1451],
        "tongue_out": packed[:, 1452],
    }


def _close_at_exit(reference):
    instance = reference()
    if instance is not None:
        instance.close()


def _copy_images(work):
    host, images, metadata = work
    for image, (start, width, height) in zip(images, metadata):
        np.copyto(host[start:start + image.size].reshape(height, width, 3), image)


class FaceLandmarker:
    """One result per RGB uint8 image. Use one instance per calling thread.

    Pixel coordinates are the default. ``normalized=True`` divides x/z by
    image width and y by image height. ``return_gpu=True`` returns CuPy arrays
    in CUDA mode. Returned arrays stay valid across subsequent calls.
    """

    def __init__(self, detector_path: str | Path, landmarks_path: str | Path,
                 device: str = "cpu", *, num_threads: int | None = None):
        self._closed = False
        self._sessions = []
        self._buffers = {}
        self._copy_pool = None
        try:
            self._initialize(detector_path, landmarks_path, device, num_threads)
        except BaseException:
            self.close()
            raise

    def _initialize(self, detector_path, landmarks_path, device, num_threads):
        import onnxruntime as ort

        automatic = device == "auto"
        if automatic:
            device = _auto_device()
        if device == "cpu":
            self._cuda = False
            self._device_id = 0
        elif device == "cuda" or (device.startswith("cuda:") and device[5:].isdigit()):
            self._cuda = True
            self._device_id = int(device.partition(":")[2] or 0)
        else:
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")
        self.device = f"cuda:{self._device_id}" if self._cuda else "cpu"
        if num_threads is not None and (not isinstance(num_threads, int) or num_threads < 1):
            raise ValueError("num_threads must be a positive integer")
        self._threads = min(num_threads or 8, get_num_threads())
        self._copy_pool = ThreadPoolExecutor(max_workers=self._threads)
        self._host_pixels = np.empty(0, np.uint8)
        self._xp = np
        self._ort = ort
        self._run_options = ort.RunOptions()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1 if self._cuda else self._threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0" if self._cuda else "1")
        if self._cuda:
            try:
                import cupy as cp
            except ImportError as error:
                raise RuntimeError("CUDA mode requires: uv sync --extra cuda") from error
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDAExecutionProvider is unavailable; use uv sync --extra cuda")
            ort.preload_dlls(directory="")
            self._xp = cp
            self._device = cp.cuda.Device(self._device_id)
            with self._device:
                self._stream = cp.cuda.Stream(non_blocking=True)
                self._kernels = cp.RawModule(code=_CUDA_SOURCE, options=("--fmad=false",))
                self._warp_kernel = self._kernels.get_function("warp")
                self._roi_kernel = self._kernels.get_function("roi")
                self._project_kernel = self._kernels.get_function("project")
            providers = [("CUDAExecutionProvider", {
                "device_id": self._device_id,
                "user_compute_stream": str(self._stream.ptr),
                "do_copy_in_default_stream": "1",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                # Avoid TF32 changing confidence/ROI selection with batch size.
                "use_tf32": "0",
            }), "CPUExecutionProvider"]
            # Keep ORT's run-end cleanup enabled: disabling provider sync in
            # ORT 1.23 also defers freeing pinned scratch buffers indefinitely.
        else:
            providers = ["CPUExecutionProvider"]
        for path in (detector_path, landmarks_path):
            self._sessions.append(ort.InferenceSession(str(path), options, providers=providers))
        if self._cuda and any("CUDAExecutionProvider" not in s.get_providers() for s in self._sessions):
            if not automatic:
                raise RuntimeError("ONNX Runtime could not initialize CUDAExecutionProvider")
            warnings.warn("Using CPU: ONNX Runtime could not initialize CUDAExecutionProvider",
                          RuntimeWarning, stacklevel=3)
            self.close()
            self.__init__(detector_path, landmarks_path, "cpu", num_threads=num_threads)
            return
        # Register after importing CuPy, so sessions are released before its
        # CUDA runtime cleanup. The weak reference does not retain instances.
        self._exit_callback = partial(_close_at_exit, weakref.ref(self))
        atexit.register(self._exit_callback)
        self._output_specs = []
        for session, kind, size, expected in zip(
            self._sessions, ("detector", "landmarks"), (128, 256),
            ({"detections": [16], "detection_scores": []},
             {"landmarks": [478, 3], "presence_scores": [], "tongue_out": []}),
        ):
            session.disable_fallback()
            if self._cuda and "CUDAExecutionProvider" not in session.get_providers():
                raise RuntimeError("ONNX Runtime could not initialize CUDAExecutionProvider")
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            metadata = session.get_modelmeta().custom_metadata_map
            if (metadata.get("format") != "face-landmark-onnx/1"
                    or metadata.get("kind") != kind
                    or len(inputs) != 1 or inputs[0].name != "images"
                    or inputs[0].shape[1:] != [3, size, size]
                    or isinstance(inputs[0].shape[0], int)
                    or inputs[0].type != "tensor(float)"
                    or {out.name: out.shape[1:] for out in outputs} != expected
                    or any(out.type != "tensor(float)" or isinstance(out.shape[0], int)
                           for out in outputs)):
                raise ValueError(f"Incompatible {kind} model; run face-landmarker-export")
            by_name = {out.name: out for out in outputs}
            self._output_specs.append([by_name[name] for name in expected])

    def close(self):
        """Release sessions and scratch storage. Returned results remain valid."""
        if getattr(self, "_closed", True):
            return
        if getattr(self, "_cuda", False) and hasattr(self, "_stream"):
            with self._device:
                self._stream.synchronize()
                self._sessions.clear()
                self._buffers.clear()
                self._host_pixels = None
                self._host_metadata = None
        else:
            self._sessions.clear()
            self._buffers.clear()
            self._host_pixels = None
        self._closed = True
        if self._copy_pool is not None:
            self._copy_pool.shutdown(wait=True)
        if hasattr(self, "_exit_callback"):
            atexit.unregister(self._exit_callback)

    def __enter__(self):
        if self._closed:
            raise RuntimeError("FaceLandmarker has been closed")
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        with suppress(Exception):
            self.close()

    def _buffer(self, name, shape, dtype=np.float32):
        count = math.prod(shape)
        buffer = self._buffers.get(name)
        if buffer is None or buffer.size < count:
            buffer = self._buffers[name] = self._xp.empty(count, dtype=dtype)
        return buffer[:count].reshape(shape)

    def _prepare(self, images):
        n = len(images)
        metadata = np.empty((n, 3), np.int64)
        offset = 0
        for i, image in enumerate(images):
            height, width = image.shape[:2]
            metadata[i] = offset, width, height
            offset += image.size
        if self._host_pixels.size < offset:
            if self._cuda:
                # The ndarray owns the pinned allocation through its base.
                memory = self._xp.cuda.alloc_pinned_memory(offset)
                self._host_pixels = np.frombuffer(memory, np.uint8, count=offset)
            else:
                self._host_pixels = np.empty(offset, np.uint8)
        host = self._host_pixels[:offset]
        # Native copies release the GIL. Large batches benefit from parallel
        # host memory writes; small batches avoid executor overhead.
        if offset >= 32 * 1024**2 and n >= 2 and self._threads > 1:
            step = (n + self._threads - 1) // self._threads
            work = [(host, images[i:i+step], metadata[i:i+step]) for i in range(0, n, step)]
            list(self._copy_pool.map(_copy_images, work))
        else:
            _copy_images((host, images, metadata))
        transforms = np.zeros((n, 7), np.float32)
        widths, heights = metadata[:, 1].astype(np.float32), metadata[:, 2].astype(np.float32)
        sides = np.maximum(widths, heights)
        transforms[:, 0] = transforms[:, 4] = sides / np.float32(128)
        transforms[:, 2] = (widths - sides) * np.float32(0.5)
        transforms[:, 5] = (heights - sides) * np.float32(0.5)
        if self._cuda:
            pixels = self._buffer("pixels", (offset,), np.uint8)
            device_metadata = self._buffer("metadata", (n, 3), np.int64)
            device_transforms = self._buffer("transforms", (n, 7))
            pixels.set(host, stream=self._stream)
            # Keep host metadata alive until the stream has completed.
            self._host_metadata = metadata, transforms
            device_metadata.set(metadata, stream=self._stream)
            device_transforms.set(transforms, stream=self._stream)
            return pixels, device_metadata, device_transforms
        return host, metadata, transforms

    def _warp(self, pixels, metadata, transforms, size):
        n = len(metadata)
        output = self._buffer(f"input_{size}", (n, 3, size, size))
        if self._cuda:
            self._warp_kernel(((n * size * size + 255) // 256,), (256,), (
                pixels, metadata, transforms, np.int32(n), np.int32(size),
                np.int32(size == 256), output,
            ))
        else:
            _warp_cpu(pixels, metadata, transforms, size, size == 256, output)
        return output

    def _run(self, index, images):
        session = self._sessions[index]
        binding = session.io_binding()
        device_type = "cuda" if self._cuda else "cpu"
        pointer = images.data.ptr if self._cuda else images.ctypes.data
        binding.bind_input("images", device_type, self._device_id, np.float32, images.shape, pointer)
        outputs = []
        for spec in self._output_specs[index]:
            output = self._buffer(spec.name, (len(images), *spec.shape[1:]))
            pointer = output.data.ptr if self._cuda else output.ctypes.data
            binding.bind_output(spec.name, device_type, self._device_id, np.float32, output.shape, pointer)
            outputs.append(output)
        session.run_with_iobinding(binding, self._run_options)
        return outputs

    def _roi(self, detections, metadata, transforms):
        if self._cuda:
            self._roi_kernel(((len(metadata) + 127) // 128,), (128,), (
                detections, metadata, np.int32(len(metadata)), transforms,
            ))
        else:
            _roi_cpu(detections, metadata, transforms)

    def _project(self, detections, scores, landmarks, presence, tongue,
                 metadata, transforms, normalized):
        n = len(metadata)
        # Final results own their allocation, unlike reusable inference buffers.
        output = self._xp.empty((n, _RESULT_WIDTH), np.float32)
        if self._cuda:
            self._project_kernel(((n * 478 + 255) // 256,), (256,), (
                detections, scores, landmarks, presence, tongue, metadata, transforms,
                np.int32(n), np.int32(normalized), output,
            ))
        else:
            _project_cpu(detections, scores, landmarks, presence, tongue,
                         metadata, transforms, normalized, output)
        return output

    def _infer(self, images, normalized):
        if not images:
            return self._xp.empty((0, _RESULT_WIDTH), np.float32)
        pixels, metadata, transforms = self._prepare(images)
        detections, scores = self._run(0, self._warp(pixels, metadata, transforms, 128))
        self._roi(detections, metadata, transforms)
        landmarks, presence, tongue = self._run(1, self._warp(pixels, metadata, transforms, 256))
        return self._project(detections, scores, landmarks, presence, tongue,
                             metadata, transforms, normalized)

    def __call__(self, images: Sequence[np.ndarray] | np.ndarray, *,
                 normalized: bool = False, return_gpu: bool = False) -> dict:
        if self._closed:
            raise RuntimeError("FaceLandmarker has been closed")
        if return_gpu and not self._cuda:
            raise ValueError("return_gpu=True requires CUDA mode")
        if isinstance(images, np.ndarray) and images.ndim != 4:
            raise ValueError("Pass a list of HWC RGB images or an NHWC batch")
        images = list(images)
        for i, image in enumerate(images):
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.ndim != 3 or image.shape[2] != 3
                    or image.shape[0] == 0 or image.shape[1] == 0):
                raise ValueError(f"Image {i} must be a nonempty uint8 HWC RGB NumPy array")
        if self._cuda:
            with self._device, self._stream:
                try:
                    result = self._infer(images, normalized)
                    if not return_gpu:
                        result = result.get(stream=self._stream, blocking=False)
                finally:
                    # Finish projection/download, also on failure before pinned
                    # host buffers may be reused by another call.
                    self._stream.synchronize()
        else:
            previous_threads = get_num_threads()
            set_num_threads(self._threads)
            try:
                result = self._infer(images, normalized)
            finally:
                set_num_threads(previous_threads)
        return _result_views(result)
