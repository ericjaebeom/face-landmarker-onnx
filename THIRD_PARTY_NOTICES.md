# Third-party notices

## MediaPipe models and processing

This project downloads Google's MediaPipe Face Landmarker v1 bundle and converts
its BlazeFace and FaceMesh-V2 models from TFLite to ONNX. It adds dynamic batching
and ONNX detector decoding, weighted NMS, and landmark output processing. The
runtime implements MediaPipe-derived anchor, image transform, face ROI, and
landmark projection algorithms for CPU and CUDA.

Upstream: [MediaPipe](https://github.com/google-ai-edge/mediapipe), by Google and
the MediaPipe Authors. Upstream copyright notices remain applicable to derived
material. MediaPipe and these models are licensed under the Apache License,
Version 2.0; the full license is included in
[licenses/MediaPipe-Apache-2.0.txt](licenses/MediaPipe-Apache-2.0.txt).

Sources:

- [Face Landmarker model bundle](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
- [BlazeFace model card](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20%28Short%20Range%29.pdf)
- [FaceMesh-V2 model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)
- [MediaPipe calculators](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/calculators)
- [MediaPipe face landmarker graphs](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/tasks/cc/vision/face_landmarker)

Model binaries are downloaded at export time and are not included in this
repository or its Python distribution. Keep the upstream notices and license
with redistributed converted models.

These notices apply to upstream material; they do not assign a license to this
repository's original code. Python dependencies retain their respective licenses.
