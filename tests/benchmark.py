"""uv run --extra cuda --group dev python tests/benchmark.py --image IMAGE.jpg"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from face_landmarker import FaceLandmarker


def profile_stages(pipeline, batch):
    measurements = []
    originals = {}

    def wrap(name, original):
        def measured(*args):
            label = name.removeprefix("_")
            if name == "_run":
                label = "detector" if args[0] == 0 else "landmarks"
            if name == "_warp":
                label += f"_{args[-1]}"
            if pipeline._cuda:
                start, end = pipeline._xp.cuda.Event(), pipeline._xp.cuda.Event()
                start.record(pipeline._stream)
                result = original(*args)
                end.record(pipeline._stream)
                measurements.append((label, start, end))
            else:
                start = time.perf_counter()
                result = original(*args)
                measurements.append((label, start, time.perf_counter()))
            return result
        return measured

    for name in ("_prepare", "_warp", "_run", "_roi", "_project"):
        originals[name] = getattr(pipeline, name)
        setattr(pipeline, name, wrap(name, originals[name]))
    try:
        start = time.perf_counter()
        pipeline(batch)
        total = (time.perf_counter() - start) * 1000
    finally:
        for name, original in originals.items():
            setattr(pipeline, name, original)
    stages = defaultdict(float)
    for label, start, end in measurements:
        stages[label] += (pipeline._xp.cuda.get_elapsed_time(start, end) if pipeline._cuda
                          else (end - start) * 1000)
    stages["return_and_overhead"] = max(0, total - sum(stages.values()))
    return dict(stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path(".cache/portrait.jpg"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 128])
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    portrait = Image.open(args.image).convert("RGB")
    images = [np.array(portrait.resize(size))
              for size in [(256, 384), (640, 480), (768, 1024), (513, 337)]]
    for device in args.devices:
        start = time.perf_counter()
        with FaceLandmarker(args.models / "blazeface.onnx", args.models / "facemesh_v2.onnx", device) as pipeline:
            init_ms = (time.perf_counter() - start) * 1000
            for n in args.batches:
                batch = [images[i % len(images)].copy() for i in range(n)]
                start = time.perf_counter()
                pipeline(batch)
                first_ms = (time.perf_counter() - start) * 1000
                times = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    pipeline(batch)
                    times.append((time.perf_counter() - start) * 1000)
                elapsed = statistics.median(times)
                record = {
                    "device": device, "batch": n, "init_ms": round(init_ms, 2),
                    "first_call_ms": round(first_ms, 2), "median_ms": round(elapsed, 2),
                    "images_per_second": round(n * 1000 / elapsed),
                    "scratch_MiB": round(sum(x.nbytes for x in pipeline._buffers.values()) / 2**20, 2),
                    "image_staging_MiB": round(pipeline._host_pixels.nbytes / 2**20, 2),
                    "stages_ms": {k: round(v, 3) for k, v in profile_stages(pipeline, batch).items()},
                }
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
