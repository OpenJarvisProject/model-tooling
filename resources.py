"""Resource use of a GGUF in the real runtime (ojp-core's smoke test:
load + 32-token completion), CPU-only and fully on the GPU.

Per run: wall time, peak RSS and CPU time of that one process (os.wait4,
so runs don't contaminate each other), and on the GPU the peak VRAM
above the pre-run baseline, sampled every 20 ms from amdgpu's
mem_info_vram_used. The desktop shares the GPU, so VRAM deltas carry
some noise from other apps; repeats show how much.
"""

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

CORE = Path("/media/openjarvisproject-core")
VRAM = next(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used"))
ENV = {**os.environ, "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/radeon_icd.json"}


def vram_mb():
    return int(VRAM.read_text()) / 1e6


def run_once(model, gpu_layers):
    baseline = vram_mb()
    peak = [baseline]
    done = threading.Event()

    def sample():
        while not done.is_set():
            peak[0] = max(peak[0], vram_mb())
            time.sleep(0.02)

    sampler = threading.Thread(target=sample)
    sampler.start()
    start = time.monotonic()
    env = {**ENV, "OJP_GPU_LAYERS": str(gpu_layers)}
    if gpu_layers == 0:
        # With a Vulkan device present, llama.cpp still offloads big
        # matmuls to it at n_gpu_layers=0 (measured: 400-600MB VRAM).
        # Hiding the ICD gives a true CPU-only run, like a GPU-less device.
        env["VK_ICD_FILENAMES"] = "/nonexistent.json"
    proc = subprocess.Popen([str(CORE / "target/release/ojp-core"), str(model)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, status, usage = os.wait4(proc.pid, 0)
    wall = time.monotonic() - start
    done.set()
    sampler.join()
    if status != 0:
        raise RuntimeError(f"ojp-core failed on {model}")
    cpu = usage.ru_utime + usage.ru_stime
    return {"wall_s": wall, "peak_rss_mb": usage.ru_maxrss / 1024, "avg_cores": cpu / wall,
            "vram_mb": peak[0] - baseline}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="label=path.gguf")
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    for label, path in (m.split("=", 1) for m in args.models):
        for device, layers in (("cpu", 0), ("gpu", 99)):
            runs = [run_once(path, layers) for _ in range(args.reps)]
            row = {"model": label, "device": device, "file_mb": round(Path(path).stat().st_size / 1e6, 1)}
            for key in runs[0]:
                vals = [r[key] for r in runs]
                row[key] = round(statistics.mean(vals), 2)
                row[key + "_sd"] = round(statistics.stdev(vals), 2)
            print("RESULT", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
