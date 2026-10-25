"""Unattended, resumable Qwen2.5-7B-Instruct study: which quantizations fit
fully on the RX 580 (8 GB) and what they cost in quality. Every
measurement runs with ALL layers on the GPU (-ngl 99); a variant that
doesn't fit is recorded as "does not fit", which is itself the answer.

Run repeatedly (the systemd timer does): each finished step is saved as
outputs/7b-study/steps/<step>.json and skipped next time, so a crash,
reboot or power cut loses at most the step in progress. A failed step is
retried on later runs, up to MAX_ATTEMPTS. Progress: outputs/7b-study/STATUS.md.
Raw tool output: outputs/7b-study/logs/.

Design, fixed before any results (one variable at a time):
  - Same harness as quant_study.py / close_compare.py: wikitext-2 test
    perplexity (64 x 512), imatrix calibrated on wikitext-2 train.
  - The imatrix is computed from Q5_K_M, not Q8_0, because Q8_0 7B
    (~8.1 GB) can't be fully on an 8 GB card, and the rule is GPU only.
    Every imatrix variant uses this same imatrix.
  - Finalists for the deep comparison are fixed now: Q5_K_M+imat (the
    reference, i.e. the largest expected to fit), Q4_K_M+imat,
    IQ4_XS+imat, IQ3_M+imat. KL divergence is measured vs Q5_K_M+imat,
    not vs f16, because f16 can't run on the GPU.
"""
import os as _os, sys as _sys
# DISABLED 2026-10-12: 7B runs crashed the GPU/system twice (hard reboots).
# Never run automatically; only on the user's explicit request.
if _os.environ.get("OJP_USER_APPROVED_7B") != "yes":
    _sys.exit("7B tests are disabled (see RESULTS.md); refusing to run.")


import fcntl
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).parent
STUDY = HERE / "outputs" / "7b-study"
STEPS, LOGS, MODELS = STUDY / "steps", STUDY / "logs", STUDY / "models"
BIN = Path("/media/openjarvisproject-model-tooling/llama.cpp/build-vk/bin")
CONVERT = Path("/media/openjarvisproject-model-tooling/llama.cpp/convert_hf_to_gguf.py")
PY = HERE / ".venv-cpu" / "bin" / "python"
SNAPSHOT = next((HERE / ".hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots").iterdir())
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/radeon_icd.json"
os.environ["OJP_N_CTX"] = "4096"  # VRAM check: does it fit with a 4K context?
MAX_ATTEMPTS = 2

F16 = MODELS / "qwen2.5-7b-f16.gguf"
IMAT_SRC = MODELS / "imatrix-src-Q5_K_M.gguf"
IMATRIX = MODELS / "imatrix.gguf"
# (label, llama-quantize type, use imatrix)
VARIANTS = [
    ("Q6_K", "Q6_K", False),
    ("Q5_K_M+imat", "Q5_K_M", True),
    ("Q4_K_M", "Q4_K_M", False),
    ("Q4_K_M+imat", "Q4_K_M", True),
    ("IQ4_XS+imat", "IQ4_XS", True),
    ("Q3_K_M+imat", "Q3_K_M", True),
    ("IQ3_M+imat", "IQ3_M", True),
    ("IQ3_XXS+imat", "IQ3_XXS", True),
    ("IQ2_M+imat", "IQ2_M", True),
]
FINALISTS = ["Q5_K_M+imat", "Q4_K_M+imat", "IQ4_XS+imat", "IQ3_M+imat"]
OOM = re.compile(r"out of (device )?memory|ErrorOutOfDeviceMemory|failed to allocate|unable to allocate", re.I)


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(STUDY / "study.log", "a") as f:
        f.write(line + "\n")


def run(name, cmd, timeout=None):
    """Run a command, keeping its full output in logs/<name>.log."""
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    out = proc.stdout + proc.stderr
    (LOGS / f"{name}.log").write_text(out)
    if proc.returncode != 0:
        kind = "does_not_fit" if OOM.search(out) else "failed"
        raise StepError(kind, out[-1500:])
    return out


class StepError(Exception):
    def __init__(self, kind, detail):
        super().__init__(f"{kind}: {detail[-300:]}")
        self.kind, self.detail = kind, detail


COOLDOWN = 60


def gpu_ready():
    """Strictly one GPU job at a time: refuse to start if the GPU looks lost or still busy."""
    out = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True).stdout
    if "RADV" not in out:
        log("GPU not visible; stopping so it is not hammered")
        sys.exit(1)
    time.sleep(COOLDOWN)  # let the card cool/settle between jobs


def step(name, fn):
    """Run fn once ever (result saved); retry failures on later runs."""
    done = STEPS / f"{name}.json"
    if done.exists():
        return json.loads(done.read_text())
    fail = STEPS / f"{name}.failed.json"
    attempts = json.loads(fail.read_text())["attempts"] if fail.exists() else 0
    if attempts >= MAX_ATTEMPTS:
        return None
    gpu_ready()
    log(f"START {name} (attempt {attempts + 1})")
    start = time.time()
    try:
        result = fn()
    except StepError as e:
        if e.kind == "does_not_fit":
            # A definitive answer, not a failure: record it and move on.
            result = {"does_not_fit": True, "detail": e.detail[-600:]}
        else:
            fail.write_text(json.dumps({"attempts": attempts + 1, "error": str(e)}, indent=2))
            log(f"FAIL  {name}: {e}")
            return None
    except Exception as e:  # noqa: BLE001 - keep going with other steps
        fail.write_text(json.dumps({"attempts": attempts + 1, "error": traceback.format_exc()[-1500:]}, indent=2))
        log(f"FAIL  {name}: {e!r}")
        return None
    result["seconds"] = round(time.time() - start)
    done.write_text(json.dumps(result, indent=2))
    fail.unlink(missing_ok=True)
    log(f"DONE  {name}: {json.dumps(result)[:300]}")
    write_status()
    return result


def gguf(label):
    return MODELS / f"{label.replace('+', '-')}.gguf"


def quantize(label, qtype, imat):
    out = gguf(label)
    if not out.exists():
        cmd = [BIN / "llama-quantize"] + (["--imatrix", IMATRIX] if imat else []) + [F16, out, qtype]
        run(f"quantize-{label}", cmd)
    return out


def perplexity(label, model):
    out = run(f"ppl-{label}", [BIN / "llama-perplexity", "-m", model, "-f", HERE / "wikitext2-test.txt",
                               "-c", "512", "--chunks", "64", "-ngl", "99"])
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", out)
    if not m:
        raise StepError("failed", out[-1500:])
    return float(m.group(1)), float(m.group(2))


def vram_at_4k(model):
    sys.path.insert(0, str(HERE))
    import resources  # noqa: E402 - needs OJP_N_CTX set first

    try:
        r = resources.run_once(model, 99)
    except RuntimeError as e:
        raise StepError("does_not_fit", str(e)) from e
    return {"vram_mb_ctx4096": round(r["vram_mb"]), "peak_rss_mb": round(r["peak_rss_mb"])}


def sweep(label, qtype, imat):
    def fn():
        model = quantize(label, qtype, imat)
        ppl, err = perplexity(label, model)
        row = {"variant": label, "mb": round(model.stat().st_size / 1e6), "ppl": ppl, "ppl_err": err}
        row.update(vram_at_4k(model))
        if label not in FINALISTS:
            model.unlink()
        return row

    return step(f"sweep-{label}", fn)


def deep(label):
    sys.path.insert(0, str(HERE))
    import close_compare as cc  # noqa: E402

    model = gguf(label)
    ref = gguf(FINALISTS[0])
    base = STUDY / "base-Q5_K_M.kld"
    metrics = {
        "hellaswag": lambda: cc.hellaswag(model),
        "winogrande": lambda: cc.winogrande(model),
        "arc": lambda: cc.multiple_choice(model, "arc-combined-validation.bin"),
        "mmlu": lambda: cc.multiple_choice(model, "mmlu-validation.bin"),
        "truthfulqa": lambda: cc.multiple_choice(model, "truthful-qa-validation.bin"),
        "agent": lambda: cc.agent(model),
        "speed_gpu": lambda: cc.speed(model, ngl="99"),
    }
    if label != FINALISTS[0]:
        metrics = {"kld_vs_Q5_K_M": lambda: cc.kl_divergence(model, base), **metrics}
    for metric, fn in metrics.items():
        if not model.exists():
            return
        if metric == "kld_vs_Q5_K_M" and not base.exists():
            step("kld-base", lambda: (run("kld-base", [BIN / "llama-perplexity", "-m", ref, "-ngl", "99", "-f",
                                                       HERE / "wikitext2-test.txt", "-c", "512", "--chunks", "20",
                                                       "--kl-divergence-base", base]), {"ok": True})[1])
        step(f"deep-{label}-{metric}", lambda fn=fn: wrap(fn))


def wrap(fn):
    try:
        result = fn()
    except RuntimeError as e:
        raise StepError("does_not_fit" if OOM.search(str(e)) else "failed", str(e)) from e
    return result if isinstance(result, dict) else {"value": result}


def write_status():
    rows = []
    for f in sorted(STEPS.glob("sweep-*.json")):
        if f.name.endswith(".failed.json"):
            continue
        r = json.loads(f.read_text())
        if r.get("does_not_fit"):
            rows.append(f"| {f.stem[6:]} | – | does not fit on the GPU | – |")
        else:
            rows.append(f"| {r['variant']} | {r['mb']} | {r['ppl']} ± {r['ppl_err']} | {r['vram_mb_ctx4096']} |")
    deep_rows = []
    for label in FINALISTS:
        parts = []
        for metric in ["kld_vs_Q5_K_M", "hellaswag", "winogrande", "arc", "mmlu", "truthfulqa", "agent", "speed_gpu"]:
            f = STEPS / f"deep-{label}-{metric}.json"
            if f.exists():
                r = json.loads(f.read_text())
                if r.get("does_not_fit"):
                    parts.append(f"{metric} doesn't fit")
                elif metric == "agent":
                    parts.append(f"agent {r['total']}")
                elif metric == "speed_gpu":
                    parts.append(f"gen {r.get('gpu_gen_tok_s')} tok/s")
                elif metric == "kld_vs_Q5_K_M":
                    parts.append(f"KLD {r['mean_kld']}")
                else:
                    parts.append(f"{metric} {r['acc']}")
        deep_rows.append(f"| {label} | {', '.join(parts) or 'pending'} |")
    failed = [f.stem.replace('.failed', '') for f in STEPS.glob("*.failed.json")]
    (STUDY / "STATUS.md").write_text(
        f"# 7B GPU study status ({time.strftime('%Y-%m-%d %H:%M')})\n\n"
        f"{'DONE' if (STUDY / 'DONE').exists() else 'in progress'}; failed steps (will retry): {failed or 'none'}\n\n"
        "## Sweep (all layers on GPU; VRAM = whole process at 4K context, desktop excluded)\n\n"
        "| Variant | MB | wikitext-2 ppl | VRAM MB @4K |\n|---|---|---|---|\n" + "\n".join(rows) +
        "\n\n## Deep comparison (finalists)\n\n| Variant | Results |\n|---|---|\n" + "\n".join(deep_rows) + "\n")


def main():
    for d in (STEPS, LOGS, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    lock = open(STUDY / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("already running")
        return
    if (STUDY / "DONE").exists():
        return
    log("run started")

    step("convert-f16", lambda: (F16.exists() or run("convert-f16", [PY, CONVERT, SNAPSHOT, "--outtype", "f16",
                                                                     "--outfile", F16]), {"mb": round(F16.stat().st_size / 1e6)})[1])
    if not F16.exists():
        return
    step("imatrix", lambda: (quantize_src(), run("imatrix", [BIN / "llama-imatrix", "-m", IMAT_SRC, "-f",
                                                            HERE / "wikitext2-train.txt", "-o", IMATRIX,
                                                            "--chunks", "100", "-c", "512", "-ngl", "99"]),
                              IMAT_SRC.unlink(), {"ok": True})[-1])
    if not IMATRIX.exists():
        return
    for label, qtype, imat in VARIANTS:
        sweep(label, qtype, imat)
    for label in FINALISTS:
        if gguf(label).exists():
            deep(label)

    pending = [f for f in STEPS.glob("*.failed.json") if json.loads(f.read_text())["attempts"] < MAX_ATTEMPTS]
    write_status()
    if not pending:
        (STUDY / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M"))
        write_status()
        log("study complete")
        subprocess.run(["systemctl", "--user", "disable", "--now", "ojp-7b-study.timer"])


def quantize_src():
    if not IMAT_SRC.exists():
        run("quantize-imatrix-src", [BIN / "llama-quantize", F16, IMAT_SRC, "Q5_K_M"])


if __name__ == "__main__":
    main()
