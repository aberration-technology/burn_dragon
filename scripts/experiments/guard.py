"""Fast Linux host-memory guard; GPU queries never block the watchdog thread."""

from dataclasses import asdict, dataclass
import csv
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from .config import Limits


@dataclass(frozen=True)
class Memory:
    total_mib: float
    available_mib: float

    @property
    def used_mib(self):
        return self.total_mib - self.available_mib

    @classmethod
    def read(cls, path=Path("/proc/meminfo")):
        values = {}
        for line in path.read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                values[key] = int(value.split()[0]) / 1024.0
        result = cls(values["MemTotal"], values["MemAvailable"])
        if result.total_mib <= 0 or not 0 <= result.available_mib <= result.total_mib:
            raise ValueError("invalid physical-memory reading")
        return result


def memory_violation(memory, limits, reserve_mib=0):
    projected = memory.used_mib + reserve_mib + limits.headroom_mib
    if projected >= memory.total_mib * limits.system_fraction:
        return f"host_memory: used={memory.used_mib:.0f} reserve={reserve_mib:.0f} headroom={limits.headroom_mib} MiB"
    return None


def gpu_read(index):
    result = subprocess.run([
        "nvidia-smi", f"--id={index}",
        "--query-gpu=utilization.gpu,power.draw,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, timeout=1.5, check=True)
    row = next(csv.reader(io.StringIO(result.stdout)))
    def number(value):
        try:
            result = float(value.strip())
            return result if math.isfinite(result) else None
        except ValueError:
            return None
    return dict(zip(("util_percent", "power_w", "used_mib", "total_mib"), map(number, row), strict=True))


def gpu_violation(sample, limits, reserve_mib=0):
    if limits.shared_gpu_memory:
        return None  # Unified RAM is counted once in MemAvailable.
    if sample is None or sample.get("total_mib") is None or sample.get("used_mib") is None:
        return "discrete_gpu_memory_unknown"
    if sample["used_mib"] + reserve_mib + limits.headroom_mib >= sample["total_mib"] * limits.system_fraction:
        return "discrete_gpu_memory"
    return None


class GpuSampler:
    def __init__(self, limits: Limits):
        self.limits = limits
        self.samples = []
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                sample = gpu_read(self.limits.gpu_index)
                self.samples.append({"monotonic_seconds": time.monotonic(), **sample})
                self.error = None
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.error = str(error)
            self.stop.wait(self.limits.gpu_sample_seconds)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)


def kill_group(process):
    """Kill descendants even if the process-group leader already exited."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def watch(process, limits, timeout_seconds, stream, gpu=None, memory_read=Memory.read):
    started = time.monotonic()
    peak = 0.0
    reason = None
    while True:
        memory = memory_read()
        peak = max(peak, memory.used_mib)
        elapsed = time.monotonic() - started
        stream.write(json.dumps({"elapsed_seconds": elapsed, **asdict(memory)}) + "\n")
        stream.flush()
        reason = memory_violation(memory, limits)
        if gpu and not limits.shared_gpu_memory:
            latest = gpu.samples[-1] if gpu.samples else None
            stale = latest is None or time.monotonic() - latest["monotonic_seconds"] > limits.gpu_sample_seconds + 2
            reason = reason or ("discrete_gpu_telemetry_unavailable" if stale or gpu.error else gpu_violation(latest, limits))
        if reason:
            break
        if process.poll() is not None:
            reason = "ok" if process.returncode == 0 else "process_failed"
            break
        if elapsed >= timeout_seconds:
            reason = "timeout"
            break
        time.sleep(limits.poll_seconds)
    return {"status": reason, "elapsed_seconds": elapsed, "peak_host_used_mib": peak}
