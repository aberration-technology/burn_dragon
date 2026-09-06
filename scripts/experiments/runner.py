"""Provenance and guarded execution for the local experiment matrix."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .config import Matrix
from .guard import GpuSampler, Memory, gpu_read, gpu_violation, kill_group, memory_violation, watch


def file_identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def repository_identity(path, archive=None):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args])
    head = git("rev-parse", "HEAD").decode().strip()
    patch = git("diff", "--binary", "HEAD")
    untracked = [Path(os.fsdecode(name)) for name in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if name]
    files = [file_identity(path / name) for name in untracked]
    if archive:
        if sum(item["bytes"] for item in files) > 64 * 1024 * 1024:
            raise ValueError("untracked source archive too large; exclude artifacts before running")
        archive.mkdir()
        (archive / "tracked.patch").write_bytes(patch)
        for name, identity in zip(untracked, files):
            if identity["bytes"] > 16 * 1024 * 1024:
                raise ValueError("untracked source archive too large; exclude artifacts before running")
            target = archive / "untracked" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path / name, target)
    return {"path": str(path.resolve()), "head": head, "patch_sha256": hashlib.sha256(patch).hexdigest(),
            "dirty": bool(patch or files), "untracked": files}


def run(matrix: Matrix, workspace: Path):
    workspace = workspace.resolve()
    output = (workspace / matrix.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "matrix.json", asdict(matrix))
    write_json(output / "results.json", {"complete": False, "cases": []})
    repositories = [repository_identity((workspace / path).resolve(), output / f"source-{index}")
                    for index, path in enumerate(matrix.repositories)]
    write_json(output / "sources.json", repositories)
    results = []
    clean_env = {key: value for key, value in os.environ.items() if not key.startswith(("BURN_", "DragonModel_"))}
    for case in matrix.cases:
        case_dir = output / case.id
        case_dir.mkdir()
        argv = [value.replace("{output}", str(case_dir)) for value in case.argv]
        executable = Path(argv[0]) if "/" in argv[0] else Path(shutil.which(argv[0]) or argv[0])
        executable = (workspace / executable).resolve()
        argv[0] = str(executable)
        inputs = [file_identity((workspace / value).resolve()) for value in case.inputs]
        manifest = {"case": asdict(case), "argv": argv, "executable": file_identity(executable), "inputs": inputs,
                    "environment": {key: value for key, value in clean_env.items()
                                    if key.startswith(("CUDA_", "CUBECL_", "WGPU_")) or key in ("LD_LIBRARY_PATH", "OMP_NUM_THREADS")}}
        binary_dir = output / "binaries" / manifest["executable"]["sha256"]
        binary_dir.mkdir(parents=True, exist_ok=True)
        captured_binary = binary_dir / executable.name
        if not captured_binary.exists():
            shutil.copy2(executable, captured_binary)
        if file_identity(captured_binary)["sha256"] != manifest["executable"]["sha256"]:
            raise ValueError("binary changed during capture")
        manifest["execution_argv"] = [str(captured_binary), *argv[1:]]
        write_json(case_dir / "manifest.json", manifest)
        result = {"id": case.id, "status": "not_started"}
        process = None
        sampler = None
        try:
            violation = memory_violation(Memory.read(), matrix.limits, case.expected_peak_mib)
            if case.gpu:
                first = gpu_read(matrix.limits.gpu_index)
                violation = violation or gpu_violation(first, matrix.limits, case.expected_peak_mib)
            if violation:
                result.update(status="admission_rejected", reason=violation)
            else:
                if case.gpu:
                    sampler = GpuSampler(matrix.limits)
                    sampler.samples.append({"monotonic_seconds": time.monotonic(), **first})
                    sampler.thread.start()
                print(f"starting {case.id}", flush=True)
                with (case_dir / "stdout.log").open("w") as stdout, (case_dir / "stderr.log").open("w") as stderr, (case_dir / "memory.jsonl").open("w") as memory:
                    process = subprocess.Popen(manifest["execution_argv"], cwd=workspace, env=clean_env, stdout=stdout, stderr=stderr, start_new_session=True)
                    result.update(watch(process, matrix.limits, case.timeout_seconds, memory, sampler))
        except BaseException as error:
            result.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "runner_error", reason=str(error))
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            if process:
                kill_group(process)
                result["exit_code"] = process.returncode
            if sampler:
                sampler.close()
                write_json(case_dir / "gpu.json", {"samples": sampler.samples, "last_error": sampler.error})
            if result["status"] == "ok":
                try:
                    after = [file_identity(Path(item["path"])) for item in inputs]
                    binary_after = file_identity(executable)
                    unchanged = after == inputs and binary_after == manifest["executable"]
                except OSError:
                    unchanged = False
                if not unchanged:
                    result.update(status="input_drift", reason="binary or declared inputs changed during execution")
            results.append(result)
            write_json(case_dir / "result.json", result)
            write_json(output / "results.json", {"complete": False, "cases": results})
        print(f"finished {case.id}: {result['status']}", flush=True)
        if result["status"] != "ok":
            break
    after = [repository_identity((workspace / path).resolve()) for path in matrix.repositories]
    source_unchanged = after == repositories
    summary = {"complete": source_unchanged and len(results) == len(matrix.cases) and all(r["status"] == "ok" for r in results),
               "source_unchanged": source_unchanged, "cases": results}
    write_json(output / "results.json", summary)
    return summary
