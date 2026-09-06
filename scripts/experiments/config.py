"""Strict experiment manifest, with no shell commands or training env overrides."""

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import tomllib


def strict(cls, values):
    unknown = values.keys() - cls.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Limits:
    system_fraction: float = 0.90
    headroom_mib: int = 4096
    poll_seconds: float = 0.25
    gpu_sample_seconds: float = 2.0
    shared_gpu_memory: bool = True
    gpu_index: int = 0

    def __post_init__(self):
        for name in ("system_fraction", "headroom_mib", "poll_seconds", "gpu_sample_seconds"):
            positive(getattr(self, name), name)
        if self.system_fraction > 0.90:
            raise ValueError("system_fraction must not exceed 0.90")
        if self.poll_seconds > 1.0 or self.gpu_sample_seconds < 1.0:
            raise ValueError("memory polling must be <=1s; GPU sampling must be >=1s")
        if type(self.shared_gpu_memory) is not bool or type(self.gpu_index) is not int or self.gpu_index < 0:
            raise ValueError("invalid GPU memory topology or index")


@dataclass(frozen=True)
class Case:
    id: str
    argv: list[str]
    expected_peak_mib: int
    timeout_seconds: float
    gpu: bool = False
    inputs: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", self.id):
            raise ValueError("case id must be a safe directory name")
        if not isinstance(self.argv, list) or not self.argv or not all(isinstance(x, str) and x for x in self.argv):
            raise ValueError("argv must be a nonempty string array")
        if not isinstance(self.inputs, list) or not all(isinstance(x, str) for x in self.inputs):
            raise ValueError("inputs must be a string array")
        if type(self.gpu) is not bool:
            raise ValueError("gpu must be boolean")
        positive(self.expected_peak_mib, "expected_peak_mib")
        positive(self.timeout_seconds, "timeout_seconds")


@dataclass(frozen=True)
class Matrix:
    version: int
    output: str
    cases: list[Case]
    limits: Limits = field(default_factory=Limits)
    repositories: list[str] = field(default_factory=lambda: ["."])

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported experiment schema")
        if not isinstance(self.output, str) or not self.output:
            raise ValueError("output must name a new directory")
        if not self.cases or len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("matrix needs unique, nonempty cases")
        if not isinstance(self.repositories, list) or not self.repositories or not all(isinstance(x, str) for x in self.repositories):
            raise ValueError("repositories must be a nonempty string array")


def load(path: Path) -> Matrix:
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    values["limits"] = strict(Limits, values.get("limits", {}))
    values["cases"] = [strict(Case, item) for item in values.get("cases", [])]
    return strict(Matrix, values)
