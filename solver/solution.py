"""Solution (candidate) model + scaffolding.

A Solution is the JSON container the SOL-ExecBench evaluator consumes
(see the repo's docs/solution.md): a build `spec` + a list of `sources`.
The evaluator calls `spec.entry_point` ("file::func") in **Destination
Passing Style** — inputs first (in `Definition.inputs` order), then
pre-allocated outputs (in `Definition.outputs` order), written in place.

`scaffold()` turns a fetched problem into a valid starting Solution:
- pytorch: a correct DPS baseline that delegates to the (inlined) reference —
  immediately valid and scoreable, the T_b-style starting point.
- other backends: a signature-correct stub raising NotImplementedError, with
  the reference inlined as `_reference_run` for the author to build against.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Languages the evaluator accepts (docs/solution.md). Python-family use a
# `.py` entry file; C++-family use `.cu`/`.cpp`.
PYTHON_LANGS = {"pytorch", "triton", "cute_dsl", "cutile", "cudnn_frontend"}
CPP_LANGS = {"cuda_cpp", "cutlass", "cudnn", "cublas"}
ALL_LANGS = PYTHON_LANGS | CPP_LANGS

# Default dependency hints per language (a valid subset of the supported set).
_DEFAULT_DEPS = {
    "pytorch": ["torch"],
    "triton": ["torch", "triton >= 2.3"],
    "cute_dsl": ["torch", "cutlass"],
    "cutile": ["torch"],
    "cudnn_frontend": ["torch"],
}


def input_output_names(definition: dict) -> tuple[list[str], list[str]]:
    """DPS argument order: input names then output names (insertion order)."""
    return list(definition.get("inputs", {})), list(definition.get("outputs", {}))


def dps_signature(definition: dict) -> str:
    ins, outs = input_output_names(definition)
    return "def run(" + ", ".join(ins + outs) + "):"


def _inline_reference(reference_src: str) -> str:
    """Rename the reference's top-level `run` to `_reference_run` so it can be
    inlined alongside our DPS `run` without a name clash."""
    return re.sub(r"(?m)^def run\(", "def _reference_run(", reference_src)


def _baseline_body(definition: dict) -> str:
    ins, outs = input_output_names(definition)
    call = "_reference_run(" + ", ".join(ins) + ")"
    if len(outs) == 1:
        assign = f"    {outs[0]}[:] = {call}"
    else:
        lines = [f"    _r = {call}"]
        lines += [f"    {name}[:] = _r[{i}]" for i, name in enumerate(outs)]
        assign = "\n".join(lines)
    return f"{dps_signature(definition)}\n{assign}\n"


@dataclass
class Solution:
    name: str
    definition: str
    author: str
    spec: dict
    sources: list[dict]
    description: str | None = None

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "definition": self.definition,
            "author": self.author,
            "spec": self.spec,
            "sources": self.sources,
        }
        if self.description:
            d["description"] = self.description
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"


def scaffold(
    problem_dir: Path,
    *,
    lang: str = "pytorch",
    author: str = "solver",
    name: str | None = None,
) -> Solution:
    """Build a starting Solution for a fetched problem directory."""
    if lang not in ALL_LANGS:
        raise ValueError(f"unknown language {lang!r}; choose from {sorted(ALL_LANGS)}")
    if lang in CPP_LANGS:
        raise ValueError(
            f"{lang!r} scaffolding not implemented yet (C++ path); use a python-family language"
        )

    problem_dir = Path(problem_dir)
    definition = json.loads((problem_dir / "definition.json").read_text())
    meta = json.loads((problem_dir / "metadata.json").read_text())
    def_name = definition["name"]
    reference_src = definition["reference"]

    inlined = _inline_reference(reference_src)
    if lang == "pytorch":
        header = "# Correct DPS baseline: delegates to the reference. Optimize from here.\n"
        body = _baseline_body(definition)
    else:
        header = (
            f"# {lang} scaffold — signature is DPS-correct; body is a stub.\n"
            f"# `_reference_run(...)` (below) is the correct-but-slow reference to build against.\n"
        )
        ins, outs = input_output_names(definition)
        body = (
            f"{dps_signature(definition)}\n"
            f'    raise NotImplementedError("implement the {lang} kernel; '
            f'DPS args = inputs {ins} then outputs {outs}")\n'
        )

    content = f"{header}\n{inlined}\n\n{body}"
    sol_name = name or f"{def_name}__{lang}_baseline"
    return Solution(
        name=sol_name,
        definition=def_name,
        author=author,
        description=f"task {meta['task_id']} ({meta['subset']}) {lang} scaffold",
        spec={
            "languages": [lang],
            "target_hardware": ["B200"],
            "entry_point": "kernel.py::run",
            "destination_passing_style": True,
            "dependencies": _DEFAULT_DEPS.get(lang, ["torch"]),
        },
        sources=[{"path": "kernel.py", "content": content}],
    )
