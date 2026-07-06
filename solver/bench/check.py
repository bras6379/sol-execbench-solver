"""Static pre-flight checks for a candidate Solution (no GPU).

Catches the failures that would waste a GPU run: malformed schema, an entry
point that doesn't exist or has the wrong DPS signature, disallowed
languages/paths, and reward-hack patterns the real harness rejects
(monkey-patching the timer, thread injection, try/except fallbacks). This is
a *subset* of the official validation — passing here means "worth sending to
the GPU", not "guaranteed to score".
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import solution as sol_mod


@dataclass
class CheckReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _dps_expected_params(definition: dict) -> list[str]:
    ins, outs = sol_mod.input_output_names(definition)
    return ins + outs


def check_solution(sol: dict, definition: dict | None = None) -> CheckReport:
    """Validate a Solution dict; if `definition` is given, also check the DPS
    signature against its inputs/outputs."""
    r = CheckReport()

    # --- top-level schema ---
    for f in ("name", "definition", "author", "spec", "sources"):
        if f not in sol:
            r.errors.append(f"missing top-level field: {f}")
    spec = sol.get("spec", {})
    sources = sol.get("sources", [])
    if r.errors:
        return r

    # --- spec ---
    langs = spec.get("languages") or []
    if not langs:
        r.errors.append("spec.languages is empty")
    bad = [l for l in langs if l not in sol_mod.ALL_LANGS]
    if bad:
        r.errors.append(f"unsupported language(s): {bad}")
    if set(langs) & sol_mod.PYTHON_LANGS and set(langs) & sol_mod.CPP_LANGS:
        r.errors.append("C++ and Python languages cannot be mixed")

    entry = spec.get("entry_point", "")
    if "::" not in entry:
        r.errors.append("spec.entry_point must be 'file::function'")
        return r
    entry_file, entry_fn = entry.split("::", 1)

    # --- sources ---
    paths = [s.get("path") for s in sources]
    if entry_file not in paths:
        r.errors.append(f"entry_point file {entry_file!r} not in sources")
    if len(paths) != len(set(paths)):
        r.errors.append("duplicate source paths")
    for p in paths:
        if not p or p.startswith("/") or ".." in Path(p).parts:
            r.errors.append(f"invalid source path: {p!r}")

    # --- entry function signature (python only) ---
    if entry_file.endswith(".py"):
        src = next((s["content"] for s in sources if s.get("path") == entry_file), "")
        r_sig = _check_python_entry(src, entry_fn, definition)
        r.errors += r_sig.errors
        r.warnings += r_sig.warnings

    return r


def _check_python_entry(src: str, fn_name: str, definition: dict | None) -> CheckReport:
    r = CheckReport()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        r.errors.append(f"entry file does not parse: {e}")
        return r

    fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name),
        None,
    )
    if fn is None:
        r.errors.append(f"entry function {fn_name!r} not defined in entry file")
        return r

    if definition is not None:
        expected = _dps_expected_params(definition)
        actual = [a.arg for a in fn.args.args]
        has_kwargs = fn.args.kwarg is not None  # **kwargs allowed and ignored
        if actual != expected and not (has_kwargs and actual == expected[: len(actual)]):
            r.errors.append(
                f"DPS signature mismatch: {fn_name}({', '.join(actual)}) "
                f"but expected inputs+outputs order ({', '.join(expected)})"
            )

    # --- reward-hack lints (whole module) ---
    dump = ast.dump(tree)
    if "elapsed_time" in src and "Event" in src:
        r.warnings.append("references torch.cuda.Event.elapsed_time — the harness rejects timer monkey-patching")
    if "import threading" in src or "threading." in src:
        r.errors.append("uses threading — the harness rejects thread injection")
    if any(isinstance(n, ast.Try) for n in ast.walk(fn)):
        r.warnings.append("entry function contains try/except — avoid correctness-hiding fallbacks")
    return r


def check_solution_file(path: Path, problems_dir: Path = Path("problems")) -> CheckReport:
    """Load a Solution JSON and check it, resolving its definition for the
    DPS signature check when the matching fetched problem is found."""
    sol = json.loads(Path(path).read_text())
    definition = _find_definition(sol.get("definition"), Path(problems_dir))
    return check_solution(sol, definition)


def _find_definition(def_name: str | None, problems_dir: Path) -> dict | None:
    if not def_name or not problems_dir.exists():
        return None
    for d in problems_dir.iterdir():
        dj = d / "definition.json"
        if dj.exists():
            try:
                obj = json.loads(dj.read_text())
            except Exception:
                continue
            if obj.get("name") == def_name:
                return obj
    return None
