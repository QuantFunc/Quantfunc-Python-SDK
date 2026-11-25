"""Self-contained static architecture, layering, hygiene, examples, and M6 mutation gates.

This test module is 100% self-contained and never references private paths,
external term lists, or real API credentials.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_LAYER_RANK: dict[str, int] = {
    "native": 0,
    "core": 1,
    "models": 2,
    "accelerate": 3,
    "sampling": 3,
    "pipelines": 4,
    "cli": 4,
}

_MODEL_AGNOSTIC_SUBPKGS: frozenset[str] = frozenset({"core", "sampling", "accelerate"})

_MODEL_NAME_PATTERN = re.compile(
    r"(?:krea2|krea_2|qwenimage|qwen_image|qwen3_vl|ltx25|ltx2|minimax_h3|minimax|h3_stage2)",
    re.IGNORECASE,
)

# Real key pattern (constructed dynamically to avoid embedding key-like literals)
_API_KEY_PATTERN = re.compile(r"\b" + "qf_" + r"[0-9a-f]{64}\b")

_EXPECTED_EXAMPLES: tuple[str, ...] = (
    "krea2_t2i.py",
    "krea2_t2i_two_stage.py",
    "qwen_image21_t2i.py",
    "qwen_image21_transparent.py",
    "qwen_image21_edit.py",
    "ltx25_t2v_two_stage.py",
    "minimax_h3_fl2va.py",
    "minimax_h3_fl2va_two_stage.py",
    "minimax_h3_ref2va.py",
    "minimax_h3_ref2va_two_stage.py",
    "custom_multi_stage.py",
    "12_http_fastapi_server.py",
)

# Numeric constants forbidden in arithmetic/shape geometry inside quantfunc/models/*.py
# (ABI/rank constants 0..5, -1..-2, and standard timestep scale 1000.0 are permitted).
_SUSPECT_GEOMETRY_INTS: frozenset[int] = frozenset(
    {8, 16, 32, 64, 128, 192, 256, 512, 768, 1024, 1536, 2048, 3072, 4096}
)


def _subpkg_of(rel_path: Path) -> str | None:
    parts = rel_path.parts
    if len(parts) < 2 or parts[0] != "quantfunc":
        return None
    if parts[1] == "__init__.py":
        return None
    if parts[1] == "cli.py":
        return "cli"
    return parts[1]


def _resolve_import_target(node: ast.AST, current_module: str) -> list[str]:
    targets: list[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            targets.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        if node.level == 0:
            if node.module:
                targets.append(node.module)
                for alias in node.names:
                    targets.append(f"{node.module}.{alias.name}")
        else:
            cur_parts = current_module.split(".")
            base_parts = cur_parts[: max(0, len(cur_parts) - node.level)]
            if node.module:
                base_parts.extend(node.module.split("."))
            base = ".".join(base_parts)
            if base:
                targets.append(base)
                for alias in node.names:
                    targets.append(f"{base}.{alias.name}")
    return targets


def check_layering_dag(repo_root: Path) -> list[str]:
    """Verify strict 5-layer unidirectional DAG (incl. local imports) + model-agnostic L1/L3."""
    errors: list[str] = []
    pkg_root = repo_root / "quantfunc"
    if not pkg_root.is_dir():
        return [f"Missing package directory: {pkg_root}"]

    subpkg_edges: dict[str, set[str]] = {k: set() for k in _LAYER_RANK}

    for py_file in sorted(pkg_root.rglob("*.py")):
        rel = py_file.relative_to(repo_root)
        src_subpkg = _subpkg_of(rel)
        if src_subpkg is None or src_subpkg not in _LAYER_RANK:
            continue
        src_rank = _LAYER_RANK[src_subpkg]

        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(rel))
        mod_parts = list(rel.with_suffix("").parts)
        if mod_parts[-1] == "__init__":
            mod_parts = mod_parts[:-1]
        current_module = ".".join(mod_parts)

        # Visit ALL nodes (including imports nested inside functions/classes)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                for target in _resolve_import_target(node, current_module):
                    if not target.startswith("quantfunc."):
                        continue
                    parts = target.split(".")
                    if len(parts) < 2:
                        continue
                    dst_subpkg = parts[1]
                    if dst_subpkg not in _LAYER_RANK:
                        continue
                    dst_rank = _LAYER_RANK[dst_subpkg]
                    if dst_rank > src_rank:
                        errors.append(
                            f"Upward import violation in {rel}:{node.lineno}: "
                            f"{src_subpkg}(L{src_rank}) -> {dst_subpkg}(L{dst_rank}) via '{target}'"
                        )
                    if dst_subpkg != src_subpkg:
                        subpkg_edges[src_subpkg].add(dst_subpkg)

            # Enforce zero model names inside model-agnostic layers (L1 core, L3 sampling/accelerate)
            if src_subpkg in _MODEL_AGNOSTIC_SUBPKGS:
                if isinstance(node, ast.Name) and _MODEL_NAME_PATTERN.search(node.id):
                    errors.append(
                        f"Model-name leak in model-agnostic {rel}:{node.lineno}: identifier {node.id!r}"
                    )
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if _MODEL_NAME_PATTERN.search(node.value):
                        errors.append(
                            f"Model-name leak in model-agnostic {rel}:{node.lineno}: literal {node.value!r}"
                        )
                elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    if _MODEL_NAME_PATTERN.search(node.name):
                        errors.append(
                            f"Model-name leak in model-agnostic {rel}:{node.lineno}: def {node.name!r}"
                        )

    # Detect cycles in subpackage import graph via DFS
    visiting: set[str] = set()
    visited: set[str] = set()

    def _dfs(node_name: str, trail: list[str]) -> None:
        visiting.add(node_name)
        for nxt in sorted(subpkg_edges.get(node_name, ())):
            if nxt in visiting:
                errors.append(f"Subpackage import cycle detected: {' -> '.join([*trail, nxt])}")
            elif nxt not in visited:
                _dfs(nxt, [*trail, nxt])
        visiting.remove(node_name)
        visited.add(node_name)

    for sub in sorted(subpkg_edges):
        if sub not in visited:
            _dfs(sub, [sub])

    return errors


def check_no_hardcoded_model_geometry(repo_root: Path) -> list[str]:
    """Scan quantfunc/models/*.py for hardcoded spatial/channel geometry constants in math."""
    errors: list[str] = []
    models_dir = repo_root / "quantfunc" / "models"
    if not models_dir.is_dir():
        return [f"Missing models directory: {models_dir}"]

    for py_file in sorted(models_dir.glob("*.py")):
        if py_file.name in ("__init__.py", "base.py", "auto.py"):
            continue
        rel = py_file.relative_to(repo_root)
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(rel))

        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(
                node.op, ast.Mult | ast.FloorDiv | ast.Div | ast.Mod
            ):
                for side in (node.left, node.right):
                    if (
                        isinstance(side, ast.Constant)
                        and type(side.value) is int
                        and side.value in _SUSPECT_GEOMETRY_INTS
                    ):
                        errors.append(
                            f"Hardcoded geometry constant {side.value} in arithmetic at {rel}:{node.lineno}"
                        )
    return errors


def check_no_real_api_keys(repo_root: Path) -> list[str]:
    """Verify no tracked source, example, test, or doc file contains a real API key."""
    errors: list[str] = []
    scan_roots = [
        repo_root / "quantfunc",
        repo_root / "examples",
        repo_root / "tests",
    ]
    single_files = [
        repo_root / "README.md",
        repo_root / "SECURITY.md",
        repo_root / "pyproject.toml",
    ]

    candidates: list[Path] = []
    for root in scan_roots:
        if root.is_dir():
            candidates.extend(sorted(p for p in root.rglob("*") if p.is_file()))
    for sf in single_files:
        if sf.is_file():
            candidates.append(sf)

    for path in candidates:
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo", ".so", ".dll", ".whl"):
            continue
        data = path.read_bytes()
        if b"\x00" in data:
            continue
        text = data.decode("utf-8", errors="replace")
        if _API_KEY_PATTERN.search(text):
            rel = path.relative_to(repo_root)
            errors.append(f"API key literal detected in {rel}")
    return errors


def _extract_top_level_exports(repo_root: Path) -> set[str]:
    init_path = repo_root / "quantfunc" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    exports: set[str] = {"__version__"}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_LAZY_EXPORTS":
                    if isinstance(node.value, ast.Dict):
                        for k in node.value.keys:
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                exports.add(k.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "_LAZY_EXPORTS":
                if isinstance(node.value, ast.Dict):
                    for k in node.value.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            exports.add(k.value)
    return exports


def check_examples_ast_and_exports(repo_root: Path) -> list[str]:
    """Verify all 11 examples exist, parse cleanly with AST, and reference real SDK exports."""
    errors: list[str] = []
    examples_dir = repo_root / "examples"
    if not examples_dir.is_dir():
        return [f"Missing examples directory: {examples_dir}"]

    actual_files = {p.name for p in examples_dir.glob("*.py")}
    missing = set(_EXPECTED_EXAMPLES) - actual_files
    if missing:
        errors.append(f"Missing expected example scripts: {sorted(missing)}")

    top_exports = _extract_top_level_exports(repo_root)

    for name in _EXPECTED_EXAMPLES:
        path = examples_dir / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            errors.append(f"SyntaxError in examples/{name}: {exc}")
            continue

        has_quantfunc_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "quantfunc":
                has_quantfunc_import = True
                for alias in node.names:
                    if alias.name not in top_exports:
                        errors.append(
                            f"examples/{name}:{node.lineno} imports unknown symbol "
                            f"quantfunc.{alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("quantfunc."):
                has_quantfunc_import = True

        if not has_quantfunc_import:
            errors.append(f"examples/{name} does not import from quantfunc")
    return errors


# ── Test 35: Public static gates ────────────────────────────────────────────


def test_ast_five_layer_dag_and_model_agnostic_layers() -> None:
    errors = check_layering_dag(REPO_ROOT)
    assert errors == [], "\n".join(errors)


def test_models_zero_hardcoded_geometry() -> None:
    errors = check_no_hardcoded_model_geometry(REPO_ROOT)
    assert errors == [], "\n".join(errors)


def test_repo_zero_real_api_keys() -> None:
    errors = check_no_real_api_keys(REPO_ROOT)
    assert errors == [], "\n".join(errors)


def test_all_11_examples_ast_and_exported_symbols() -> None:
    errors = check_examples_ast_and_exports(REPO_ROOT)
    assert errors == [], "\n".join(errors)


def test_clean_import_quantfunc_without_optional_deps() -> None:
    script = """
import sys

BLOCKED = (
    "diffusers",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "modelscope",
    "fastapi",
    "uvicorn",
)
for mod in BLOCKED:
    sys.modules[mod] = None

import quantfunc

assert quantfunc.__version__ == "0.1.0"
for mod in BLOCKED:
    assert sys.modules.get(mod) is None, f"Optional dependency eagerly imported: {mod}"

sol = quantfunc.SolConfig(enabled=True, min_sequence_length=4096)
assert sol.min_sequence_length == 4096
ov = quantfunc.RuntimeOverrides.from_dict({"fbcache": 0.05, "sol": {"enabled": True, "tau": 1.3}})
assert ov.block_cache == 0.05 and ov.sol.enabled is True and ov.quality_enhance is quantfunc.UNSET
for mod in BLOCKED:
    assert sys.modules.get(mod) is None, f"Optional dependency imported by L1 exports: {mod}"
"""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


# ── Test 49: L1 overrides zero-torch, L3 no-cycle mutation & Example 12 hygiene ─


def test_l1_overrides_zero_torch_l3_no_cycle_and_example12_hygiene(tmp_path: Path) -> None:
    # (1) Pure-stdlib L1 RuntimeOverrides & SolConfig.from_dict with torch + all 7 optional deps blocked
    script = """
import sys

BLOCKED = (
    "diffusers",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "modelscope",
    "fastapi",
    "uvicorn",
    "torch",
)
for mod in BLOCKED:
    sys.modules[mod] = None

from quantfunc.core.overrides import RuntimeOverrides, UNSET
from quantfunc.core.sol import SolConfig

ov = RuntimeOverrides.from_dict(
    {
        "fbcache": 0.05,
        "stepcache": 0.1,
        "sol": {"enabled": True, "tau": 1.3, "min_tokens": 4096},
        "loras": [{"name": "style", "scale": 0.8}],
    }
)
assert ov.block_cache == 0.05
assert ov.step_cache == 0.1
assert isinstance(ov.sol, SolConfig) and ov.sol.enabled is True and ov.sol.min_sequence_length == 4096
assert ov.quality_enhance is UNSET

try:
    RuntimeOverrides.from_dict({"fbcache": 0.05, "block_cache": 0.1})
    raise SystemExit("MUST FAIL on conflicting aliases")
except ValueError:
    pass

try:
    RuntimeOverrides.from_dict({"unknown_dial": 1})
    raise SystemExit("MUST FAIL on unknown key")
except ValueError:
    pass

for mod in BLOCKED:
    assert sys.modules.get(mod) is None, f"Blocked dependency imported by L1 core/overrides.py: {mod}"
"""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    # (2) Model-agnostic check on both L3a accelerate/serving.py and L4a pipelines/serving.py
    for rel_path in ("quantfunc/accelerate/serving.py", "quantfunc/pipelines/serving.py"):
        src = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        tree = ast.parse(src, filename=rel_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert not _MODEL_NAME_PATTERN.search(node.id), f"Model name leak in {rel_path}: {node.id!r}"
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not _MODEL_NAME_PATTERN.search(node.value), f"Model name leak in {rel_path}: {node.value!r}"
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                assert not _MODEL_NAME_PATTERN.search(node.name), f"Model name leak in {rel_path}: {node.name!r}"

    # Verify accelerate/serving.py NEVER imports quantfunc.sampling
    acc_serving_src = (REPO_ROOT / "quantfunc" / "accelerate" / "serving.py").read_text(encoding="utf-8")
    acc_tree = ast.parse(acc_serving_src, filename="quantfunc/accelerate/serving.py")
    for node in ast.walk(acc_tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            for target in _resolve_import_target(node, "quantfunc.accelerate.serving"):
                assert not target.startswith("quantfunc.sampling"), (
                    f"Illegal L3a->L3b import in quantfunc/accelerate/serving.py: {target}"
                )

    # (3) Mutation arm: injecting accelerate -> sampling cycle into accelerate/serving.py MUST turn RED!
    copy_root = tmp_path / "cycle_copy"
    shutil.copytree(REPO_ROOT / "quantfunc", copy_root / "quantfunc")
    assert check_layering_dag(copy_root) == []

    serving_copy = copy_root / "quantfunc" / "accelerate" / "serving.py"
    orig_serving = serving_copy.read_text(encoding="utf-8")
    serving_copy.write_text(
        orig_serving + "\ndef _bad_cycle():\n    from quantfunc.sampling.runner import MultiStageRunner\n    return MultiStageRunner\n",
        encoding="utf-8",
    )
    cycle_errs = check_layering_dag(copy_root)
    assert any("Subpackage import cycle detected" in e for e in cycle_errs), (
        f"Expected subpackage import cycle detection, got {cycle_errs}"
    )
    serving_copy.write_text(orig_serving, encoding="utf-8")


# ── Mutation Arm M6: Every static gate MUST turn RED on injected violation ──


def test_mutation_m6_static_gates(tmp_path: Path) -> None:
    # Copy quantfunc/ and examples/ into an isolated temporary tree
    copy_root = tmp_path / "sdk_copy"
    shutil.copytree(REPO_ROOT / "quantfunc", copy_root / "quantfunc")
    shutil.copytree(REPO_ROOT / "examples", copy_root / "examples")
    (copy_root / "tests").mkdir()

    # Baseline on clean copy must pass all gates
    assert check_layering_dag(copy_root) == []
    assert check_no_hardcoded_model_geometry(copy_root) == []
    assert check_no_real_api_keys(copy_root) == []

    # (1) Inject L1 -> L2 upward import inside quantfunc/core/sol.py
    sol_path = copy_root / "quantfunc" / "core" / "sol.py"
    if not sol_path.exists():
        sol_path.parent.mkdir(parents=True, exist_ok=True)
        sol_path.write_text("", encoding="utf-8")
    orig_sol = sol_path.read_text(encoding="utf-8")
    sol_path.write_text(
        orig_sol + "\ndef _bad():\n    from quantfunc.models.base import QuantFuncBaseTransformer\n",
        encoding="utf-8",
    )
    l1_errs = check_layering_dag(copy_root)
    assert any("Upward import violation" in e for e in l1_errs), f"Expected L1->L2 failure, got {l1_errs}"
    sol_path.write_text(orig_sol, encoding="utf-8")

    # (2) Inject model name inside L3 quantfunc/sampling/runner.py
    runner_path = copy_root / "quantfunc" / "sampling" / "runner.py"
    orig_runner = runner_path.read_text(encoding="utf-8")
    runner_path.write_text(
        orig_runner + "\n_LEAK = 'krea2'\n",
        encoding="utf-8",
    )
    leak_errs = check_layering_dag(copy_root)
    assert any("Model-name leak" in e for e in leak_errs), f"Expected model-name leak, got {leak_errs}"
    runner_path.write_text(orig_runner, encoding="utf-8")

    # (3) Inject synthetic API key literal ("qf_" + "a" * 64)
    secret_file = copy_root / "examples" / "leaked_sample.py"
    secret_file.write_text(f'API_KEY = "{"qf_" + "a" * 64}"\n', encoding="utf-8")
    key_errs = check_no_real_api_keys(copy_root)
    assert any("leaked_sample.py" in e for e in key_errs), f"Expected key detection, got {key_errs}"
    secret_file.unlink()

    # (4) Inject hardcoded geometry `w = x.shape[-1] * 16` into quantfunc/models/krea2.py
    krea_path = copy_root / "quantfunc" / "models" / "krea2.py"
    if not krea_path.exists():
        krea_path.parent.mkdir(parents=True, exist_ok=True)
        krea_path.write_text("", encoding="utf-8")
    orig_krea = krea_path.read_text(encoding="utf-8")
    krea_path.write_text(
        orig_krea + "\ndef _bad_geom(x):\n    w = x.shape[-1] * 16\n    return w\n",
        encoding="utf-8",
    )
    geom_errs = check_no_hardcoded_model_geometry(copy_root)
    assert any("Hardcoded geometry constant 16" in e for e in geom_errs), (
        f"Expected geometry gate failure, got {geom_errs}"
    )
    krea_path.write_text(orig_krea, encoding="utf-8")
