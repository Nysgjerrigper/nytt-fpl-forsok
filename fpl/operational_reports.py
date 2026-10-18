"""Immutable operational evidence reports, separate from paired FPL scorecards."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import re
from typing import Any

from fpl import config
from fpl.artifacts import sha256_file, write_json_new
from fpl.operational_evidence import inspect_evidence


def _json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(path.read_bytes(), object_pairs_hook=pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    json.dumps(value, allow_nan=False)
    return value


def _slug(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", value) is not None


def _files(root: Path) -> dict[str, str]:
    if root.is_symlink() or any(p.is_symlink() for p in root.parents) or not root.is_dir():
        raise ValueError("regular evidence directory required")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("symlinks and nonregular evidence are forbidden")
        if path.is_file():
            result[str(path.relative_to(root))] = sha256_file(path)
    return result


def _spec(path: Path, *, check_source: bool = True) -> tuple[dict, Path, bytes]:
    spec = _json(path)
    if (set(spec) - {"schema_version", "id", "name", "hypothesis", "source", "registration"}
            or type(spec.get("schema_version")) is not int or spec["schema_version"] != 1
            or not _slug(spec.get("id"))
            or any(not isinstance(spec.get(key), str) or not spec[key].strip() for key in ("name", "hypothesis"))):
        raise ValueError("invalid operational import specification")
    source = spec.get("source")
    if (not isinstance(source, dict) or set(source) != {"kind", "path", "terminal_sha256"}
            or source["kind"] not in {"solio_schedule", "native_diagnostics"}
            or not isinstance(source["path"], str) or not source["path"]):
        raise ValueError("explicit operational source kind/path/terminal pin required")
    digest = source["terminal_sha256"]
    if digest is not None and (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
        raise ValueError("invalid terminal SHA256")
    registration = spec.get("registration")
    if registration is not None and (not isinstance(registration, dict) or set(registration) != {"path", "sha256"}
        or not isinstance(registration["path"], str) or not registration["path"]
        or not isinstance(registration["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", registration["sha256"]) is None):
        raise ValueError("invalid registration link")
    root = Path(source["path"])
    root = root if root.is_absolute() else path.parent / root
    # Check before resolve so a symlink to an otherwise valid directory is not hidden.
    if check_source and (root.is_symlink() or any(parent.is_symlink() for parent in root.parents)):
        raise ValueError("symlinked source paths are forbidden")
    return spec, root.resolve(), path.read_bytes()


def _implementation() -> dict:
    return {name: sha256_file(Path(__file__).parent / name)
            for name in ("operational_reports.py", "operational_evidence.py", "artifacts.py", "config.py")}


def _card(spec: dict, evidence: dict, registration: dict | None, identity: dict) -> dict:
    return {"schema_version": 1, "report_kind": "OPERATIONAL_EVIDENCE",
        "experiment": {key: spec[key] for key in ("id", "name", "hypothesis")},
        "operations": evidence, "registration": registration,
        "report_identity": {key: identity[key] for key in ("attempt", "spec_sha256", "implementation", "reporting_runtime")},
        "performance": {"rank": None, "points": None, "delta": None, "ci": None},
        "promotion_eligible": False,
        "limitations": ["Recorded telemetry and hashes do not independently prove legality or causal validity.",
            "Operational entries are unranked; no paired FPL outcome is assessed.",
            "An incomplete capture is an observation, not a claim that its process stopped.",
            "A linked registration is not independently proven to have preceded execution."]}


def render_operations(card: dict) -> str:
    """Render measured operations and the exact card, retaining explicit unknowns."""
    def label(value):
        return str(value).replace('\n', ' ').replace('\r', ' ').replace('|', '\\|')
    operations = card['operations']
    title = label(card['experiment']['name'])
    lines = [f"# Training Ground operations: {title}", "",
        f"Status: **{label(operations['status'])}**; evidence: `{operations['kind']}`.", "",
        "Unranked. Points, differences and paired intervals are not assessed.", "",
        card['experiment']['hypothesis'], "", "## Recorded activity", "",
        "| Observation | Value |", "| --- | --- |"]
    for key in ('requested_origins', 'completed_origins', 'saved_origins', 'failed_origin',
                'failed_stage', 'solver_result_accepted', 'recorded_error'):
        if key in operations:
            value = operations[key]
            lines.append(f"| {label(key)} | {label(value) if value is not None else 'Not available'} |")
    labels = {'attempted_vendor_calls': 'Attempted solver calls', 'returned_vendor_calls': 'Returned solver calls',
              'failed_vendor_calls': 'Solver calls that raised', 'unknown_in_flight_calls': 'Calls with no recorded outcome',
              'native_started': 'Recorded native starts', 'native_finished': 'Recorded native returns',
              'native_error': 'Recorded native exceptions', 'native_run_unknown': 'Native calls with no recorded outcome',
              'saved_origins': 'Saved origin decisions'}
    lines.append(f"| Counting scope | {label(operations['count_scope'])} |")
    for key, value in operations['counts'].items():
        lines.append(f"| {label(labels.get(key, key))} | {label(value) if value is not None else 'Not available'} |")
    info = operations.get('native_info')
    if info is not None:
        lines.append(f"| Recorded native status | {label(info['status_name'])} |")
        lines.append(f"| Recorded native gap | {label(info['values'].get('mip_gap'))} |")
    lines += ["", "## Timing evidence", "", "Seconds, within each declared boundary; overlapping timings must not be added.", "",
              "| Boundary | Recorded evidence |", "| --- | --- |"]
    for key, value in operations['timings'].items():
        lines.append(f"| {label(key)} | {label(json.dumps(value, sort_keys=True, allow_nan=False))} |")
    lines += ["", "## Interpretation", ""]
    lines += ['- ' + item for item in card['limitations']]
    lines += ["", "## Complete machine record", "", "```json",
              json.dumps(card, indent=2, sort_keys=True, allow_nan=False), "```", ""]
    return "\n".join(lines)


def import_operations(path: str | Path, *, attempt: str = "first",
                      runs_dir: str | Path | None = None) -> Path:
    """Verify and snapshot saved telemetry without fitting, solving or scoring.

    A source terminal must be SHA-pinned. Unsealed source captures are explicitly
    incomplete snapshots. Registration links bind bytes, not historical chronology.
    """
    path = Path(path)
    spec, source, content = _spec(path)
    if not _slug(attempt):
        raise ValueError("invalid attempt name")
    files = _files(source)
    evidence = inspect_evidence(source, kind=spec['source']['kind'],
                                terminal_sha256=spec['source']['terminal_sha256'])
    registration = spec.get('registration')
    registration_bytes = None
    if registration is not None:
        registered = Path(registration['path'])
        registered = registered if registered.is_absolute() else path.parent / registered
        if registered.is_symlink() or any(p.is_symlink() for p in registered.parents):
            raise ValueError("symlinked registration path is forbidden")
        registration_bytes = registered.read_bytes()
        if hashlib.sha256(registration_bytes).hexdigest() != registration['sha256']:
            raise ValueError("registration bytes changed")
    identity = {"attempt": attempt, "spec_sha256": hashlib.sha256(content).hexdigest(), "source_files": files,
                "implementation": _implementation(), "reporting_runtime": {"python": platform.python_version()}}
    root = Path(config.PLAYGROUND_RUNS_DIR if runs_dir is None else runs_dir)
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        raise ValueError("symlinked output root is forbidden")
    root = root.resolve()
    if any(root.is_relative_to(folder.resolve()) for folder in (config.DATASETT_DIR, config.ROOT / 'fpl')):
        raise ValueError("operational reports cannot target production code or datasets")
    directory = root / spec['id'] / attempt
    if directory.is_symlink() or directory.parent.is_symlink():
        raise ValueError("symlinked attempt paths are forbidden")
    if directory.exists():
        if not (directory / 'operational-complete.json').exists():
            raise FileExistsError("unfinished import preserved; choose a new attempt")
        if _json(directory / 'operational-complete.json').get('identity') != identity:
            raise ValueError("existing operational import has different inputs or implementation")
        read_operations(directory)
        return directory
    if directory.is_relative_to(source) or source.is_relative_to(directory):
        raise ValueError("source and destination must not overlap")
    directory.mkdir(parents=True, exist_ok=False)
    write_json_new(directory / 'operational-started.json', {'schema_version': 1, 'identity': identity})
    try:
        (directory / 'spec.json').write_bytes(content)
        captured = directory / 'evidence'
        captured.mkdir()
        for name, digest in files.items():
            target = captured / name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = (source / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("source changed during import")
            target.write_bytes(data)
        if _files(source) != files:
            raise ValueError("source file set changed during import")
        if registration_bytes is not None:
            (directory / 'registration.json').write_bytes(registration_bytes)
        captured_evidence = inspect_evidence(captured, kind=spec['source']['kind'],
                                             terminal_sha256=spec['source']['terminal_sha256'])
        if captured_evidence != evidence:
            raise ValueError("captured operational evidence differs")
        card = _card(spec, evidence, registration, identity)
        write_json_new(directory / 'operational-card.json', card)
        (directory / 'REPORT.md').write_text(render_operations(card), encoding='utf-8')
        outputs = _files(directory)
        write_json_new(directory / 'operational-complete.json', {'schema_version': 1,
            'status': 'OPERATIONAL_REPORT_CAPTURED', 'identity': identity, 'outputs': outputs})
    except Exception as exc:
        write_json_new(directory / 'operational-failed.json', {'type': type(exc).__name__, 'reason': str(exc)})
        raise
    return directory


def read_operations(directory: str | Path) -> dict:
    """Revalidate the captured evidence and report; never read mutable original paths."""
    root = Path(directory)
    files = _files(root)
    if 'complete.json' in files or 'operational-failed.json' in files:
        raise ValueError("conflicting report lifecycle receipts")
    receipt = _json(root / 'operational-complete.json')
    if (type(receipt.get('schema_version')) is not int or receipt['schema_version'] != 1 or receipt.get('status') != 'OPERATIONAL_REPORT_CAPTURED'
            or not isinstance(receipt.get('outputs'), dict)):
        raise ValueError("invalid operational reporting receipt")
    if {k:v for k,v in files.items() if k != 'operational-complete.json'} != receipt['outputs']:
        raise ValueError("operational output coverage or hashes changed")
    spec, _, content = _spec(root / 'spec.json', check_source=False)
    identity = receipt['identity']
    if identity['spec_sha256'] != hashlib.sha256(content).hexdigest():
        raise ValueError("operational spec identity changed")
    if _json(root / 'operational-started.json').get('identity') != identity:
        raise ValueError("operational started identity changed")
    if _files(root / 'evidence') != identity['source_files']:
        raise ValueError("captured source identity changed")
    registration = spec.get('registration')
    if registration is not None:
        if sha256_file(root / 'registration.json') != registration['sha256']:
            raise ValueError("captured registration differs")
    elif (root / 'registration.json').exists():
        raise ValueError("unexpected registration artifact")
    evidence = inspect_evidence(root / 'evidence', kind=spec['source']['kind'],
                                terminal_sha256=spec['source']['terminal_sha256'])
    card = _card(spec, evidence, registration, identity)
    if _json(root / 'operational-card.json') != card:
        raise ValueError("operational card differs from verified evidence")
    if (root / 'REPORT.md').read_text() != render_operations(card):
        raise ValueError("operational human report differs from machine card")
    return card
