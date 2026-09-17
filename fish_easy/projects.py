from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"
PROJECTS.mkdir(exist_ok=True)


def safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._-")
    if not value:
        raise ValueError("Project name is required.")
    return value


def _path(name: str) -> Path:
    return PROJECTS / safe_name(name) / "project.json"


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return _normalize_training_state(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}




_LEGACY_EPOCH_KEYS = {
    "training_mode", "epochs", "save_every_epochs", "eval_every_epochs",
}

def _normalize_training_state(payload: dict) -> dict:
    training = payload.get("training")
    if isinstance(training, dict):
        for key in _LEGACY_EPOCH_KEYS:
            training.pop(key, None)
        evaluation = training.get("eval")
        if isinstance(evaluation, dict):
            for key in ("audio", "transcript", "whisper_model", "whisper_batch"):
                evaluation.pop(key, None)
    return payload


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def list_projects(surface: str | None = None) -> list[str]:
    names: list[str] = []
    for folder in PROJECTS.iterdir():
        if not folder.is_dir() or not (folder / "project.json").is_file():
            continue
        data = _read(folder / "project.json")
        if surface == "dataset" and data.get("dataset_deleted"):
            continue
        if surface == "training" and data.get("training_deleted"):
            continue
        if data.get("dataset_deleted") and data.get("training_deleted"):
            continue
        names.append(folder.name)
    return sorted(names, key=str.casefold)


def save_project(name: str, data: dict | None = None) -> str:
    name = safe_name(name)
    existing = load_project(name)
    incoming = dict(data or {})
    # Preserve independent surface blocks unless explicitly replaced.
    payload = {**existing, **incoming}
    payload["schema_version"] = 5
    payload["project"] = name
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload.setdefault("dataset", existing.get("dataset", {}))
    payload.setdefault("training", existing.get("training", {}))
    # Keep the two project surfaces explicit even when one has never been
    # edited. "None" in a GUI dropdown is never serialized as a project.
    if not isinstance(payload["dataset"], dict):
        payload["dataset"] = {}
    if not isinstance(payload["training"], dict):
        payload["training"] = {}
    payload.setdefault("dataset_deleted", False)
    payload.setdefault("training_deleted", False)
    _normalize_training_state(payload)
    _write(_path(name), payload)
    return name


def load_project(name: str) -> dict:
    if not name:
        return {}
    return _read(_path(name))


def update_surface(name: str, surface: str, values: dict) -> str:
    if surface not in {"dataset", "training"}:
        raise ValueError("Unknown project surface.")
    data = load_project(name)
    data[surface] = dict(values or {})
    data[f"{surface}_deleted"] = False
    return save_project(name, data)


def delete_surface(name: str, surface: str) -> str:
    if surface not in {"dataset", "training"}:
        raise ValueError("Unknown project surface.")
    data = load_project(name)
    if not data:
        return f"Project '{name}' does not exist."
    data[f"{surface}_deleted"] = True
    save_project(name, data)
    if data.get("dataset_deleted") and data.get("training_deleted"):
        shutil.rmtree(PROJECTS / safe_name(name), ignore_errors=True)
        return f"Deleted '{name}' from Dataset and Training projects."
    return f"Removed '{name}' from {surface.title()} projects; the other project state was preserved."


def next_incremental_name(source: str) -> str:
    source = safe_name(source)
    base = re.sub(r"[-_ ]\d+$", "", source) or source
    occupied = set(list_projects())
    if base not in occupied:
        # Cloning an already suffixed legacy name should still create base-2.
        occupied.add(base)
    index = 2
    while f"{base}-{index}" in occupied:
        index += 1
    return f"{base}-{index}"


def clone_project(source: str, destination: str | None = None) -> str:
    source = safe_name(source)
    destination = safe_name(destination) if destination else next_incremental_name(source)
    src = PROJECTS / source
    dst = PROJECTS / destination
    if not (src / "project.json").is_file():
        raise ValueError("Source project not found.")
    if dst.exists():
        raise ValueError("Destination project already exists.")
    shutil.copytree(src, dst)
    data = load_project(destination)
    data["project"] = destination
    data["schema_version"] = 5
    data["cloned_from"] = source
    data["cloned_at"] = datetime.now(timezone.utc).isoformat()
    data["updated_at"] = data["cloned_at"]
    data.setdefault("dataset", {})
    data.setdefault("training", {})
    _normalize_training_state(data)
    _write(dst / "project.json", data)
    return destination
