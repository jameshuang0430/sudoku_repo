from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_run_metadata(
    *,
    command_name: str,
    argv: Sequence[str] | None = None,
    checkpoint_path: Path | str | None = None,
    dataset_path: Path | str | None = None,
    model_type: str | None = None,
    decode_preset: str | None = None,
    decode_mode: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    run_id = f"{timestamp.replace('-', '').replace(':', '').replace('T', '_').replace('Z', '')}-{uuid4().hex[:8]}"

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "utc_timestamp": timestamp,
        "git_commit": get_git_commit_sha(),
        "entrypoint": command_name,
        "argv": argv_list,
        "command": _format_command(command_name, argv_list),
    }

    optional_fields = {
        "checkpoint_path": checkpoint_path,
        "dataset_path": dataset_path,
        "model_type": model_type,
        "decode_preset": decode_preset,
        "decode_mode": decode_mode,
    }
    for key, value in optional_fields.items():
        if value is not None:
            metadata[key] = _normalize_value(value)

    if extra is not None:
        for key, value in extra.items():
            if value is not None:
                metadata[key] = _normalize_value(value)

    return metadata


def get_git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def _format_command(command_name: str, argv: Sequence[str]) -> str:
    parts = [sys.executable, "-m", command_name, *argv]
    return " ".join(_shell_quote(part) for part in parts)


def _shell_quote(value: str) -> str:
    if not value or any(character.isspace() for character in value) or '"' in value:
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value
