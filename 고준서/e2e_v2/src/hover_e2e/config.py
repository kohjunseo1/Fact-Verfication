from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    config_path: Path
    raw: dict[str, Any]
    data_root: Path
    output_root: Path

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def data_path(self, key: str) -> Path:
        return (self.data_root / self.raw["data"][key]).resolve()

    def stage_dir(self, run_name: str, stage: str) -> Path:
        return (self.output_root / run_name / stage).resolve()


def load_settings(
    config_path: str | Path,
    *,
    data_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> Settings:
    path = Path(config_path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 2:
        raise ValueError(f"Unsupported config schema: {raw.get('schema_version')}")
    # configs/base.json -> e2e_v2 -> 고준서; data lives at 고준서/data.
    default_data = path.parents[2] / "data"
    default_output = path.parents[1] / "runs"
    resolved_data = Path(
        data_root or os.environ.get("HOVER_DATA_ROOT", default_data)
    ).expanduser().resolve()
    resolved_output = Path(
        output_root or os.environ.get("HOVER_E2E_OUTPUT_ROOT", default_output)
    ).expanduser().resolve()
    return Settings(path, raw, resolved_data, resolved_output)


def require_paths(settings: Settings, keys: list[str], *, allow_missing: bool = False) -> dict[str, Path]:
    paths = {key: settings.data_path(key) for key in keys}
    missing = [path for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing and not allow_missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required local inputs are missing:\n{listing}")
    return paths
