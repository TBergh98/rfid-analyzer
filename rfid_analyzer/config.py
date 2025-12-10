from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass
class CleanerConfig:
    input_folder: Path
    output_folder: Path
    ovodeposition_start_date: date
    threshold_pre_ovodeposition: int
    threshold_post_ovodeposition: int
    datetime_format: str = "%d/%m/%Y %H:%M:%S"

    def threshold_for(self, when: datetime) -> int:
        """Pick the correct threshold based on ovodeposition start date."""
        return (
            self.threshold_post_ovodeposition
            if when.date() >= self.ovodeposition_start_date
            else self.threshold_pre_ovodeposition
        )


def _resolve_path(base: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return (base / path).resolve() if not path.is_absolute() else path


def load_config(path: Path) -> CleanerConfig:
    """Load YAML config into a typed object."""
    with open(path, "r", encoding="utf-8") as handle:
        data: Dict[str, Any] = yaml.safe_load(handle) or {}

    missing = [
        key
        for key in [
            "input_folder",
            "output_folder",
            "ovodeposition_start_date",
            "threshold_pre_ovodeposition",
            "threshold_post_ovodeposition",
        ]
        if key not in data
    ]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    base = path.parent
    input_folder = _resolve_path(base, str(data["input_folder"]))
    output_folder = _resolve_path(base, str(data["output_folder"]))

    ovodeposition_start_date = date.fromisoformat(str(data["ovodeposition_start_date"]))

    return CleanerConfig(
        input_folder=input_folder,
        output_folder=output_folder,
        ovodeposition_start_date=ovodeposition_start_date,
        threshold_pre_ovodeposition=int(data["threshold_pre_ovodeposition"]),
        threshold_post_ovodeposition=int(data["threshold_post_ovodeposition"]),
        datetime_format=data.get("datetime_format", "%d/%m/%Y %H:%M:%S"),
    )
