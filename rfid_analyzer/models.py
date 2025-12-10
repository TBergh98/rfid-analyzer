from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


VALID_ACTIONS = {"IN", "OUT"}


@dataclass
class Event:
    timestamp: datetime
    action: str
    chicken_id: str
    nest_id: str
    source_file: Optional[Path] = None

    def __post_init__(self) -> None:
        normalized = self.action.strip().upper()
        if normalized not in VALID_ACTIONS:
            raise ValueError(f"Invalid action: {self.action}")
        self.action = normalized

    @property
    def date_str(self) -> str:
        return self.timestamp.strftime("%d/%m/%Y")

    @property
    def time_str(self) -> str:
        return self.timestamp.strftime("%H:%M:%S")

    def to_row(self) -> list[str]:
        return [self.date_str, self.time_str, self.action, self.chicken_id, self.nest_id]
