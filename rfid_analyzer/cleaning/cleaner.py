import csv
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from rfid_analyzer.config import CleanerConfig
from rfid_analyzer.models import Event

logger = logging.getLogger(__name__)


class EventCleaner:
    """Cleans RFID events using configurable thresholds."""

    def __init__(self, config: CleanerConfig) -> None:
        self.config = config

    def load_events_from_file(self, file_path: Path) -> List[Event]:
        rows: List[Event] = []
        with open(file_path, "r", encoding="utf-8", newline="") as handle:
            sample = handle.read(1024)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ";" if ";" in sample else ","

            reader = csv.reader(handle, delimiter=delimiter)
            for raw in reader:
                if not raw:
                    continue
                # Skip header if present
                if raw[0].lower().startswith("data"):
                    continue
                if len(raw) < 5:
                    logger.warning("Skipping short row in %s: %s", file_path.name, raw)
                    continue
                date_part, time_part, action, chicken_id, nest_id = [part.strip() for part in raw[:5]]
                try:
                    ts = datetime.strptime(
                        f"{date_part} {time_part}", self.config.datetime_format
                    )
                except ValueError:
                    logger.warning(
                        "Invalid datetime in %s: %s %s", file_path.name, date_part, time_part
                    )
                    continue
                try:
                    rows.append(
                        Event(
                            timestamp=ts,
                            action=action,
                            chicken_id=chicken_id,
                            nest_id=nest_id,
                            source_file=file_path,
                        )
                    )
                except ValueError as exc:
                    logger.warning("Invalid action in %s: %s", file_path.name, exc)
        return rows

    def load_events(self) -> Dict[Path, List[Event]]:
        events: Dict[Path, List[Event]] = {}
        for file_path in sorted(self.config.input_folder.glob("*.csv")):
            file_events = self.load_events_from_file(file_path)
            if not file_events:
                logger.info("No valid events in %s", file_path.name)
            events[file_path] = file_events
        return events

    def _merge_short_gaps(self, events: List[Event]) -> List[Event]:
        cleaned: List[Event] = []
        i = 0
        while i < len(events):
            current = events[i]
            if (
                current.action == "OUT"
                and i + 1 < len(events)
                and events[i + 1].action == "IN"
            ):
                gap_seconds = (events[i + 1].timestamp - current.timestamp).total_seconds()
                threshold = self.config.threshold_for(current.timestamp)
                if gap_seconds <= threshold:
                    # Drop the OUT and the following IN
                    i += 2
                    continue
            cleaned.append(current)
            i += 1
        return cleaned

    def clean_events(self, events: Iterable[Event]) -> List[Event]:
        grouped: Dict[Tuple[str, str], List[Event]] = defaultdict(list)
        for event in sorted(events, key=lambda e: e.timestamp):
            grouped[(event.chicken_id, event.nest_id)].append(event)

        cleaned_all: List[Event] = []
        for key, group in grouped.items():
            group_sorted = sorted(group, key=lambda e: e.timestamp)
            cleaned = self._merge_short_gaps(group_sorted)
            cleaned_all.extend(cleaned)
            logger.debug("Group %s cleaned: %d -> %d", key, len(group_sorted), len(cleaned))

        return sorted(cleaned_all, key=lambda e: e.timestamp)

    def write_events(self, events: List[Event], target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=",")
            writer.writerow(["Data", "Ora", "Azione", "ID Gallina", "ID Nido"])
            for event in events:
                writer.writerow(event.to_row())

    def clean_file(self, file_path: Path, output_folder: Path) -> int:
        events = self.load_events_from_file(file_path)
        cleaned = self.clean_events(events)
        output_name = output_folder / file_path.name
        self.write_events(cleaned, output_name)
        logger.info(
            "Cleaned %s: %d -> %d rows", file_path.name, len(events), len(cleaned)
        )
        return len(cleaned)

    def run(self) -> Dict[str, int]:
        output_folder = (
            self.config.output_folder
            / f"threshold_pre{self.config.threshold_pre_ovodeposition}s_post{self.config.threshold_post_ovodeposition}s"
        )
        counts: Dict[str, int] = {}
        for file_path in sorted(self.config.input_folder.glob("*.csv")):
            counts[file_path.name] = self.clean_file(file_path, output_folder)
        return counts
