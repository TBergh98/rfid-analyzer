from datetime import datetime, date
from pathlib import Path
import csv

from rfid_analyzer.cleaning import EventCleaner
from rfid_analyzer.config import CleanerConfig, load_config
from rfid_analyzer.models import Event


def make_config(tmp_path: Path) -> CleanerConfig:
    return CleanerConfig(
        input_folder=tmp_path / "input",
        output_folder=tmp_path / "output",
        ovodeposition_start_date=date(2023, 11, 13),
        threshold_pre_ovodeposition=60,
        threshold_post_ovodeposition=120,
    )


def test_merge_short_gap_removes_exit_and_reentry(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    cleaner = EventCleaner(config)
    events = [
        Event(datetime(2023, 10, 10, 15, 30, 0), "IN", "231", "1.1"),
        Event(datetime(2023, 10, 10, 15, 31, 0), "OUT", "231", "1.1"),
        Event(datetime(2023, 10, 10, 15, 31, 20), "IN", "231", "1.1"),
    ]

    cleaned = cleaner.clean_events(events)

    assert len(cleaned) == 1
    assert cleaned[0].action == "IN"


def test_interleaved_hens_do_not_block_merge(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    cleaner = EventCleaner(config)
    events = [
        Event(datetime(2023, 10, 10, 10, 0, 0), "IN", "A", "1.1"),
        Event(datetime(2023, 10, 10, 10, 5, 0), "OUT", "A", "1.1"),
        Event(datetime(2023, 10, 10, 10, 5, 10), "IN", "B", "1.1"),
        Event(datetime(2023, 10, 10, 10, 5, 20), "IN", "A", "1.1"),
    ]

    cleaned = cleaner.clean_events(events)

    actions = [(e.chicken_id, e.action) for e in cleaned]
    assert ("A", "OUT") not in actions
    assert ("A", "IN") in actions
    assert ("B", "IN") in actions
    assert len(actions) == 2


def test_run_writes_cleaned_csv(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    sample_file = input_dir / "sample.csv"

    with open(sample_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Data", "Ora", "Azione", "ID Gallina", "ID Nido"])
        writer.writerow(["12/11/2023", "10:00:00", "IN", "1", "1.1"])
        writer.writerow(["12/11/2023", "10:02:00", "OUT", "1", "1.1"])
        writer.writerow(["12/11/2023", "10:02:20", "IN", "1", "1.1"])
        writer.writerow(["14/11/2023", "11:00:00", "OUT", "1", "1.1"])
        writer.writerow(["14/11/2023", "11:03:10", "IN", "1", "1.1"])

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_folder: {input_dir}",
                f"output_folder: {output_dir}",
                "ovodeposition_start_date: \"2023-11-13\"",
                "threshold_pre_ovodeposition: 60",
                "threshold_post_ovodeposition: 120",
                "datetime_format: \"%d/%m/%Y %H:%M:%S\"",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    cleaner = EventCleaner(config)
    counts = cleaner.run()

    output_file = output_dir / "threshold_pre60s_post120s" / sample_file.name
    assert output_file.exists()
    assert counts[sample_file.name] == 3

    rows = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 4  # header + 3 cleaned rows
