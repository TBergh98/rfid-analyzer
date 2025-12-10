"""Minimal example to clean RFID data.

Usage (from repo root):
    .venv/Scripts/python -m rfid_analyzer.cli --config rfid-analyzer/config.yaml
"""

from pathlib import Path
import sys

# Ensure package root is on sys.path when running this file directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rfid_analyzer import EventCleaner, load_config


def main() -> None:
    config = load_config(Path("rfid-analyzer\\config.yaml"))
    cleaner = EventCleaner(config)
    cleaner.run()


if __name__ == "__main__":
    main()
