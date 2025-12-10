import argparse
import logging
from pathlib import Path

from rfid_analyzer import EventCleaner, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean RFID nest events")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config (default: config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    cleaner = EventCleaner(config)
    counts = cleaner.run()
    for name, count in counts.items():
        logger.info("%s -> %d cleaned rows", name, count)


if __name__ == "__main__":
    main()
