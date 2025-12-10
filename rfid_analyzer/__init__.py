"""RFID Analyzer package (OOP refactor draft)."""

from .config import CleanerConfig, load_config
from .cleaning.cleaner import EventCleaner
from .analysis.copresence import create_copresence_matrix

__all__ = ["CleanerConfig", "load_config", "EventCleaner", "create_copresence_matrix"]
