# Agent Instructions for `rfid-analyzer`

These guidelines define how to extend the project going forward.

## Design
- Prefer small, composable classes with single responsibility (e.g., event parsing, cleaning logic, orchestration).
- Use type hints and dataclasses for value objects (events, sessions, config models).
- Keep modules cohesive: domain models in `models.py`, services in dedicated subpackages, config in `config.py`.
- Avoid global state; pass dependencies explicitly.
- Make I/O (paths, thresholds) configurable via YAML and overridable via function/CLI parameters.

## Cleaning Logic Baseline
- Input CSVs are semicolon/`,` separated with columns: date, time, action (`IN`/`OUT`), chicken_id, nest_id.
- Two thresholds in seconds: `threshold_pre_ovodeposition`, `threshold_post_ovodeposition`.
- Events before the ovodeposition start date use the pre threshold; events on/after use the post threshold.
- If a hen exits a nest and re-enters the same nest within the threshold, drop the exit and the re-entry (treat as continuous presence). Apply this per hen+nest, sorted by datetime, even when other hens appear between those rows in the original CSV.

## Configuration
- Default config file: `config.yaml` at repo root. Keys: `input_folder`, `output_folder`, `ovodeposition_start_date` (`YYYY-MM-DD`), `threshold_pre_ovodeposition`, `threshold_post_ovodeposition`, `datetime_format` (default `%d/%m/%Y %H:%M:%S`).
- New modules should load config via a typed loader; validate required keys and paths.

## Testing & Examples
- Use `pytest` in `tests/` with small, deterministic fixtures; cover threshold merge logic and edge cases.
- Provide a minimal usage example (script or markdown) showing how to run the module with a sample config.

## Code Style
- Follow PEP 8; keep functions short; add concise comments only where logic is non-obvious.
- Log info-level steps and warning-level data issues; no prints.
- Keep outputs deterministic; avoid hidden filesystem side effects.
