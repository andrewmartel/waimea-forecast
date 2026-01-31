# Data

Place the wide-format buoy CSV here as **`wide.csv`**.

- **Default path:** `data/wide.csv` (relative to the project root).
- **Source:** The assignment provides `wide.csv` (e.g. from the repo or your Downloads folder). Copy it here or set the path via:
  - CLI: `--data /path/to/wide.csv`
  - Environment: `WAIMEA_DATA_PATH=/path/to/wide.csv`

The file must include:

- `date` — daily dates (parsed as datetime).
- `wave_height_51201h` — target (Waimea Bay / North Shore buoy wave height, metres).

Other columns (air temp, wave heights and periods from other buoys) are used as features. Missing values can be `NA` or blank; they are handled in the pipeline (see main README).
