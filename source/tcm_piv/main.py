from pathlib import Path
import sys
from tcm_piv import init_config as cfg

"""CLI entrypoint for config initialization.

Usage:
    python -m tcm_piv.main path/to/config.toml

If no config path is provided, a file dialog will open.
"""


config_file: Path | None
if len(sys.argv) > 1:
    config_file = Path(sys.argv[1])
else:
    config_file = None

cfg.read_file(config_file)
