from pathlib import Path
import sys
from tcm_piv import init_config as cfg


config_file: Path | None
if len(sys.argv) > 1:
    config_file = Path(sys.argv[1])
else:
    config_file = None

cfg.read_file(config_file)
