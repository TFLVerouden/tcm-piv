from pathlib import Path
import sys
from tcm_piv import init_config as cfg

if len(sys.argv) > 1:
    config_file = Path(sys.argv[1])
else:
    config_file = Path()
cfg.read_file(config_file)
