# tcm-piv
Twente Cough Machine - Particle Image Velocimetry

## Configuration

Configuration is TOML-based (Python >= 3.11).

Set `source.frames_to_use` to `"all"`, an explicit list of frame numbers, or an
inclusive range such as `frames_to_use = [0, 1500]`. A two-number array is
interpreted as a range.

- Example config: [source/tcm_piv/config/config.toml](source/tcm_piv/config/config.toml)
- Packaged defaults: [source/tcm_piv/config/default_config.toml](source/tcm_piv/config/default_config.toml)


## Todo
- swap order of peak detection and global filter?
- save log files (basically all print statements saved with timestamps?)
- fix weird error in filter_neighbours