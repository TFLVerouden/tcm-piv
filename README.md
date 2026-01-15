# tcm-piv
Twente Cough Machine - Particle Image Velocimetry

## Configuration

Configuration is TOML-based (Python >= 3.11).

- Example config: [source/tcm_piv/config/config.toml](source/tcm_piv/config/config.toml)
- Packaged defaults: [source/tcm_piv/config/default_config.toml](source/tcm_piv/config/default_config.toml)


## Todo
- range option for frames_to_use
- swap order of peak detection and global filter?
- save log files (basically all print statements saved with timestamps?)
- fix resume run (not working currently)
- fix weird error in filter_neighbours