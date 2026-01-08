"""Twente Cough Machine particle image velocimetry (PIV) package.
PIV Functions Module

A collection of functions for Particle Image Velocimetry (PIV) analysis of cough machine images.
Provides functions for image processing, cross-correlation, displacement
calculation, data filtering, and visualization.

Usage:
    import tcm_piv as piv

    # Read images
    imgs = piv.read_imgs(data_path, frame_numbers)
    
    # Calculate correlations
    corrs = piv.calc_corrs(imgs, n_wins=(8, 8))
    
    # Find displacements
    disps, peaks = piv.find_disps(corrs, n_wins=(8, 8))
    
    # Filter and smooth data
    disps_filtered = piv.filter_outliers('median', disps)
    disps_smooth = piv.smooth(time_array, disps_filtered)
"""

__version__ = "1.0.0"
__author__ = "Tommie Verouden"
__email__ = "t.f.l.verouden@utwente.nl"

from .io import save_backup, load_backup, read_img, read_imgs, save_cfig, init_subfolder
from .preprocessing import downsample, split_n_shift
from .correlation import calc_corr, calc_corrs, sum_corr, sum_corrs
from .displacement import find_peaks, three_point_gauss, subpixel, find_disp, find_disps
from .postprocessing import (
    filter_outliers, filter_neighbours, first_valid, strip_peaks, smooth)
from .plotting import plot_vel_comp, plot_vel_med, plot_vel_prof, plot_flow_rate
from .utils import cart2polar, get_time, disp2shift, vel2flow

__all__ = [
    # I/O functions
    'save_backup', 'load_backup', 'read_img', 'read_imgs', 'save_cfig', 'init_subfolder',

    # Preprocessing functions
    'downsample', 'split_n_shift',

    # Correlation functions
    'calc_corr', 'calc_corrs', 'sum_corr', 'sum_corrs',

    # Displacement functions
    'find_peaks', 'three_point_gauss', 'subpixel', 'find_disp', 'find_disps',

    # Postprocessing functions
    'filter_outliers', 'filter_neighbours',
    'first_valid', 'strip_peaks', 'smooth',

    # Plotting functions
    'plot_vel_comp', 'plot_vel_med', 'plot_vel_prof', 'plot_flow_rate',

    # Utility functions
    'cart2polar', 'get_time', 'disp2shift', 'vel2flow',


]

# Module-level documentation
__doc__ = """
PIV Functions Module
===================

This module provides a comprehensive set of functions for Particle Image 
Velocimetry (PIV) analysis, organized into logical submodules:

- io: Input/output operations for images and data
- preprocessing: Image preparation and windowing
- correlation: Cross-correlation calculations
- displacement: Peak finding and displacement extraction
- postprocessing: Data filtering, validation, and smoothing
- plotting: Visualization and analysis plots
- utils: General utility functions

The module is designed to support the complete PIV workflow from camera
calibration through raw image processing to final velocity field analysis 
and visualization.
"""
