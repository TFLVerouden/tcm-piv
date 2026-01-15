"""Twente Cough Machine particle image velocimetry (PIV) package.
PIV Functions Module

A collection of functions for Particle Image Velocimetry (PIV) analysis of cough machine images.
Provides functions for image processing, cross-correlation, displacement
calculation, data filtering, and visualization.

Usage:
    import tcm_piv as piv

    # (Low-level functional API is available by importing submodules.
    # The supported end-to-end pipeline lives in `tcm_piv.run`.)
"""

__version__ = "1.0.0"
__author__ = "Tommie Verouden"
__email__ = "t.f.l.verouden@utwente.nl"

from .preprocessing import downsample, split_n_shift
from .correlation import calc_corr, calc_corrs, sum_corr, sum_corrs
from .displacement import find_peaks, three_point_gauss, subpixel_correction, find_disp, find_disps
from .postprocessing import (
    filter_outliers, filter_neighbours, first_valid, strip_peaks, smooth)
from .utils import cart2polar, get_time, disp2shift, vel2flow

__all__ = [
    # Preprocessing functions
    'downsample', 'split_n_shift',

    # Correlation functions
    'calc_corr', 'calc_corrs', 'sum_corr', 'sum_corrs',

    # Displacement functions
    'find_peaks', 'three_point_gauss', 'subpixel_correction', 'find_disp', 'find_disps',

    # Postprocessing functions
    'filter_outliers', 'filter_neighbours',
    'first_valid', 'strip_peaks', 'smooth',

    # Utility functions
    'cart2polar', 'get_time', 'disp2shift', 'vel2flow',


]

# Module-level documentation
__doc__ = """
PIV Functions Module
===================

This module provides a comprehensive set of functions for Particle Image 
Velocimetry (PIV) analysis, organized into logical submodules:

- preprocessing: Image preparation and windowing
- correlation: Cross-correlation calculations
- displacement: Peak finding and displacement extraction
- postprocessing: Data filtering, validation, and smoothing
- utils: General utility functions

The module is designed to support the complete PIV workflow from camera
calibration through raw image processing to final velocity field analysis 
and visualization.
"""
