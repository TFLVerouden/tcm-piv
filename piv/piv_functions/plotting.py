"""
Visualization functions for PIV analysis.

This module contains functions for creating plots and visualizations
of PIV displacement and velocity data.
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as ani
from tqdm import trange
import sys

from tcm_utils import cvd_check as cvd

from .utils import get_time
from .io import save_cfig


def plot_vel_comp(disp_glo, disp_nbs, disp_spl, res, frs, dt, proc_path=None, file_name=None, test_mode=False,
                  disp_rejected=None, **kwargs):
    # TODO Add docstring and typing
    # Might break with horizontal windows.

    # Define a time array using helper function
    time = get_time(frs, dt)

    # If lengths don't match, assume all data was supplied; slice accordingly
    if disp_glo.shape[0] != time.shape[0]:
        disp_glo = disp_glo[frs[0]:frs[-1], :, :, :]
        disp_nbs = disp_nbs[frs[0]:frs[-1], :, :, :]
        disp_spl = disp_spl[frs[0]:frs[-1], :, :, :]
        if disp_rejected is not None:
            disp_rejected = disp_rejected[frs[0]:frs[-1], :, :, :]

    # Convert displacement to velocity
    vel_glo = disp_glo * res / dt
    vel_nbs = disp_nbs * res / dt
    vel_spl = disp_spl * res / dt

    # Scatter plot vx(t)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot rejected points if provided
    if disp_rejected is not None:
        vel_rejected = disp_rejected * res / dt
        # Plot all candidate peaks for vx and vy
        ax.plot(np.tile(time[:, None] * 1000, (1, vel_rejected.shape[-2])).flatten(),
                vel_rejected[:, 0, 0, :, 1].flatten(), 'x', c='red', alpha=0.3, ms=2,
                label='vx (all candidate peaks)')
        ax.plot(np.tile(time[:, None] * 1000, (1, vel_rejected.shape[-2])).flatten(),
                vel_rejected[:, 0, 0, :, 0].flatten(), 'x', c='black', alpha=0.3, ms=2,
                label='vy (all candidate peaks)')

    ax.plot(1000 * time, vel_glo[:, 0, 0, 1], 'o', ms=4, c='gray',
            label='vx (filtered globally)')
    ax.plot(1000 * time, vel_nbs[:, 0, 0, 1], '.', ms=2, c='black',
            label='vx (filtered neighbours)')
    ax.plot(1000 * time, vel_nbs[:, 0, 0, 0], c=cvd.get_color(0),
            label='vy (filtered neighbours)')
    ax.plot(1000 * time, vel_spl[:, 0, 0, 1], c=cvd.get_color(1),
            label='vx (smoothed for 2nd pass)')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('First pass')
    ax.set(**kwargs)

    ax.legend(loc='upper right')
    ax.grid()

    if proc_path is not None and file_name is not None and not test_mode:
        # Save the figure
        save_cfig(proc_path, file_name, test_mode=test_mode, verbose=True)

    return fig, ax


def plot_vel_med(disp, res, frs, dt, proc_path=None, file_name=None, test_mode=False, **kwargs):
    # TODO Add docstring and typing
    # Might break with horizontal windows.

    # Define a time array
    time = get_time(frs, dt)

    # If lengths don't match, assume all data was supplied; slice accordingly
    if disp.shape[0] != time.shape[0]:
        disp = disp[frs[0]:frs[-1], :, :, :]

    # Convert displacement to velocity
    vel = disp * res / dt

    # Plot the median velocity in time, show the min and max as a shaded area
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate statistics with warning suppression for all-NaN slices
    with np.errstate(invalid='ignore'):
        # Plot vy (vertical velocity)
        med_vy = np.nanmedian(vel[:, :, :, 0], axis=(1, 2))
        min_vy = np.nanmin(vel[:, :, :, 0], axis=(1, 2))
        max_vy = np.nanmax(vel[:, :, :, 0], axis=(1, 2))

        # Plot vx (horizontal velocity)
        med_vx = np.nanmedian(vel[:, :, :, 1], axis=(1, 2))
        min_vx = np.nanmin(vel[:, :, :, 1], axis=(1, 2))
        max_vx = np.nanmax(vel[:, :, :, 1], axis=(1, 2))

    # Plot vy (vertical velocity)
    ax.plot(time * 1000, med_vy, label='Median vy')

    ax.fill_between(time * 1000, min_vy, max_vy, alpha=0.3, label='Min/max vy')

    # Plot vx (horizontal velocity)
    ax.plot(time * 1000, med_vx, label='Median vx')
    ax.fill_between(time * 1000, min_vx, max_vx, alpha=0.3, label='Min/max vx')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Median velocity in time')
    ax.set(**kwargs)

    ax.legend(loc='upper right')
    ax.grid()

    if proc_path is not None and file_name is not None and not test_mode:
        # Save the figure
        save_cfig(proc_path, file_name, test_mode=test_mode, verbose=True)

    return fig, ax


def plot_vel_Gupta(disp, res, frs, dt, proc_path=None, file_name=None, test_mode=False, **kwargs):
    # TODO Add docstring and typing
    # Might break with horizontal windows.

    # Define a time array
    time = get_time(frs, dt)
    # Gupta PLOTTER, Abe
    Flowrate_Gupta, time_Gupta = Gupta_plotter("Male", 70, 1.90)
    A_coughmachine = 2e-4  # m^2

    # v (m/s) = Q (L/s) / A(m), divide Q by a 1000
    v_Gupta = Flowrate_Gupta / A_coughmachine / 1000
    ####

    # If lengths don't match, assume all data was supplied; slice accordingly
    if disp.shape[0] != time.shape[0]:
        disp = disp[frs[0]:frs[-1], :, :, :]

    # Convert displacement to velocity
    vel = disp * res / dt

    # Plot the median velocity in time, show the min and max as a shaded area
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate statistics with warning suppression for all-NaN slices
    with np.errstate(invalid='ignore'):
        # Plot vy (vertical velocity)
        med_vy = np.nanmedian(vel[:, :, :, 0], axis=(1, 2))
        min_vy = np.nanmin(vel[:, :, :, 0], axis=(1, 2))
        max_vy = np.nanmax(vel[:, :, :, 0], axis=(1, 2))

        # Plot vx (horizontal velocity)
        med_vx = np.nanmedian(vel[:, :, :, 1], axis=(1, 2))
        min_vx = np.nanmin(vel[:, :, :, 1], axis=(1, 2))
        max_vx = np.nanmax(vel[:, :, :, 1], axis=(1, 2))

    # Plot vy (vertical velocity)
    ax.plot(time * 1000, med_vy, label='Median vy')
    ax.plot(time_Gupta*1000, v_Gupta, label="Gupta", c='k')
    ax.fill_between(time * 1000, min_vy, max_vy, alpha=0.3, label='Min/max vy')

    # Plot vx (horizontal velocity)
    ax.plot(time * 1000, med_vx, label='Median vx')
    ax.fill_between(time * 1000, min_vx, max_vx, alpha=0.3, label='Min/max vx')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Median velocity in time')
    ax.set(**kwargs)

    ax.legend()
    ax.grid()
    plt.show()

    # if proc_path is not None and file_name is not None and not test_mode:
    # Save the figure
    # save_cfig(proc_path, file_name, test_mode=test_mode, verbose=True)

    return fig, ax


def plot_vel_prof(disp, res, frs, dt, win_pos,
                  mode="random", proc_path=None, file_name=None, subfolder=None, test_mode=False,
                  disp_rejected=None, frame_skip=1, avg_start_time=None, avg_end_time=None, **kwargs):

    # Define a time array
    n_corrs = disp.shape[0]
    time = get_time(frs, dt)

    # Convert displacement to velocity
    vel = disp * res / dt

    # Handle rejected data if provided
    vel_rejected = None
    if disp_rejected is not None:
        vel_rejected = disp_rejected * res / dt

    # Raise error if one tries to make a video, but proc_path is not specified
    if mode == "video":
        if test_mode:
            return
        elif proc_path is None or file_name is None:
            raise ValueError(
                "proc_path and file_name must be specified to create a video.")

    # Set up save path if subfolder is specified
    if proc_path is not None and subfolder is not None and not test_mode:
        from .io import init_subfolder
        save_path = init_subfolder(proc_path, subfolder, debug=test_mode)
    else:
        if mode == "all":
            # Error: we don't want to save all images to the root folder
            raise RuntimeWarning(
                f"Are you sure you want to save {n_corrs} files directly to {proc_path}?")
        save_path = proc_path

    # Determine which frames to process
    if mode == "random":
        np.random.seed(42)  # For reproducible results
        frames_to_plot = np.sort(np.random.choice(
            n_corrs, size=min(10, n_corrs), replace=False))
    elif mode == "all" or mode == "video":
        if mode == "video" and frame_skip > 1:
            # Skip frames: take every frame_skip-th frame
            frames_to_plot = range(frame_skip-1, n_corrs, frame_skip)
        else:
            frames_to_plot = range(n_corrs)
    elif mode == "average":
        # For average mode, determine frames within the specified time range
        if avg_start_time is None or avg_end_time is None:
            raise ValueError(
                "avg_start_time and avg_end_time must be specified for average mode.")

        # Find frame indices corresponding to the time range
        start_idx = np.argmin(np.abs(time - avg_start_time))
        end_idx = np.argmin(np.abs(time - avg_end_time))
        if end_idx <= start_idx:
            raise ValueError(
                "avg_end_time must be greater than avg_start_time.")

        frames_to_plot = range(start_idx, end_idx + 1)
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Use 'video', 'all', 'random', or 'average'.")

    # Set up video writer if needed
    if mode == "video":
        fig_video, ax_video = plt.subplots(figsize=(6, 4))
        plt.tight_layout()  # Minimize borders for video frames
        writer = ani.FFMpegWriter(fps=10)
        video_path = os.path.join(proc_path, file_name+'.mp4')
        video_context = writer.saving(fig_video, video_path, dpi=300)
        frames_iter = trange(len(frames_to_plot), desc='Rendering video     ')
    else:
        video_context = None
        frames_iter = frames_to_plot

    # Common plotting function
    def plot_frame(frame_idx, ax):
        y_pos = win_pos[:, 0, 0] * res * 1000
        vx = vel[frame_idx, :, 0, 1]
        vy = vel[frame_idx, :, 0, 0]

        ax.plot(vx, y_pos, '-o', c=cvd.get_color(1), label='vx')
        ax.plot(vy, y_pos, '-o', c=cvd.get_color(0), label='vy')

        # Plot rejected points if provided and enabled
        if vel_rejected is not None:
            vx_rejected = vel_rejected[frame_idx,
                                       :, :, :, 1]  # All peaks for vx
            vy_rejected = vel_rejected[frame_idx,
                                       :, :, :, 0]  # All peaks for vy

            # Create y positions for each peak (repeat y_pos for each peak)
            n_peaks = vel_rejected.shape[-2]
            y_pos_expanded = np.repeat(y_pos[:, np.newaxis], n_peaks, axis=1)

            # Plot all rejected peaks as smaller, transparent points
            ax.scatter(vx_rejected.flatten(), y_pos_expanded.flatten(),
                       c='black', s=10, alpha=0.5, marker='x', label='Rejected vx')
            ax.scatter(vy_rejected.flatten(), y_pos_expanded.flatten(),
                       c='red', s=10, alpha=0.5, marker='x', label='Rejected vy')

        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('y position (mm)')
        ax.set_title(
            f'Velocity profiles at frame {frame_idx + 1} ({time[frame_idx] * 1000:.0f} ms)')
        ax.legend(loc='upper right')
        ax.grid()
        ax.set(**kwargs)

        # Minimize white borders
        plt.tight_layout()

    # Process frames
    if mode == "average":
        # Average mode: compute median and IQR over the specified time range
        y_pos = win_pos[:, 0, 0] * res * 1000

        # Extract velocity data for the specified frames
        vx_data = vel[frames_to_plot, :, 0, 1]  # Shape: (n_frames, n_windows)
        vy_data = vel[frames_to_plot, :, 0, 0]  # Shape: (n_frames, n_windows)

        # Compute median and interquartile range across frames
        vx_median = np.nanmedian(vx_data, axis=0)
        vx_q25 = np.nanpercentile(vx_data, 25, axis=0)
        vx_q75 = np.nanpercentile(vx_data, 75, axis=0)

        vy_median = np.nanmedian(vy_data, axis=0)
        vy_q25 = np.nanpercentile(vy_data, 25, axis=0)
        vy_q75 = np.nanpercentile(vy_data, 75, axis=0)

        # Create figure for average profile
        fig, ax = plt.subplots(figsize=(7, 4))

        # Plot median profiles
        ax.plot(vx_median, y_pos, '-o', c=cvd.get_color(1),
                label='vx (median)', linewidth=2)
        ax.plot(vy_median, y_pos, '-o', c=cvd.get_color(0),
                label='vy (median)', linewidth=2)

        # Add shaded regions for interquartile range
        ax.fill_betweenx(y_pos, vx_q25, vx_q75,
                         color=cvd.get_color(1), alpha=0.3, label='vx IQR')
        ax.fill_betweenx(y_pos, vy_q25, vy_q75,
                         color=cvd.get_color(0), alpha=0.3, label='vy IQR')

        # Optionally plot rejected data (average of all rejected points in time range)
        if disp_rejected is not None:
            raise NotImplementedError(
                "Plotting rejected data in average mode is not implemented.")

        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('y position (mm)')
        ax.set_title(
            f'Median velocity profiles ({avg_start_time*1000:.0f}-{avg_end_time*1000:.0f} ms, n={len(frames_to_plot)})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid()
        ax.set(**kwargs)
        plt.tight_layout()

        # Save if path is specified
        if proc_path is not None and file_name is not None and not test_mode:
            save_cfig(proc_path, file_name + "_median",
                      test_mode=test_mode, verbose=True)

        return fig, ax

    elif video_context is not None:
        # Video mode
        with video_context:
            for i in frames_iter:
                frame_idx = frames_to_plot[i]  # Get the actual frame index
                ax_video.clear()
                plot_frame(frame_idx, ax_video)
                writer.grab_frame()
        plt.close(fig_video)
        print(f"Video saved to {video_path}")
    else:
        # Plot mode (random or all)
        for frame_idx in frames_iter:
            fig, ax = plt.subplots(figsize=(6, 4), clear=True, num=99)
            plot_frame(frame_idx, ax)

            # Save if path is specified
            if save_path is not None:
                save_cfig(save_path, file_name +
                          f"_{frame_idx:04d}", test_mode=test_mode)

                # Close figure
                plt.close(fig)


def plot_flow_rate(q, frs, dt, q_model=None, t_model=None, proc_path=None, file_name=None, test_mode=False,
                   frame_skip=1, plot_model=True, **kwargs):

    # Define a time array
    time = get_time(frs, dt)

    # If lengths don't match, assume all data was supplied; slice accordingly
    if q.shape[0] != time.shape[0]:
        q = q[frs[0]:frs[-1]]

    # Apply frame skipping if specified
    if frame_skip > 1:
        skip_indices = slice(0, len(time), frame_skip)
        time = time[skip_indices]
        q = q[skip_indices]

    # Plot the flow rate in time
    fig, ax = plt.subplots(figsize=(6, 4))

    # If a model is provided and plotting is enabled, plot it
    if plot_model and q_model is not None and t_model is not None:
        ax.plot(t_model * 1000, q_model,
                label='Gupta et al., 2009', c=cvd.get_color(2))

    ax.plot(time * 1000, q * 1000, label='Flow rate')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Flow rate (L/s)')
    ax.set_title('Flow rate in time')
    ax.set(**kwargs)

    if plot_model:
        ax.legend(loc='upper right')

    ax.grid()

    # Minimize white borders
    plt.tight_layout()

    if proc_path is not None and file_name is not None and not test_mode:
        # Save the figure
        save_cfig(proc_path, file_name, test_mode=test_mode, verbose=True)

    return fig, ax
