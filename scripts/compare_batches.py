import matplotlib.pyplot as plt
from pathlib import Path
import csv
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib import ticker
from natsort import natsorted
from tcm_utils.file_dialogs import ask_directory
from tcm_utils.cvd_check import set_cvd_friendly_colors, get_color
from tcm_utils.plot_style import use_tcm_poster_style
from scipy.optimize import curve_fit
import cmcrameri.cm as cmc

# COLORS
colors_seq = cmc.batlow_r.resampled(7).colors
colors_cat = cmc.batlowS.colors

# FIT FUNCTIONS


def hill(x, y_max, x_c, K, n):
    dx = np.maximum(x - x_c, 0)
    return y_max * dx**n / (K**n + dx**n)


def linear(x, slope, intercept):
    return slope * x + intercept


def sqrt_fit(x, amplitude, offset):
    return amplitude * np.sqrt(x) + offset


def power_fit(x, amplitude, exponent, offset):
    return amplitude * x**exponent + offset

# Select the top-level (root) directory containing subfolders with flow rate summaries
# root_path = ask_directory(
#     key="compare_batches_root",
#     title="Select the root directory",
# )


root_path = "/Users/tommieverouden/Documents/Data/PIV/260820_piv"
# root_path = "/Volumes/Data/PIV/260820_piv"

if root_path is None:
    raise RuntimeError("No root selected; aborting.")

root_path = Path(root_path)

# Search (recursively) for all CSV files in the root directory that do not start with a period and end with "_flow_rate_summary.csv"
csv_files = [
    p
    for step_dir in root_path.glob("step*")
    if step_dir.is_dir()
    for p in step_dir.rglob("*_flow_rate_summary.csv")
    if p.is_file() and not p.name.startswith(".")
]

# For each csv file, read the data and store it in a dictionary with the pressure and current as the key
data_dict = {}
for csv_file in csv_files:
    # Read the row where row_type == "all" and store the data in the dictionary
    with csv_file.open("r", newline="", encoding="utf-8") as fp:
        # print(f"Reading {csv_file}")
        rows = list(csv.DictReader(fp))
        all_row = next((row for row in rows if row["row_type"] == "all"), None)
        if all_row is not None:
            # Extract the columns pressure_bar, current_mA, diff_med_L_s, diff_std_L_s, delay_avg_ms, delay_std_ms
            pressure = float(all_row["pressure_bar"])
            pressure_med = float(all_row["step_pressure_bar"])
            pressure_std = float(all_row["step_pressure_std_bar"])
            current = float(all_row["current_mA"])
            diff_med = float(all_row["diff_med_L_s"])
            diff_std = float(all_row["diff_std_L_s"])
            base_med = float(all_row["base_med_L_s"])
            base_std = float(all_row["base_std_L_s"])
            delay_avg = float(all_row["delay_avg_ms"])
            delay_std = float(all_row["delay_std_ms"])
            velocity_med = [
                float(all_row[f"winx0_winy{j}_vx_med_m_s"]) for j in range(20)]
            velocity_std = [
                float(all_row[f"winx0_winy{j}_vx_std_m_s"]) for j in range(20)]
            window_locations = [
                float(all_row[f"winx0_winy{j}_win_pos_y_mm"]) for j in range(20)]
            data_dict.setdefault(pressure, {})[current] = {
                "pressure_med": pressure_med,
                "pressure_std": pressure_std,
                "flow": diff_med,
                "flow_std": diff_std,
                "base_flow": base_med,
                "base_flow_std": base_std,
                "delay": delay_avg,
                "delay_std": delay_std,
                "velocity_med": velocity_med,
                "velocity_std": velocity_std,
                "window_locations": window_locations,
            }

# Sort the dictionary by pressure, then current
data_dict = {
    pressure: dict(sorted(current_data.items()))
    for pressure, current_data in sorted(data_dict.items())
}

# Do a quick check by recalculating the flow rate from the velocity profiles and comparing to the flow rate in the summary CSV files
depth_mm = 10
image_height_mm = (1024 - 2*59) * 0.022101482641127258

# Calculate the ratio of the average of the velocity profile and the maximum
fig, ax = plt.subplots(figsize=(7, 6))

for i, pressure in enumerate(sorted(data_dict)):
    currents = sorted(data_dict[pressure])
    flow_rates = np.asarray(
        [data_dict[pressure][current]["flow"] for current in currents])
    flow_rates_std = np.asarray(
        [data_dict[pressure][current]["flow_std"] for current in currents])
    ratios = np.asarray([
        np.mean(data_dict[pressure][current]["velocity_med"]) /
        np.max(data_dict[pressure][current]["velocity_med"])
        for current in currents])

    currents = np.asarray(currents)

    plt.errorbar(x=flow_rates, y=100*ratios,
                 xerr=flow_rates_std,
                 fmt="o",
                 #  linestyle=":",
                 #  linewidth=1.5,
                 markersize=4,
                 capsize=3,
                 label=f"{pressure} bar",
                 color=colors_seq[i],
                 )

# Make a linear fit through all data points to determine the correction factor as a function of flow rate
p0_linear = [-0.003, 0.94]
popt_linear, pcov_linear = curve_fit(
    linear,
    flow_rates,
    ratios,
    p0=p0_linear,
    sigma=None,
    absolute_sigma=True,
    maxfev=50000,
)

# Parameter values
slope, intercept = popt_linear

# 1-sigma parameter uncertainties
parameter_errors = np.sqrt(np.diag(pcov_linear))
slope_err, intercept_err = parameter_errors

print("\n=== LINEAR FIT ===")
print("To correct the flow rate for the fact that the velocity profile is not "
      "flat in the vertical direction.")
print(f"factor = {intercept:.4f} - {-slope:.4f} Q_flat")
print(f"  slope = {slope:.8f} ± {slope_err:.8f} s/L")
print(f"  intercept = {intercept:.8f} ± {intercept_err:.8f}")
print("Note that this correction is NOT applied in the plots, since all "
      "calibration was done with the uncorrected flow rates. Only in the cough "
      "machine itself, it is used to input the correct flow rate.")

flow_rates_fit = np.linspace(-5, 20, 200)

ax.plot(flow_rates_fit, 100*linear(flow_rates_fit, slope, intercept),
        color="black", linestyle="--", linewidth=1.5, label=rf"${intercept:.2f} - {100*-slope:.2f}Q_\mathrm{{flat}}$")

plt.xlabel("\"Flat\" flow rate (L/s)")
plt.ylabel("Ratio of average to max. velocity (%)")
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(85, 100)
ax.grid(True, linestyle=":", alpha=0.5)
plt.title("Determining correction factor\nfor velocity profile not being flat in the vertical direction")
plt.legend()
plt.draw()
fig.savefig(str(root_path) + "/velocity_profile_correction_factor.pdf", dpi=200)


def flat_to_corrected_flow_rate(flow_rate_flat: float,
                                slope: float = -0.00341232,
                                intercept: float = 0.94534592) -> float:
    """Convert the flow rate calculated with the vertically flat velocity profile assumption to the corrected flow rate."""
    return (intercept + slope * flow_rate_flat) * flow_rate_flat


def corrected_to_flat_flow_rate(flow_rate_corrected: float,
                                slope: float = -0.00341232,
                                intercept: float = 0.94534592) -> float:
    """Convert the corrected flow rate to the flow rate calculated with the vertically flat velocity profile assumption."""
    return (-intercept+np.sqrt(intercept**2 + 4*slope*flow_rate_corrected))/(2*slope)


# HILL FIT

fit_pressure = 1.5
fit_currents = sorted(data_dict[fit_pressure])
fit_flow_rates = [data_dict[fit_pressure][current]["flow"]
                  for current in fit_currents]
fit_sigma = np.asarray([
    data_dict[fit_pressure][current]["flow_std"]
    for current in fit_currents
])

I = np.asarray(fit_currents)
Q = np.asarray(fit_flow_rates)
sigma = fit_sigma

p0_hill = [
    np.max(Q),
    11.5,
    2.5,
    3.0,
]

popt_hill, pcov_hill = curve_fit(hill, I, Q, p0=p0_hill,
                                 sigma=sigma, absolute_sigma=True,
                                 bounds=(
                                     [0, 8, 0.01, 0.1],
                                     [20, 12.5, 20, 10],
                                 ),
                                 maxfev=50000,
                                 )

# Parameter values
Q_max, I_c, K, n = popt_hill

# 1-sigma parameter uncertainties
parameter_errors = np.sqrt(np.diag(pcov_hill))
Q_max_err, I_c_err, K_err, n_err = parameter_errors

# Chi-squared
residuals = Q - hill(I, *popt_hill)
chi2 = np.sum((residuals / sigma)**2)

# Degrees of freedom and reduced chi-squared
dof = len(Q) - len(popt_hill)
reduced_chi2 = chi2 / dof

print("\n=== HILL FIT ===")
print("Here we relate current to flow rate for the pressure with the most"
      " data points (1.5 bar).")
print("Q = Q_max * (I - I_c)^n / (K^n + (I - I_c)^n)")
print(f"  Q_max = {Q_max:.8f} ± {Q_max_err:.8f} L/s")
print(f"  I_c   = {I_c:.8f} ± {I_c_err:.8f} mA")
print(f"  K     = {K:.8f} ± {K_err:.8f} mA")
print(f"  n     = {n:.8f} ± {n_err:.8f}")
print("Fit goodness:")
print(f"  chi²  = {chi2:.3f}")
print(f"  dof   = {dof}")
print(f"  reduced chi² = {reduced_chi2:.3f}")

set_cvd_friendly_colors(first_color="#000000")

# ==============================================================================
# PLOT 1
# ==============================================================================

fig, ax = plt.subplots(figsize=(7, 6))

max_flow_points = []
qmax_fit_points = []

print("The other pressures are fitted with the same "
      "I_c, K, and n values (to keep the shape the same), but varying Q_max:")

# For each unique pressure value
for i, pressure in enumerate(sorted(data_dict)):

    # Generate an array of current values
    currents = sorted(data_dict[pressure])
    flow_rates = [data_dict[pressure][current]["flow"]
                  for current in currents]
    current_sigma = np.asarray([
        data_dict[pressure][current]["flow_std"]
        for current in currents
    ])

    currents = np.asarray(currents)
    flow_rates = np.asarray(flow_rates)

    max_flow_idx = int(np.argmax(flow_rates))
    max_flow_points.append(
        (
            pressure,
            data_dict[pressure][currents[max_flow_idx]]["pressure_med"],
            data_dict[pressure][currents[max_flow_idx]]["pressure_std"],
        )
    )

    # Plot these
    plt.errorbar(
        x=currents,
        y=flow_rates,
        yerr=current_sigma,
        fmt="o",
        markersize=4,
        label=f"{pressure} bar",
        capsize=3,
        color=colors_seq[i]
    )

    if pressure == fit_pressure:
        qmax_popt = np.asarray([Q_max])
        qmax_pcov = np.asarray([[Q_max_err**2]])
    else:
        qmax_popt, qmax_pcov = curve_fit(
            lambda x, q_max: hill(x, q_max, I_c, K, n),
            currents,
            flow_rates,
            p0=[np.max(flow_rates)],
            sigma=current_sigma,
            absolute_sigma=True,
            bounds=([0], [20]),
            maxfev=50000,
        )

    qmax_value = float(qmax_popt[0])
    qmax_error = float(np.sqrt(np.diag(qmax_pcov))[0])
    qmax_fit_points.append((pressure, qmax_value, qmax_error))

    x_fit = np.linspace(12, 20, 500)
    plt.plot(
        x_fit,
        hill(x_fit, qmax_value, I_c, K, n),
        "--",
        linewidth=1.5,
        label="_nolegend_",
        color=colors_seq[i]
    )

    print(f"  {pressure} bar fit: Q_max = {qmax_value:.4f} ± {qmax_error:.4f} L/s")

plt.xlabel("Current (mA)")
plt.ylabel("Flow rate (L/s)")
ax.grid(True, linestyle=":", alpha=0.5)
plt.title("Step function flow rate vs. current")
plt.legend()
plt.draw()
fig.savefig(str(root_path) + "/flow_rate_vs_current.pdf", dpi=200)

# ==============================================================================
# PLOT 2
# ==============================================================================

fig_max, (ax_max, ax_max_residuals) = plt.subplots(
    2,
    1,
    figsize=(7, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

step_pressures = np.asarray([point[1] for point in max_flow_points])
step_pressure_stds = np.asarray([point[2] for point in max_flow_points])
q_max_values = np.asarray([point[1] for point in qmax_fit_points])
q_max_errors = np.asarray([point[2] for point in qmax_fit_points])

ax_max.errorbar(
    x=step_pressures,
    y=q_max_values,
    xerr=step_pressure_stds,
    yerr=q_max_errors,
    fmt="o",
    markersize=5,
    capsize=3,
    color=colors_seq[3],
    label="fit points",
)

linear_popt, linear_pcov = curve_fit(
    linear,
    step_pressures,
    q_max_values,
    sigma=q_max_errors,
    absolute_sigma=True,
)
slope, intercept = linear_popt
slope_err, intercept_err = np.sqrt(np.diag(linear_pcov))

sqrt_popt, sqrt_pcov = curve_fit(
    sqrt_fit,
    step_pressures,
    q_max_values,
    sigma=q_max_errors,
    absolute_sigma=True,
)
sqrt_amplitude, sqrt_offset = sqrt_popt
sqrt_amplitude_err, sqrt_offset_err = np.sqrt(np.diag(sqrt_pcov))

power_popt, power_pcov = curve_fit(
    power_fit,
    step_pressures,
    q_max_values,
    sigma=q_max_errors,
    absolute_sigma=True,
    p0=[1.0, 1.0, 0.0],
    bounds=([0, 0, -np.inf], [np.inf, 10, np.inf]),
)
power_amplitude, power_exponent, power_offset = power_popt
power_amplitude_err, power_exponent_err, power_offset_err = np.sqrt(
    np.diag(power_pcov))

x_line = np.linspace(np.min(step_pressures) - 0.05,
                     np.max(step_pressures) + 0.05, 200)
ax_max.plot(
    x_line,
    linear(x_line, slope, intercept),
    "--",
    linewidth=1.5,
    color=colors_cat[0],
    label=rf"$Q_\mathrm{{max}} = ({slope:.3f} \pm {slope_err:.3f}) P + ({intercept:.3f} \pm {intercept_err:.3f})$",
)

ax_max.plot(
    x_line,
    sqrt_fit(x_line, sqrt_amplitude, sqrt_offset),
    ":",
    linewidth=1.8,
    color=colors_cat[1],
    label=rf"$Q_\mathrm{{max}} = ({sqrt_amplitude:.3f} \pm {sqrt_amplitude_err:.3f}) \sqrt{{P}} + ({sqrt_offset:.3f} \pm {sqrt_offset_err:.3f})$",
)

ax_max.plot(
    x_line,
    power_fit(x_line, power_amplitude, power_exponent, power_offset),
    "-.",
    linewidth=1.8,
    color=colors_cat[3],
    label=rf"$Q_\mathrm{{max}} = ({power_amplitude:.3f} \pm {power_amplitude_err:.3f}) P^{{({power_exponent:.3f} \pm {power_exponent_err:.3f})}} + ({power_offset:.3f} \pm {power_offset_err:.3f})$",
)

print("\n=== FITS FOR MAX FLOW RATE VS PRESSURE ===")
# print("Linear fit for Q_max vs step pressure:")
# print(f"  slope     = {slope:.4f} ± {slope_err:.4f} L/s/bar")
# print(f"  intercept = {intercept:.4f} ± {intercept_err:.4f} L/s")
# print("Sqrt fit for Q_max vs step pressure:")
# print(
#     f"  amplitude = {sqrt_amplitude:.4f} ± {sqrt_amplitude_err:.4f} L/s/bar^0.5")
# print(f"  offset    = {sqrt_offset:.4f} ± {sqrt_offset_err:.4f} L/s")
print("We choose the power fit for Q_max vs step pressure:")
print("  Q_max = amplitude * P^exponent + offset")
print(
    f"  amplitude = {power_amplitude:.8f} ± {power_amplitude_err:.8f} L/s/bar^n")
print(f"  exponent  = {power_exponent:.8f} ± {power_exponent_err:.8f}")
print(f"  offset    = {power_offset:.8f} ± {power_offset_err:.8f} L/s")

linear_residuals = q_max_values - linear(step_pressures, slope, intercept)
sqrt_residuals = q_max_values - \
    sqrt_fit(step_pressures, sqrt_amplitude, sqrt_offset)
power_residuals = q_max_values - power_fit(
    step_pressures, power_amplitude, power_exponent, power_offset)

ax_max_residuals.axhline(0, color="k", linestyle="--", linewidth=1)
ax_max_residuals.errorbar(
    x=step_pressures,
    y=linear_residuals,
    yerr=q_max_errors,
    fmt="o",
    markersize=0,
    capsize=3,
    color=colors_cat[0],
    label="linear residuals",
)
ax_max_residuals.errorbar(
    x=step_pressures,
    y=sqrt_residuals,
    yerr=q_max_errors,
    fmt="s",
    markersize=0,
    capsize=3,
    color=colors_cat[1],
    label="sqrt residuals",
)
ax_max_residuals.errorbar(
    x=step_pressures,
    y=power_residuals,
    yerr=q_max_errors,
    fmt="^",
    markersize=0,
    capsize=3,
    color=colors_cat[3],
    label="power residuals",
)

# ax_max.set_xlabel("Pressure at max flow rate (bar)")
ax_max.set_ylabel(r"Fitted $Q_\mathrm{max}$ (L/s)")
ax_max.set_xlim(left=0)
ax_max.set_ylim(bottom=0)
ax_max.grid(True, linestyle=":", alpha=0.5)
ax_max.set_title("Max flow rate vs. pressure")
ax_max.legend()
ax_max_residuals.axhline(0, color="k", linestyle="--", linewidth=1)
ax_max_residuals.set_xlabel("Pressure at max flow rate (bar)")
ax_max_residuals.set_ylabel("Residuals (L/s)")
ax_max_residuals.grid(True, linestyle=":", alpha=0.5)
# ax_max_residuals.legend(loc="best")
# fig_max.tight_layout()
fig_max.savefig(str(root_path) +
                "/max_flow_rate_vs_step_pressure.pdf", dpi=200)

# ==============================================================================
# PLOT 3
# ==============================================================================

# Plot residuals for 1.5 bar fit
fig_residuals, ax_residuals = plt.subplots(figsize=(7, 3))
residuals = Q - hill(I, *popt_hill)
ax_residuals.errorbar(
    x=I,
    y=residuals,
    yerr=sigma,
    fmt="o",
    markersize=4,
    capsize=3,
    color=colors_seq[3],
    # label=label_str
)

ax_residuals.axhline(0, color="k", linestyle="--", linewidth=1)
ax_residuals.set_xlabel("Current (mA)")
ax_residuals.set_ylabel("Residuals (L/s)")
ax_residuals.grid(True, linestyle=":", alpha=0.5)
ax_residuals.set_title("Residuals of Hill fit for 1.5 bar data")
# ax_residuals.legend(loc="lower left")
fig_residuals.savefig(str(root_path) + "/fit_residuals_1-5bar.pdf", dpi=200)

# ==============================================================================
# PLOT 4
# ==============================================================================

# Plot step pressure vs flow rate, colored by current in 0.5 mA steps.
fig, ax = plt.subplots(figsize=(7, 6))

points = [
    (
        values["pressure_med"],
        values["flow"],
        values["flow_std"],
        current,
    )
    for pressure, current_data in data_dict.items()
    for current, values in current_data.items()
]
currents = sorted({current for _, _, _, current in points})
current_step = 0.5
boundaries = np.arange(
    min(currents) - current_step / 2,
    max(currents) + current_step,
    current_step,
)
cmap = cmc.batlow_r
norm = BoundaryNorm(boundaries, cmap.N)

for step_pressure, diff_med, diff_std, current in points:
    color = cmap(norm(current))
    ax.errorbar(
        x=step_pressure,
        y=diff_med,
        yerr=diff_std,
        markersize=4,
        fmt="o",
        color=color,
        ecolor=color,
        capsize=3,
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
min_current = np.floor(min(currents))
max_current = np.ceil(max(currents))
cbar = fig.colorbar(
    sm,
    ax=ax,
    ticks=np.arange(min_current, max_current + 1, 1),
)
cbar.set_label("Current (mA)")
cbar.ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
cbar.ax.tick_params(which="minor", length=3)

ax.set_xlabel("Pressure (bar)")
ax.set_xlim(left=0)
ax.set_ylabel("Flow rate (L/s)")
ax.grid(True, linestyle=":", alpha=0.5)
ax.set_title("Step function flow rate vs. tank pressure")

fig.savefig(str(root_path) + "/flow_rate_vs_pressure.pdf", dpi=200)

# ==============================================================================
# PLOT 5
# ==============================================================================

# Make a plot of the flow delay as a function of the current for each pressure, with error bars for the standard deviation of the delay
fig, ax = plt.subplots(figsize=(7, 6))

for i, pressure in enumerate(sorted(data_dict)):
    # Only plot points where the delay is non-negative
    data_dict[pressure] = {current: values for current,
                           values in data_dict[pressure].items() if values["delay"] >= 0}

    currents = np.asarray(sorted(data_dict[pressure]))
    delays = np.asarray([data_dict[pressure][current]["delay"]
                        for current in currents])
    delay_std = np.asarray(
        [data_dict[pressure][current]["delay_std"] for current in currents])

    ax.errorbar(
        currents,
        delays,
        yerr=delay_std,
        fmt="o",
        markersize=4,
        capsize=3,
        linestyle=":",
        color=colors_seq[i],
        label=f"{pressure} bar",
    )

ax.set_xlabel("Current (mA)")
ax.set_ylabel("Flow delay (ms)")
ax.grid(True, linestyle=":", alpha=0.5)
ax.set_title("Flow delay vs. current")
ax.legend()

# fig.tight_layout()
fig.savefig(str(root_path) + "/flow_delay_vs_current.pdf", dpi=200)

# ==============================================================================
# PLOT 6
# ==============================================================================

# Plot the ratio between the


# ==============================================================================
# PLOT 7
# ==============================================================================
def calculate_flow_rate_Lps(
        pressure_bar: float,
        current_mA: float,
    I_c: float = 12.4878,
    K: float = 1.9746,
    n: float = 2.1949,
    A: float = 5.1517,
    B: float = 0.7575,
    C: float = 0.7152,
) -> float | None:

    max_flow_rate_Lps = A * (pressure_bar**B) + C
    if current_mA < 12.0 or current_mA > 20.0:
        return np.nan

    if current_mA < I_c:
        return 0.0

    flow_rate_Lps = max_flow_rate_Lps * \
        ((current_mA - I_c)**n) / ((current_mA - I_c)**n + K**n)

    if flow_rate_Lps < 0 or flow_rate_Lps > max_flow_rate_Lps:
        return np.nan
    return flow_rate_Lps


def calculate_current_mA(
    pressure_bar: float,
    flow_rate_Lps: float,
    I_c: float = 12.4878,
    K: float = 1.9746,
    n: float = 2.1949,
    A: float = 5.1517,
    B: float = 0.7575,
    C: float = 0.7152,


) -> float | None:

    max_flow_rate_Lps = A * (pressure_bar**B) + C
    if flow_rate_Lps < 0 or flow_rate_Lps > max_flow_rate_Lps:
        return np.nan

    current_mA = I_c + K *\
        (flow_rate_Lps / (max_flow_rate_Lps - flow_rate_Lps))**(1 / n)

    if current_mA < 12.0 or current_mA > 20.0:
        return np.nan

    return current_mA


(A, B, C) = (power_amplitude, power_offset, power_exponent)

print("Example: flow rate at 1.5 bar and 12.0 mA:",
      calculate_flow_rate_Lps(1.5, 12.0, I_c=I_c, K=K, n=n, A=A, B=B, C=C))
print("Example: required current for 5 L/s at 1.5 bar: ",
      calculate_current_mA(1.5, 5.0, I_c=I_c, K=K, n=n, A=A, B=B, C=C))
print("Example: required current for 10 L/s at 3.0 bar: ",
      calculate_current_mA(3.0, 10.0, I_c=I_c, K=K, n=n, A=A, B=B, C=C))
print("and when taking into account the correction for the non-flat profile: ",
      calculate_current_mA(3.0, corrected_to_flat_flow_rate(10.0), I_c=I_c, K=K, n=n, A=A, B=B, C=C))

# Plot a heatmap of the flow rate for a range of currents (x axis) and pressures (y axis)
current_range = np.linspace(12.0, 20.0, 200)
pressure_range = np.linspace(0.0, 5.0, 200)

fig, ax = plt.subplots(figsize=(7, 6))
flow_rate_matrix = np.zeros((len(pressure_range), len(current_range)))
for i, pressure in enumerate(pressure_range):
    for j, current in enumerate(current_range):
        flow_rate_matrix[i, j] = calculate_flow_rate_Lps(
            pressure, current, I_c=I_c, K=K, n=n, A=A, B=B, C=C)


ax.set_xlabel("Current (mA)")
ax.set_ylabel("Pressure (bar)")
im = ax.pcolormesh(
    current_range,
    pressure_range,
    flow_rate_matrix,
    shading="auto",
    cmap=cmc.batlow,
    vmin=0, vmax=18
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Flow rate (L/s)")


fig.savefig(str(root_path) + "/heatmap_flow_rate.pdf", dpi=200)

# Plot a heatmap of the required current for a range of flow rates (x axis) and pressures (y axis)
flow_rate_range = np.linspace(0.0, 18, 200)
fig, ax = plt.subplots(figsize=(7, 6))
current_matrix = np.zeros((len(pressure_range), len(flow_rate_range)))
for i, pressure in enumerate(pressure_range):
    for j, flow_rate in enumerate(flow_rate_range):
        current_matrix[i, j] = calculate_current_mA(
            pressure, flow_rate, I_c=I_c, K=K, n=n, A=A, B=B, C=C)


ax.set_xlabel("Flow rate (L/s)")
ax.set_ylabel("Pressure (bar)")
im = ax.pcolormesh(
    flow_rate_range,
    pressure_range,
    current_matrix,
    shading="auto",
    cmap=cmc.batlow,
    vmin=12, vmax=20
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Current (mA)")

# Set colorbar limits
# cbar.set_clim(12, 20)

fig.savefig(str(root_path) + "/heatmap_current.pdf", dpi=200)

# ==============================================================================
plt.show()
