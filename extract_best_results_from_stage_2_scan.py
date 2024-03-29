#!/usr/bin/env python

# This script is run after "stage_2_scan.py" has generated some optimized coils.
# This script reads the results.json files in the subdirectories, plots the
# distribution of results, filters out unacceptable runs, and prints out runs that
# are Pareto-optimal.

import glob
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from paretoset import paretoset

# Initialize an empty DataFrame
df = pd.DataFrame()

results = glob.glob("*/results.json")
# df = None
for results_file in results:
    with open(results_file, "r") as f:
        data = json.load(f)

    # Wrap lists in another list
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = [value]

    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

#########################################################
# Here you can define criteria to filter out the most interesting runs.
#########################################################

succeeded = df["linking_number"] < 0.1
succeeded = np.logical_and(succeeded, df["coil_coil_distance"] > 0.03)
# succeeded = np.logical_and(succeeded, df["Jf"] < 1e-3)
# succeeded = np.logical_and(succeeded, df["max_max_curvature"] < 12)

#########################################################
# End of filtering criteria
#########################################################

df_filtered = df[succeeded]

pareto_mask = paretoset(df_filtered[["Jf", "max_max_curvature"]], sense=[min, min])
df_pareto = df_filtered[pareto_mask]

print("Best Pareto-optimal results:")
print(
    df_pareto[
        [
            "directory",
            "Jf",
            "max_max_curvature",
            "length",
            "max_mean_squared_curvature",
            "coil_coil_distance",
        ]
    ]
)
print("Directory names only:")
for dirname in df_pareto["directory"]:
    print(dirname)

#########################################################
# Plotting
#########################################################

plt.figure(figsize=(14.5, 8))
plt.rc("font", size=8)
nrows = 3
ncols = 6
markersize = 5

subplot_index = 1
plt.subplot(nrows, ncols, subplot_index)
subplot_index += 1
plt.scatter(df["Jf"], df["max_max_curvature"], c=df["length"], s=1)
plt.colorbar(label="length")
plt.scatter(
    df_filtered["Jf"],
    df_filtered["max_max_curvature"],
    c=df_filtered["length"],
    s=markersize,
)
plt.scatter(
    df_pareto["Jf"], df_pareto["max_max_curvature"], c=df_pareto["length"], marker="+"
)
plt.xlabel("Bnormal objective")
plt.ylabel("Max curvature")
plt.xscale("log")

plt.subplot(nrows, ncols, subplot_index)
subplot_index += 1
plt.scatter(
    df_filtered["length_target"],
    df_filtered["length"],
    c=df_filtered["Jf"],
    s=markersize,
    norm=colors.LogNorm(),
)
plt.colorbar(label="Bnormal objective")
plt.xlabel("length_target")
plt.ylabel("length")

plt.subplot(nrows, ncols, subplot_index)
subplot_index += 1
plt.scatter(
    df_filtered["max_curvature_threshold"],
    df_filtered["max_max_curvature"],
    c=df_filtered["Jf"],
    s=markersize,
    norm=colors.LogNorm(),
)
plt.colorbar(label="Bnormal objective")
plt.xlabel("max_curvature_threshold")
plt.ylabel("max_max_curvature")


def plot_2d_hist(field, log=False):
    global subplot_index
    plt.subplot(nrows, ncols, subplot_index)
    subplot_index += 1
    nbins = 20
    if log:
        data = df[field]
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), nbins)
    else:
        bins = nbins
    plt.hist(df[field], bins=bins, label="before filtering")
    plt.hist(df_filtered[field], bins=bins, alpha=1, label="after filtering")
    plt.xlabel(field)
    plt.legend(loc=0, fontsize=6)
    if log:
        plt.xscale("log")


# 2nd entry of each tuple is True if the field should be plotted on a log x-scale.
fields = (
    ("R1", False),
    ("order", False),
    ("length", False),
    ("length_target", False),
    ("length_weight", True),
    ("max_curvature_threshold", False),
    ("max_curvature_weight", True),
    ("max_max_curvature", False),
    ("msc_threshold", False),
    ("msc_weight", True),
    ("coil_coil_distance", False),
    ("cc_threshold", False),
    ("cc_weight", True),
)

for field, log in fields:
    plot_2d_hist(field, log)

plt.figtext(0.5, 0.995, os.path.abspath(__file__), ha="center", va="top", fontsize=6)
plt.tight_layout()
plt.show()
