import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle


def get_vbounds(data:xr.DataArray, vbounds_str:str):
	if vbounds_str == "min:min+4std":
		vmin = data.min().__float__()
		vmax = vmin + 4 * data.std().__float__()
	elif vbounds_str == "-2std:2std":
		vmax = 2 * data.std().__float__()
		vmin = -vmax
	elif vbounds_str == "0:2std":
		vmax = 2 * data.std().__float__()
		vmin = 0
	elif vbounds_str == "0:4std":
		vmax = 4 * data.std().__float__()
		vmin = 0
	elif vbounds_str:
		vmin, vmax = map(int, vbounds_str.split(':'))
	else:
		# bounds must be known to sync the color bar of all subplots together
		raise ValueError()
	
	return vmin, vmax

def axis_to_rectangle(ax:Axes, **rect_kwargs):
	xmin, xmax = ax.get_xlim()
	ymin, ymax = ax.get_ylim()
	width = xmax - xmin
	height = ymax - ymin
	return Rectangle((xmin, ymin), width, height, **rect_kwargs)

def plot_quantile_heatmap(filepath_template:str, data:xr.DataArray, xtick_indices:list[int], ytick_indices:list[int], vbounds_str:str, cbar_label:str, prefix:str):
	metric_key:str = data.coords["metric"].values.item()
	filepath = filepath_template.format(label=metric_key)
	title_name = f"{prefix.upper()} {metric_str_to_title_str(metric_key)}"

	loss_names:list[str] = data.coords["loss_name"].values
	epx_keys:np.ndarray = data.coords["epx"].values
	quantile_keys:np.ndarray = data.coords["exclusive_quantile"].values

	fig = plt.figure(figsize=(6.5, 3), dpi=600)
	fig.suptitle(title_name)
	fig.supxlabel("E[px]")
	fig.supylabel("EQ")

	xticks = [epx_keys[i] for i in xtick_indices]
	xtick_labels = [str(xtick) for xtick in xticks]
	
	yticks = [quantile_keys[j] for j in ytick_indices]
	ytick_labels = [f"{ytick:.2f}" for ytick in yticks]

	vmin, vmax = get_vbounds(data, vbounds_str)
	n_losses = len(loss_names)

	sns_ax_x, sns_ax_w = 0.075, 0.9 / n_losses
	sns_ax_y, sns_ax_h = 0.2, 0.65
	sns_ax_wpad = -0.05

	cbar_y, cbar_h = sns_ax_y, sns_ax_h
	cbar_w = 0.015

	sns_axes:list[Axes] = []
	for _ in range(n_losses):
		sns_axes.append(fig.add_axes((sns_ax_x, sns_ax_y, sns_ax_w, sns_ax_h)))
		sns_ax_x += sns_ax_w + sns_ax_wpad
	
	sns_ax_x += 0.5 * abs(sns_ax_wpad)
	cbar_ax = fig.add_axes((sns_ax_x, cbar_y, cbar_w, cbar_h))
	cbar_ax.tick_params(labelsize=8)

	sns_ax:Axes
	for loss_name, sns_ax, data in zip(loss_names, sns_axes, data):
		heatmap_ax = sns.heatmap(data.T, square=True, ax=sns_ax, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax, center=0, cbar_kws={'label': cbar_label})
		# Data is transposed as the data is treated as an image (index 0: row, index 1: col).
		# 0th axis is the 'epx' axis, which should be the x-axis.
		sns_ax.tick_params(labelsize=8)
		sns_ax.set_title(loss_name)
		sns_ax.set_xticks(xtick_indices, xtick_labels, rotation=90)
		sns_ax.set_yticks(ytick_indices)
		sns_ax.invert_yaxis()
	
	sns_axes[0].set_yticklabels(ytick_labels)

	
	for spine in cbar_ax.spines.values():
		spine.set_edgecolor('black')
		spine.set_linewidth(1)
	
	plt.savefig(filepath, bbox_inches='tight', pad_inches=0.001)
	print(filepath)

def metric_str_to_title_str(x:str):
	x = x.replace('_', ' ')
	x = x.replace('measured', 'Measured')
	x = x.replace('counted', 'Counted')
	x = x.replace('symm', 'Symm')
	x = x.replace('fp', 'FP')
	x = x.replace('fn', 'FN')
	x = x.replace('diffs', 'MBE')
	x = x.replace('qlower', 'Q-Lower')
	x = x.replace('qupper', 'Q-Upper')
	x = x.replace('sums', 'MAE')
	return x

def main(metrics_dir, output_dir, mm_name):
	os.makedirs(output_dir, exist_ok=True)
	
	filename_template = f"{output_dir}/{mm_name.lower()}-{{label}}.png"
	xr_analytics = xr.load_dataarray(f"{metrics_dir}/{mm_name}-test_analytics.nc")

	epx_indices = [0, 24, 49, 74, 98]
	eq_indices = [0, 24, 49, 74, 99, 124, 149, 174, 198]

	quantile_vbounds_strs = [
		"min:min+4std", "min:min+4std", "0:1", "0:1",
		"0:4std", "-2std:2std", "-1:1", "0:4std", "-2std:2std",  "-1:1",
	]

	quantile_cbar_labels = [
		"T1 Error", "T2 Error", "T1 Error Likelihood", "T2 Error Likelihood",
		"MAE", "MBE", "TE %", "MAE", "MBE", "TE %",
	]

	for metric_results, quantile_vbounds_str, quantile_cbar_label in zip(xr_analytics, quantile_vbounds_strs, quantile_cbar_labels):
		plot_quantile_heatmap(filename_template, metric_results, epx_indices, eq_indices, quantile_vbounds_str, quantile_cbar_label, mm_name)


if __name__ == '__main__':
	print(f"Running {__file__}...")
	
	mm_names = ["smm", "imm"]

	metrics_dir = "analytics/metrics"
	output_dir = "analytics/heatmaps"

	for mm_name in mm_names:
		main(metrics_dir, output_dir, mm_name)
