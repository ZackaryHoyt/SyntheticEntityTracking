import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from scipy.spatial.distance import jensenshannon

from dataset_util.arr_util import l1_normalization


def build_js_dist_mat(a:xr.DataArray, b:xr.DataArray, axis:str):
	assert a.shape == b.shape

	loss_names:list[str] = a.coords["loss_name"].values
	axis_values:list = a.coords[axis].values

	n_loss_names = len(loss_names)

	js_dist_mat = np.zeros(shape=(n_loss_names, n_loss_names))
	for i0, i1 in itertools.product(range(n_loss_names), range(n_loss_names)):
		a_i = a.sel(loss_name=loss_names[i0])
		b_i = b.sel(loss_name=loss_names[i1])

		js_distances = []
		for dim_value in axis_values:
			kwd = { axis: dim_value }
			a_ij = a_i.sel(**kwd)
			b_ij = b_i.sel(**kwd)
			js_distances.append(jensenshannon(l1_normalization(a_ij.to_numpy()), l1_normalization(b_ij.to_numpy())))

		js_dist_mat[i0, i1] = np.mean(js_distances)
	
	return js_dist_mat

def plot_js_dist_mat(js_dist_mat:np.ndarray, ticklabels:list[str], title:str, xlabel:str, ylabel:str, filename_suffix:str):
	fig = plt.figure(figsize=(3.5, 3.5), dpi=300)
	ax_heatmap = sns.heatmap(js_dist_mat,
		cmap='viridis',
		annot=True,
		fmt='.3f',
		xticklabels=ticklabels,
		yticklabels=ticklabels,
		vmin=0,
		vmax=0.3,
		square=True,
		lw=0.2,
		annot_kws={"size": 10},
		cbar_kws={"shrink": 0.7, "orientation": "horizontal", "pad": 0.046},
	)
	
	cbar = ax_heatmap.collections[0].colorbar
	if cbar is not None:
		for spine in cbar.ax.spines.values():
			spine.set_edgecolor('black')
			spine.set_linewidth(1)

	ax_heatmap.tick_params(labelbottom=False, labeltop=True, bottom=False, top=True)
	ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=0, ha='right')
	ax_heatmap.xaxis.set_label_position('top')
	
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)

	plt.tight_layout()
	# plt.show()
	filename = f"{output_dir}/js-{filename_suffix}.png"
	fig.savefig(filename, bbox_inches='tight', pad_inches=0.001)
	plt.close(fig)

	print(filename)

def plot_combined_js_mat(
		smm_mat:np.ndarray,
		imm_mat:np.ndarray,
		labels:list[str],
		title:str,
		output_path:str,
		vmin:float=0.0,
		vmax:float=0.3,
		cmap_name:str="viridis",
	):
	n = len(labels)
	ax:Axes
	fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
	ax.set_frame_on(False)
	norm = plt.Normalize(vmin=vmin, vmax=vmax) # type: ignore
	cmap = plt.get_cmap(cmap_name)

	# set up a single colorbar
	sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
 	# fig.colorbar(sm, ax=ax, shrink=0.75, label="JS distance")
	sm.set_array([])
	cb = fig.colorbar(
		sm,
		ax=ax,
		orientation="horizontal",
		pad=0.15,        # space between plot and colorbar
		fraction=0.046   # height of the bar relative to the figure
	)
	cb.set_label("JS distance")

	# invert y so row 0 is at top
	ax.set_xlim(0, n)
	ax.set_ylim(n, 0)

	padding = 0.05

	# ticks at cell centers
	ax.set_xticks(np.arange(n) + 0.5)
	ax.set_yticks(np.arange(n) + 0.5)
	ax.set_xticklabels(labels, rotation=0, ha="center")
	ax.set_yticklabels(labels, rotation=90, va="center")
	ax.xaxis.tick_top()

	ax.set_aspect("equal")
	# ax.tick_params(length=0)

	# draw each cell
	for i,j in itertools.product(range(n), range(n)):
		if i < j: # Upper Triangle
			x, y = j + padding, i - padding
			js_dist = imm_mat[i, j]
			c = cmap(norm(js_dist))
			rect = Rectangle((x, y), 1, 1, facecolor=c, edgecolor="white", lw=0.5)
			ax.add_patch(rect)
			text_color = "white" if js_dist < (vmin + vmax) / 2 else "black"
			ax.text(x+0.5, y+0.5, f"{imm_mat[i,j]:.3f}", ha="center", va="center", fontsize=10, color=text_color)
		elif i > j: # Lower Triangle
			x, y = j - padding, i + padding
			js_dist = smm_mat[i, j]
			c = cmap(norm(js_dist))
			rect = Rectangle((x, y), 1, 1, facecolor=c, edgecolor="white", lw=0.5)
			ax.add_patch(rect)
			text_color = "white" if js_dist < (vmin + vmax) / 2 else "black"
			ax.text(x+0.5, y+0.5, f"{smm_mat[i,j]:.3f}", ha="center", va="center", fontsize=10, color=text_color)
		else: # Diagonal
			# Can use either smm or imm here, as all values should be 0.
			x, y = j, i
			c = cmap(norm(smm_mat[i, j]))
			rect = Rectangle((x, y), 1, 1, facecolor=c, edgecolor="white", lw=0.5)
			ax.add_patch(rect)
			ax.text(x+0.5, y+0.5, "0", ha="center", va="center", fontsize=10, color="w")
	
	ax.plot([0 - padding, (n - 1) - 2 * padding], [n, n], color="black", linewidth=4)
	ax.plot([n, n], [0 - padding, (n - 1) - 2 * padding], color="black", linewidth=4)

	ax.annotate(
		"IMM Triangle",
		xy=(1.0, 0.66),
		xycoords="axes fraction",
		xytext=(1.075, 0.66),
		textcoords="axes fraction",
		ha="left", va="center",
		fontsize=10,
		rotation=90,
		arrowprops=dict(arrowstyle="-", lw=1)
	)

	ax.annotate(
		"SMM Triangle",
		xy=(0.33, 0),
		xycoords="axes fraction",
		xytext=(0.33, -0.075),
		textcoords="axes fraction",
		ha="center", va="top",
		fontsize=10,
		arrowprops=dict(arrowstyle="-", lw=1)
	)

	ax.set_title(title)

	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.001)
	plt.close(fig)
	print(output_path)

def main(metrics_dir, output_dir, sel_metric, title_suffix:str, axis:str):
	os.makedirs(output_dir, exist_ok=True)
	
	xr_smm_metric_densities = xr.load_dataarray(f"{metrics_dir}/smm-test_analytics.nc").sel(metric=sel_metric)
	xr_imm_metric_densities = xr.load_dataarray(f"{metrics_dir}/imm-test_analytics.nc").sel(metric=sel_metric)

	loss_names:list[str] = xr_smm_metric_densities.coords["loss_name"].values

	title = f"JS Dists {title_suffix}"
	
	diff_mm_js_dist_mat = build_js_dist_mat(xr_smm_metric_densities, xr_imm_metric_densities, axis)
	plot_js_dist_mat(diff_mm_js_dist_mat, loss_names, "Cross MM " + title, "IMM", "SMM", f"{sel_metric}-cross_mm")

	smm_js_dist_mat = build_js_dist_mat(xr_smm_metric_densities, xr_smm_metric_densities, axis)
	# plot_js_dist_mat(smm_js_dist_mat, loss_names, title, "SMM", "SMM", f"{sel_metric}-smm")

	imm_js_dist_mat = build_js_dist_mat(xr_imm_metric_densities, xr_imm_metric_densities, axis)
	# plot_js_dist_mat(imm_js_dist_mat, loss_names, title, "IMM", "IMM", f"{sel_metric}-imm")
	
	out_fn = os.path.join(output_dir, f"js-{sel_metric}-same_mm.png")
	plot_combined_js_mat(
		smm_js_dist_mat, imm_js_dist_mat,
		labels=loss_names,
		title="Same MM " + title,
		output_path=out_fn
	)


if __name__ == '__main__':
	print(f"Running {__file__}...")
	
	axis = "exclusive_quantile"

	sel_metrics = ["measured_sums", "measured_diffs"]
	title_suffices = ["MAE (by EQ)", "MBE (by EQ)"]

	metrics_dir = "analytics/metrics"
	output_dir = "analytics/js_distances"

	for sel_metric, title_suffix in zip(sel_metrics, title_suffices):
		main(metrics_dir, output_dir, sel_metric, title_suffix, axis)
