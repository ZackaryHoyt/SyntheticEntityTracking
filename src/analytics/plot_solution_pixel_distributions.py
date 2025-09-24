import json
import os
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from dataset_util.imm_dataset import IMMDataset, imm_epxv_shape_map
from dataset_util.smm_dataset import SMMDataset, smm_epxv_shape_map
from train_mm_models import TrainingSettings


@torch.no_grad()
def collect_target_values(dataset:Dataset, partitions:tuple[float, float, float], seed:int, batch_size:int=128, max_iter:int|None=None) -> np.ndarray:
	_, _, test_dataset = random_split(dataset, partitions, generator=torch.Generator().manual_seed(seed))
	dataloader = DataLoader(test_dataset, batch_size=batch_size)

	ys_targ = []
	for i, (_, y_batch) in enumerate(dataloader):
		if max_iter and i * batch_size >= max_iter:
			break
		ys_targ.extend(y_batch)

	ys_targ = torch.stack(ys_targ)
	return ys_targ.cpu().numpy().flatten()

def compute_pdf(values:Sequence[float], bins:Sequence[float]):
	counts, _ = np.histogram(values, bins=bins, density=True)
	bin_edges = bins
	return counts, bin_edges

def generate_pdf_matrix(densities:Sequence[float], make_dataset:Callable[[float],Dataset], settings:TrainingSettings, n_bins:int) -> tuple[np.ndarray, np.ndarray]:
	pdfs = []
	common_bin_edges = np.arange(int(0.4 * n_bins)) / n_bins

	for density in densities:
		values = collect_target_values(make_dataset(density), settings.data_partitions, settings.seed)
		nonzero_values = values[values != 0] # There is at most a single 0-value pixel in each 128x128 sample, so filtering them out is not significant.
		normalized_nl_values = np.log(nonzero_values) / np.log(nonzero_values).min()
		mean_val, min_val, max_val = values.mean().__float__(), normalized_nl_values.min().__float__(), normalized_nl_values.max().__float__()
		pdf, _ = np.histogram(normalized_nl_values, bins=common_bin_edges, density=True)
		pdfs.append(pdf / n_bins)
		print(f"{mean_val:.3f}\t{min_val:.3f}\t{max_val:.3f}\t{np.sum(pdf):.3f}")

	return np.stack(pdfs), common_bin_edges

def plot_pdf_contour(pdf_matrix:np.ndarray, bin_edges:Sequence[float], densities:Sequence[float], label:str) -> None:
	fig = plt.figure(figsize=(4, 4), dpi=300)
	ax = sns.heatmap(
		pdf_matrix.T,
		cmap='viridis',
		# square=True,
		cbar_kws={"shrink": 0.75, 'label': "relative density"},
	)
	yticks = list(range(len(bin_edges)))[::(len(bin_edges)//10)]
	ytick_labels = [bin_edges[tick] for tick in yticks]
	ax.set_yticks(yticks, labels=[f"{tick_label:.2}" for tick_label in ytick_labels])

	xticks = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 98]
	xticks_labels = [densities[tick] for tick in xticks]
	ax.set_xticks(xticks, labels=[str(tick_label) for tick_label in xticks_labels])
	ax.invert_yaxis()

	ax.set_xlabel("E[px]")
	ax.set_ylabel("maxnorm(-log(px))")
	ax.set_title(f"{label.upper()} Pixel Distribution Heatmap")
	plt.tight_layout()

	output_path = os.path.join("analytics", "solution_pixel_distributions")
	os.makedirs(output_path, exist_ok=True)
	filename = os.path.join(output_path, f"{label}_pdf_contour.png")
	fig.savefig(filename, bbox_inches='tight', pad_inches=0.001)
	plt.close()
	print(filename)

def main(motion_model_name:str, settings_filepath:str) -> None:
	with open(settings_filepath, 'r') as f:
		config = json.load(f)
	settings = TrainingSettings(**config)
	
	match(motion_model_name):
		case 'smm':
			shape_map = smm_epxv_shape_map
			dataset_cls = SMMDataset
		case 'imm':
			shape_map = imm_epxv_shape_map
			dataset_cls = IMMDataset
		case _:
			raise ValueError()

	epxs = list(shape_map.keys())

	xs_files = [settings.env_dataset_file, settings.signals_dataset_file]
	ys_file = settings.dataset_file
	make_dataset = lambda density : dataset_cls.from_density(xs_files, ys_file, density)

	pdf_matrix, bin_edges = generate_pdf_matrix(epxs, make_dataset, settings, n_bins=1000)
	plot_pdf_contour(pdf_matrix, bin_edges.tolist(), epxs, motion_model_name)


if __name__ == "__main__":
	print(f"Running {__file__}...")
	
	main('smm', "settings/smm_trainer.json")
	main('imm', "settings/imm_trainer.json")
