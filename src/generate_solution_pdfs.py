import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split

from train_mm_models import TrainingSettings
from dataset_util.smm_dataset import SMMDataset, smm_epxv_shape_map
from dataset_util.imm_dataset import IMMDataset, imm_epxv_shape_map


def load_settings(settings_filepath:str):
	settings_filename = os.path.basename(settings_filepath)
	with open(settings_filepath, 'r') as f:
		config = json.load(f)

	# Dynamic motion model selection.
	if settings_filename.startswith('smm'):
		print("Auto-selecting the 'smm'.")
		settings = SMMTrainingSettings(**config)
		density_map = smm_epxv_shape_map
		label = 'smm'
		make_dataset = lambda d:SMMDataset(
			[settings.env_dataset_file, settings.signals_dataset_file],
			settings.dataset_file,
			epsilon=density_map[d]
		)
	elif settings_filename.startswith('imm'):
		print("Auto-selecting the 'imm'.")
		settings = IMMTrainingSettings(**config)
		density_map = imm_epxv_shape_map
		label = 'imm'
		make_dataset = lambda d:IMMDataset(
			[settings.env_dataset_file, settings.signals_dataset_file],
			settings.dataset_file,
			epsilon=density_map[d]
		)
	else:
		raise ValueError("Unidentified motion model found.")

	return settings, density_map, label, make_dataset


def collect_target_values(dataset, settings) -> np.ndarray:
	_, _, test_dataset = random_split(
		dataset, settings.data_partitions,
		generator=torch.Generator().manual_seed(settings.seed)
	)
	dataloader = DataLoader(test_dataset, batch_size=128)

	ys_targ = []
	with torch.no_grad():
		for _, y_batch in dataloader:
			ys_targ.extend(y_batch)

	ys_targ = torch.stack(ys_targ)
	print(ys_targ.mean())
	return ys_targ.cpu().numpy().flatten()


def compute_pdf(values:np.ndarray, bins:np.ndarray):
	counts, _ = np.histogram(values, bins=bins, density=True)
	bin_edges = bins
	return counts, bin_edges


def generate_pdf_matrix(densities, make_dataset, settings, n_bins):
	normalized_histograms = []
	common_bin_edges = np.arange(n_bins) / n_bins

	for density in densities:
		values = collect_target_values(make_dataset(density), settings)
		pdf, _ = np.histogram(values, bins=common_bin_edges, density=True)
		normalized_histograms.append(pdf / pdf.max())

	return np.stack(normalized_histograms), common_bin_edges


def plot_pdf_contour(pdf_matrix:np.ndarray, bin_edges:np.ndarray, densities:list, label:str):
	plt.figure(figsize=(6, 6), dpi=300)
	ax = sns.heatmap(
		pdf_matrix.T,
		cmap='viridis',
		square=True,
		cbar_kws={"shrink": 0.75},
	)
	yticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
	ytick_labels = list(map(str, [0] + [bin_edges[tick] for tick in yticks[1:-1]] + [1]))
	ax.set_yticks(yticks, labels=ytick_labels)

	xticks = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 98]
	xticks_labels = [densities[tick] for tick in xticks]
	ax.set_xticks(yticks, labels=xticks_labels)
	ax.invert_yaxis()

	ax.set_xlabel("Dataset Density")
	ax.set_ylabel("Value Bin")
	ax.set_title(f"PDF Contour over Densities - {label.upper()}")
	plt.tight_layout()

	output_path = os.path.join("output", "analytics", "solution_histograms", label)
	os.makedirs(output_path, exist_ok=True)
	plt.savefig(os.path.join(output_path, f"{label}_pdf_contour.png"))
	plt.close()


def generate_density_pdf_contours(settings_file: str):
	settings, density_map, label, make_dataset = load_settings(settings_file)
	densities = list(density_map.keys())

	pdf_matrix, bin_edges = generate_pdf_matrix(densities, make_dataset, settings, n_bins=100)
	plot_pdf_contour(pdf_matrix, bin_edges, densities, label)


if __name__ == "__main__":
	print(f"Running {__file__}...")
	generate_density_pdf_contours("settings/smm_trainer.json")  # or imm_trainer.json
