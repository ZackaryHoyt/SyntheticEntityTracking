import json
import os
from dataset_util.visualizers import save_as_heatmap
from target_generation.settings import TargetGenSettings
from dataset_util.imm_dataset import IMMDataset
from dataset_util.smm_dataset import SMMDataset

def main(motion_model_name:str, settings_filepath:str, example_densities:list[float]) -> None:
	# Dynamic motion model selection.
	with open(settings_filepath, 'r') as ifs:
		match(motion_model_name):
			case 'smm':
				settings = TargetGenSettings(**json.load(ifs))
				dataset_cls = SMMDataset
			case 'imm':
				settings = TargetGenSettings(**json.load(ifs))
				dataset_cls = IMMDataset
			case _:
				raise ValueError("Unidentified motion model found.")
	
	# Make output directories.

	# Generate target examples.
	xs_files = [settings.environments_dataset_file, settings.signals_dataset_file]
	ys_file = settings.motion_models_dataset_file

	os.makedirs(settings.motion_model_examples_output_dir, exist_ok=True)
	dataset = None
	for density in example_densities:
		target_examples_dir = os.path.join(settings.motion_model_examples_output_dir, f"{density:.2f}")
		os.makedirs(target_examples_dir, exist_ok=True)

		dataset = dataset_cls.from_density(xs_files=xs_files, ys_file=ys_file, density=density)
		for idx in range(settings.n_examples):
			label = str(idx)
			_, y = dataset[idx]
			save_as_heatmap(arr=y.squeeze().numpy(), file=f"{target_examples_dir}/{label}.png")

	# Generate input examples
	if dataset is None:
		# Bounds check to verify the dataset was defined.
		# Note all densities share the same input set, so the last dataset defined can be used.
		raise ValueError()
	
	os.makedirs(settings.signal_examples_output_dir, exist_ok=True)
	for idx in range(settings.n_examples):
		label = str(idx)
		x, _ = dataset[idx]
		environment_grid, signals_grid = x
		save_as_heatmap(arr=signals_grid.squeeze().numpy(), file=f"{settings.signal_examples_output_dir}/{label}.png")


if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--motion_model", help="Motion model type.", default='smm', choices=['smm', 'imm'])
	parser.add_argument("--settings", help="Settings filepath.")
	args = parser.parse_args()

	motion_model_name:str = args.motion_model.lower()
	settings_filepath:str = args.settings

	print(f"Running {__file__}...")
	print(f"Configuared motion model: {motion_model_name}")
	print(f"Configured settings filepath: {settings_filepath}")

	example_densities = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
	main(motion_model_name, settings_filepath, example_densities)
