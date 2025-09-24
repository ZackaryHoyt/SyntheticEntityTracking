import pathlib
import json
import os

import numpy as np

from dataset_util.visualizers import save_as_heatmap
from environment_generation.perlin.noisemap_builder import create_perlin_noisemap_builder
from environment_generation.perlin.settings import PerlinEnvGenSettings

def main(settings_file:str) -> None:
	with open(settings_file, 'r') as ifs:
		settings = PerlinEnvGenSettings(**json.load(ifs))
	noisemap_builder = create_perlin_noisemap_builder(settings)

	if os.path.exists(settings.env_data_output_file):
		print("!!!Environments already exist; delete existing environments before attempting to build new ones.")
		exit(-1)

	os.makedirs(settings.examples_output_dir, exist_ok=True)

	data = { }
	for idx in range(settings.n_samples):
		# the `idx` doubles as the offset along the diagonal of the noisemap space.
		idx_str = str(idx)
		data[idx_str] = noisemap_builder(offset=idx)
		if idx < settings.n_examples:
			save_as_heatmap(arr=data[idx_str], file=f"{settings.examples_output_dir}/{idx_str}.png")
		print(f"\rGenerated {idx+1}/{settings.n_samples} environments...", end='')

	os.makedirs(pathlib.Path(settings.env_data_output_file).parent, exist_ok=True)
	np.savez_compressed(settings.env_data_output_file, **data)
	print(f"\nSaved compressed archive of environments to {settings.env_data_output_file}.")

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--settings", help="Settings filepath.")
	args = parser.parse_args()

	settings_filepath:str = args.settings

	print(f"Running {__file__}...")
	print(f"Configured settings filepath: {settings_filepath}")

	main(settings_filepath)
