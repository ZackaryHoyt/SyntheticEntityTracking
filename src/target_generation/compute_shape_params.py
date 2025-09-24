import copy
import json
import time

import numpy as np
from numpy.lib.npyio import NpzFile

from target_generation.settings import TargetGenSettings


def find_distribution_shape(arr:np.ndarray, y_target:float, maxiter:int=50, precision:int=9) -> tuple[int,float,float,float]:
	if np.any(arr < 0) or np.any(arr > 1):
		raise ValueError("All arr entries must lie in [0,1].")
	if not 0 <= y_target <= 1:
		raise ValueError("y_target must lie between 0 and 1.")
	
	y_neutral = arr.mean()
	if np.round(np.abs(y_neutral - y_target), decimals=precision) == 0:
		return 0, 1, arr.std(), 0
	
	arr = copy.deepcopy(arr)
	nonzero_mask = arr > 0
	arr[nonzero_mask] = np.log(arr[nonzero_mask])
	
	alpha_min, alpha_max = 0, int(y_neutral < y_target)
	alpha = 1
	for i in range(maxiter):
		if alpha_max:
			alpha = (alpha_min + alpha_max) / 2
		else:
			alpha *= 2
		
		w = np.exp(alpha * arr)
		w[~nonzero_mask] = 0

		y = w.mean()
		error = np.round(np.abs(y - y_target), decimals=precision)
		if error == 0:
			break
		if y < y_target:
			alpha_max = alpha
		else:
			alpha_min = alpha

	return i + 1, alpha, w.std(), error # type: ignore

def main(settings_filepath:str) -> None:
	with open(settings_filepath, 'r') as ifs:
		settings = TargetGenSettings(**{k:v for k,v in json.load(ifs).items() if k in TargetGenSettings.__annotations__})

	archive:NpzFile
	with np.load(settings.motion_models_dataset_file) as archive:
		data = np.stack([archive[k] for k in archive.files])
	arr = np.exp(data.ravel())

	target_means = (1 + np.arange(99)) / 100
	# 'np.arange(0.01, 1, step=0.01)' should be equivalent, but there is some weird floating point error there that the above approach corrects.
	for target_mean in target_means:
		t0 = time.time()
		n_steps, alpha, stddev, error = find_distribution_shape(arr, target_mean)
		duration = time.time() - t0
		print(f"{n_steps:>3} steps ({duration:>6.2f}s): μ={target_mean:.2f}, σ={stddev:.6f}: α={alpha:.9f}, error={error}")

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--settings", help="Settings filepath.")
	args = parser.parse_args()

	settings_filepath:str = args.settings

	print(f"Running {__file__}...")
	print(f"Configured settings filepath: {settings_filepath}")

	main(settings_filepath)
