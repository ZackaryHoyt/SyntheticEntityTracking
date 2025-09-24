import numpy as np

def minmax_normalization(arr:np.ndarray) -> np.ndarray:
	return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def l1_normalization(x:np.ndarray) -> np.ndarray:
	x = x - x.min()
	return x / x.sum()