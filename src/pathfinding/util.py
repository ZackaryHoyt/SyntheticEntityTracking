import itertools
from typing import TypeVar

import numpy as np

T = TypeVar('T')


def path_cost_map_to_array(path_cost_map:dict[tuple[int,int],float], shape:tuple[int,int]|tuple[int,...]) -> np.ndarray:
	arr = np.zeros(shape) # energy required to reach the goal from some position
	h, w = shape
	for i, j in itertools.product(range(h), range(w)):
		arr[i, j] = path_cost_map[(i, j)]
	return arr
