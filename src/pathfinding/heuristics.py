from typing import Any

import numpy as np


def get_manhattan_distance_2d(a:tuple[int, int], b:tuple[int, int]) -> int:
	"""
	Calculates the Manhattan distance between two points in a 2D grid.
	
	Parameters:
	- a (tuple[int, int]): The first point (x1, y1).
	- b (tuple[int, int]): The second point (x2, y2).
	
	Returns:
	- int: The Manhattan distance between the two points.
	"""
	return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_chebyshev_distance_2d(a:tuple[int, int], b:tuple[int, int]) -> int:
	"""
	Calculates the Chebyshev distance between two points in a 2D grid.
	
	The Chebyshev distance is the maximum of the absolute differences of their coordinates.
	
	Parameters:
	- a (tuple[int, int]): The first point (x1, y1).
	- b (tuple[int, int]): The second point (x2, y2).
	
	Returns:
	- int: The Chebyshev distance between the two points.
	"""
	return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def get_euclidean_distance_2d(a:tuple[int, int], b:tuple[int, int]) -> float:
	"""
	Calculates the Euclidean distance between two points in a 2D grid.
	
	The Euclidean distance is the square root of the sum of the squared differences of their coordinates.
	
	Parameters:
	- a (tuple[int, int]): The first point (x1, y1).
	- b (tuple[int, int]): The second point (x2, y2).
	
	Returns:
	- float: The Euclidean distance between the two points.
	"""
	return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def get_zero(a:Any, b:Any) -> float:
	"""
	A heuristic function that always returns 0, effectively ignoring the heuristic part of A*.
	
	This function can be used to turn the A* algorithm into Dijkstra's algorithm by not favoring any particular direction.
	
	Parameters:
	- a (T): The first point (not used).
	- b (T): The second point (not used).
	
	Returns:
	- float: Always returns 0.
	"""
	return 0
