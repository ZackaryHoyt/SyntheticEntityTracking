from typing import Callable

import numpy as np

import pathfinding.astar as astar
import pathfinding.heuristics as heuristics
from pathfinding.ndarray_graph import NDArray2DNodeType, NDArrayGraph2D, T
from pathfinding.util import path_cost_map_to_array


def flow(
		source:NDArray2DNodeType,
		graph:NDArrayGraph2D,
		priority:bool=False,
		f_heuristic:Callable[[T, T], float]=heuristics.get_zero
	) -> np.ndarray:
	# Computes the least-cost from the source to all reachable nodes in the graph.
	# Priority should be enabled only if costs are not guaranteed to be monotonically increasing.
	path_result = astar.a_star2(
		start=source,
		goal=None,
		graph=graph,
		f_heuristic=f_heuristic,
		enable_priority=priority,
		precision=9
	)
	return path_cost_map_to_array(
		path_cost_map=path_result.path_cost_map,
		shape=graph.array.shape
	)
