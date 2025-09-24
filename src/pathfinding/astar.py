from typing import Any
from dataclasses import dataclass
from queue import PriorityQueue, Queue

import numpy as np

import pathfinding.comparators as pf_util
from pathfinding.graph import *


@dataclass
class PathResult(Generic[T]):
	path:list[T]|None # The list of nodes forming the shortest path from start to goal, or None if no path is found.
	path_costs:list[EdgeCostType]|None # The list of cumulative costs associated with each node in the shortest path, or None if no path is found.
	path_map:PathMapType # A dictionary mapping each node to its predecessor on the shortest path.
	path_cost_map:PathCostMapType # A dictionary mapping each node to the total cost of reaching it from the start node.

def find_path(current:Any, path_map:PathMapType, path_cost_map:PathCostMapType) -> PathResult:
	path = []
	path_costs = []
	while current is not None:
		path.append(current)
		path_costs.append(path_cost_map[current])
		current = path_map[current]

	return PathResult(path[::-1], path_costs[::-1], path_map, path_cost_map)

def a_star(
		start:T,
		goal:T,
		f_cost:EdgeCostFuncType,
		f_neighbors:NeighborsFuncType,
		f_heuristic:HeuristicCostFuncType,
		reverse_comparator:bool=False,
		reverse_priority:bool=False,
		enable_priority:bool=True,
		precision:int|None=None,
		debug:bool=False
	) -> PathResult:
	assert f_neighbors is not None
	assert f_heuristic is not None

	# 'start' and 'goal' can be 'None', but it is expected that the provided functions will be compatible.
	# 'goal' being outside of the graph (relative to 'start') will simply build the optimal path from the 'start' to every node in the space.

	# following the chain of nodes in the path_map will lead to the starting node

	if precision is None:
		f_cost_comparator = pf_util.cost_comparator_reverse if reverse_comparator else pf_util.cost_comparator
	else:
		_f_cost_comparator = pf_util.cost_comparator_w_precision_reverse if reverse_comparator else pf_util.cost_comparator_w_precision
		f_cost_comparator = lambda a, b : _f_cost_comparator(a, b, precision)

	priority_direction = -1 if reverse_priority else 1
	open_queue = PriorityQueue() if enable_priority else Queue()
	open_queue.put((0, start)) # priority (cost), node, set of nodes in the path history
	path_map:PathMapType = { start : None } # child : parent
	path_cost_map:PathCostMapType = { start : 0 } # node : path cost to node from start

	current:T
	while not open_queue.empty():
		_, current = open_queue.get()

		if debug:
			print(f"current: {current}")

		if current == goal:
			return find_path(current, path_map, path_cost_map)
		
		for neighbor in f_neighbors(current):
			path_cost = path_cost_map[current] + f_cost(current, neighbor) # cumulative cost from the start to the neighbor node.

			if neighbor not in path_cost_map or f_cost_comparator(path_cost, path_cost_map[neighbor]):
				path_map[neighbor] = current
				path_cost_map[neighbor] = path_cost
				path_heuristic = path_cost + f_heuristic(neighbor, goal)

				if debug:
					print(f"\t{current} -> {neighbor} : path=(cost={path_cost}, heuristic={path_heuristic})")
				
				open_queue.put((priority_direction * path_heuristic, neighbor))

	return PathResult(None, None, path_map, path_cost_map)

def a_star2(
		start:T,
		goal:T,
		graph:Graph,
		f_heuristic:HeuristicCostFuncType,
		reverse_comparator:bool=False,
		reverse_priority:bool=False,
		enable_priority:bool=True,
		precision:int|None=None,
		debug:bool=False
	) -> PathResult:
	return a_star(
			start=start,
			goal=goal,
			f_cost=graph.get_cost,
			f_neighbors=graph.get_neighbors,
			f_heuristic=f_heuristic,
			reverse_comparator=reverse_comparator,
			reverse_priority=reverse_priority,
			enable_priority=enable_priority,
			precision=precision,
			debug=debug
		)

if __name__ == "__main__":
	import pathfinding.heuristics as heuristics
	import pathfinding.ndarray_graph as ndarray_graph

	# arr = np.array([
	# 	[0, 0, 0, 0, 0],
	# 	[0, 1, 1, 1, 0],
	# 	[0, 0, 0, 1, 0],
	# 	[0, 1, 1, 1, 0],
	# 	[0, 0, 0, 0, 0]
	# ])
	# graph = ndarray_graph.NDArrayGraph2D(array=arr)
	# start = (2, 0)
	# goal = None
	# path_result = a_star2(start=start, goal=goal, graph=graph, f_heuristic=heuristics.get_zero)
	# # for k,v in path_result.path_cost_map.items():
	# # 	print(f"{k}: {v}")
	# print("Path from start to goal:", path_result.path)

	s = 4
	# arr = np.arange(s * s).reshape(s, s)
	arr = 1 + np.abs(np.random.normal(loc=0, scale=1, size=(s, s)))
	graph = ndarray_graph.NDArrayGraph2D(array=arr)
	path_result = a_star2(start=(2, 0), goal=None, graph=graph, f_heuristic=heuristics.get_zero)
	for k,v in path_result.path_cost_map.items():
		print(f"{k}: {v}")