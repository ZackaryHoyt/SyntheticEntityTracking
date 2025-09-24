from typing import TypeAlias

import numpy as np

from pathfinding.graph import *

NDArray2DNodeType:TypeAlias = tuple[int,int]

NDArray2DPathCostFuncType:TypeAlias = PathCostFuncType[NDArray2DNodeType]
NDArray2DEdgeCostFuncType:TypeAlias = EdgeCostFuncType[NDArray2DNodeType]
NDArray2DHeuristicCostFuncType:TypeAlias = HeuristicCostFuncType[NDArray2DNodeType]
NDArray2DNeighborsFuncType:TypeAlias = NeighborsFuncType[NDArray2DNodeType]
NDArray2DPathMapType:TypeAlias = PathMapType[NDArray2DNodeType]
NDArray2DPathCostMapType:TypeAlias = PathCostMapType[NDArray2DNodeType]


class NDArrayGraph2D(Graph[NDArray2DNodeType]):
	"""
	Represents a 2D numpy array as a graph for pathfinding, where each cell's value can represent the cost to enter.
	
	Attributes:
	- array (np.ndarray): The 2D array representing the graph.
	- h, w (int): The height and width of the array.
	"""
	def __init__(self, array:np.ndarray) -> None:
		"""
		Initializes the graph with a 2D numpy array.
		
		Parameters:
		- array (np.ndarray): A 2D numpy array where each cell's value represents the cost to enter.
		"""
		self.array = array
		self.h, self.w = array.shape

	def in_bounds(self, i:int, j:int) -> bool:
		"""
		Checks if the specified cell is within the bounds of the array.
		
		Parameters:
		- i, j (int): The row and column indices of the cell.
		
		Returns:
		- bool: True if the cell is within bounds; otherwise, False.
		"""
		return 0 <= i < self.h and 0 <= j < self.w
	
	def validate_neighbor(self, i:int, j:int) -> bool:
		"""
		Validates whether a neighboring cell is within bounds and has a finite value.
		
		Parameters:
		- i, j (int): The row and column indices of the neighbor cell.
		
		Returns:
		- bool: True if the neighbor is valid; otherwise, False.
		"""
		return self.in_bounds(i, j) and np.isfinite(self.array[i, j])
	
	def get_cost(self, a, b) -> float:
		"""
		Returns the cost of moving from cell 'a' to cell 'b'.
		
		Parameters:
		- a, b (tuple[int, int]): The (row, column) indices of the start cell and the target cell.
		
		Returns:
		- float: The cost to enter cell 'b'.
		"""
		return self.array[b]
	
	def get_neighbors(self, a) -> Iterable[NDArray2DNodeType]:
		"""
		Generates a list of valid neighboring cells around a given cell, based on 4-directional movement.
		
		Parameters:
		- a (tuple[int, int]): The (row, column) indices of the cell.
		
		Returns:
		- list[tuple[int, int]]: A list of tuples representing valid neighbors.
		"""
		i, j = a
		offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
		return [(i + di, j + dj) for di, dj in offsets if self.validate_neighbor(i + di, j + dj)]
