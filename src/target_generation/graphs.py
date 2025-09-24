import numpy as np

from pathfinding.ndarray_graph import NDArrayGraph2D, NDArray2DNodeType


class NDArrayGraph2D_Inertial(NDArrayGraph2D):
	def __init__(self, array:np.ndarray) -> None:
		super().__init__(array=array)
	
	def get_cost(self, a:NDArray2DNodeType, b:NDArray2DNodeType) -> float:
		return abs(self.array[b] - self.array[a]) ** 2
