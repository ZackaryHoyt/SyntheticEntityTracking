from typing import Callable, Iterable, TypeAlias, TypeVar, Generic, Dict

T = TypeVar('T')


EdgeCostType:TypeAlias = float|int
PathCostType:TypeAlias = float|int
EdgeCostFuncType:TypeAlias = Callable[[T,T],EdgeCostType]
HeuristicCostFuncType:TypeAlias = Callable[[T,T],PathCostType]
PathCostFuncType:TypeAlias = Callable[[T,T],PathCostType]
NeighborsFuncType:TypeAlias = Callable[[T],Iterable[T]]
PathMapType:TypeAlias = Dict[T,T]
PathCostMapType:TypeAlias = Dict[T,PathCostType]

class Graph(Generic[T]):
	"""
	Abstract base class for graph structures.
	"""
	def __init__(self) -> None:
		pass
	
	def get_cost(self, a:T, b:T) -> EdgeCostType:
		"""
		Gets the cost of moving from node 'a' to node 'b'.
		
		Parameters:
		- a (T): The starting node.
		- b (T): The destination node.
		- path_cost_a (float): Cumulative path cost to the starting node 'a'.
		
		Returns:
		- float: The cost of moving from 'a' to 'b'.
		
		Raises:
		- NotImplementedError: If the method is not overridden in a subclass.
		"""
		raise NotImplementedError()
	
	def get_neighbors(self, a:T) -> Iterable[T]:
		"""
		Gets the neighbors of node 'a'.
		
		Parameters:
		- a (T): The node for which neighbors are to be found.
		
		Returns:
		- Iterable[T]: An iterable of neighbors for node 'a'.
		
		Raises:
		- NotImplementedError: If the method is not overridden in a subclass.
		"""
		raise NotImplementedError()
