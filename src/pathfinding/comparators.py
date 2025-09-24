import numpy as np

def cost_comparator(a:float, b:float) -> bool:
	"""
	Compares two costs for regular ordering.
	
	Parameters:
	- a (float): The first cost value.
	- b (float): The second cost value.
	
	Returns:
	- bool: True if 'a' is less than 'b'; otherwise, False.
	"""
	return a < b

def cost_comparator_reverse(a:float, b:float) -> bool:
	"""
	Compares two costs for reverse ordering.
	
	Parameters:
	- a (float): The first cost value.
	- b (float): The second cost value.
	
	Returns:
	- bool: True if 'a' is greater than 'b'; otherwise, False.
	"""
	return a > b

def cost_comparator_w_precision(a:float, b:float, precision:int) -> bool:
	"""
	Compares two costs for regular ordering with specified precision.
	
	Parameters:
	- a (float): The first cost value.
	- b (float): The second cost value.
	- precision (int): The decimal precision for comparison.
	
	Returns:
	- bool: True if 'a' is less than 'b' when rounded to the specified precision; otherwise, False.
	"""
	return np.round(a, decimals=precision) < np.round(b, decimals=precision)

def cost_comparator_w_precision_reverse(a:float, b:float, precision:int) -> bool:
	"""
	Compares two costs for reverse ordering with specified precision.
	
	Parameters:
	- a (float): The first cost value.
	- b (float): The second cost value.
	- precision (int): The decimal precision for comparison.
	
	Returns:
	- bool: True if 'a' is greater than 'b' when rounded to the specified precision; otherwise, False.
	"""
	return np.round(a, decimals=precision) > np.round(b, decimals=precision)
