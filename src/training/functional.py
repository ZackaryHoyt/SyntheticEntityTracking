import torch


def _quantile_mask(a:torch.Tensor, b:torch.Tensor, q:float):
	if q < 0:
		return -b <= -b.quantile(1 + q) # Selects Upper Q%
	else:
		return b <= b.quantile(q) # Selects Lower Q%

def quantile_mask(a:torch.Tensor, b:torch.Tensor, q:float):
	if abs(q) != 1:
		mask = _quantile_mask(a, b, q)
		a = a[mask]
		b = b[mask]
	return a, b


def measure_false_positives(a:torch.Tensor, b:torch.Tensor, q:float=1) -> torch.Tensor:
	a, b = quantile_mask(a, b, q)
	return torch.where(a > b, a - b, 0).sum() / b.numel() # TODO scale the denominator to match the max range of the numerator.

def measure_false_negatives(a:torch.Tensor, b:torch.Tensor, q:float=1) -> torch.Tensor:
	a, b = quantile_mask(a, b, q)
	return torch.where(a < b, b - a, 0).sum() / b.numel() # TODO scale the denominator to match the max range of the numerator.

def count_false_positives(a:torch.Tensor, b:torch.Tensor, q:float=1) -> torch.Tensor:
	a, b = quantile_mask(a, b, q)
	return torch.where(a > b, 1, 0).sum() / b.numel() # TODO scale the denominator to match the max range of the numerator.

def count_false_negatives(a:torch.Tensor, b:torch.Tensor, q:float=1) -> torch.Tensor:
	a, b = quantile_mask(a, b, q)
	return torch.where(a < b, 1, 0).sum() / b.numel() # TODO scale the denominator to match the max range of the numerator.


def nbte(a:torch.Tensor, b:torch.Tensor, lb:float=0, ub:float=1) -> torch.Tensor:
	"""
	Computes the Normalized Balanced Type Error (NBTE) between prediction `a` and target `b`.

	Args:
		a (torch.Tensor): Predicted tensor of shape (B, C, ...).
		b (torch.Tensor): Ground-truth tensor of same shape as `a`.
		lb (float): Lower bound for normalization (default: 0).
		ub (float): Upper bound for normalization (default: 1).

	Returns:
		torch.Tensor: NBTE loss per (B, C) pair.
	"""
	dim = tuple(range(2, a.ndim))

	# Measure Total Potential Type Error of Each Batch and Channel
	pte1_ttl = torch.sum(ub - b, dim=dim)
	pte2_ttl = torch.sum(b - lb, dim=dim)

	# Measure Total Observed Type Error of Each Batch and Channel
	te1_ttl = torch.sum(torch.clamp_min(a - b, min=0), dim=dim)
	te2_ttl = torch.sum(torch.clamp_min(b - a, min=0), dim=dim)

	# Compute Type Error Weights
	pte1_nonzero_mask = pte1_ttl != 0
	te1_weight = torch.zeros_like(pte1_ttl)
	te1_weight[pte1_nonzero_mask] = 1 / pte1_ttl[pte1_nonzero_mask]

	pte2_nonzero_mask = pte2_ttl != 0
	te2_weight = torch.zeros_like(pte2_ttl)
	te2_weight[pte2_nonzero_mask] = 1 / pte2_ttl[pte2_nonzero_mask]

	# Return the N-Array Balanced Type Error
	return te1_weight * te1_ttl + te2_weight * te2_ttl


