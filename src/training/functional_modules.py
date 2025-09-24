import torch
import torch.nn.functional as F

import training.functional as F_Ext


class BCELossND(torch.nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.flatten = torch.nn.Flatten()
		
	def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
		return F.binary_cross_entropy(self.flatten(y_pred), torch.clamp(self.flatten(y_true), min=1e-9, max=1-1e-9), reduction='none').mean(dim=-1)

class MSELossND(torch.nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.mse_loss = torch.nn.MSELoss(reduction='none')
		self.flatten = torch.nn.Flatten()
		
	def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
		return self.mse_loss(self.flatten(y_pred), self.flatten(y_true)).mean(dim=-1)

class HuberLossND(torch.nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.mse_loss = torch.nn.HuberLoss(reduction='none')
		self.flatten = torch.nn.Flatten()
		
	def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
		return self.mse_loss(self.flatten(y_pred), self.flatten(y_true)).mean(dim=-1)

class BTELossND(torch.nn.Module):
	def __init__(self, lb:float=0, ub:float=1, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.lb = lb
		self.ub = ub
		
	def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
		return F_Ext.nbte(y_pred, y_true, self.lb, self.ub)

class MeasureFalsePositivesND(torch.nn.Module):
	def __init__(self, q:float, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.q = q
		
	def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
		return F_Ext.measure_false_positives(y_pred, y_true, q=self.q)

class MeasureFalseNegativesND(torch.nn.Module):
	def __init__(self, q:float, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.q = q
		
	def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
		return F_Ext.measure_false_negatives(y_pred, y_true, q=self.q)

class CountFalsePositivesND(torch.nn.Module):
	def __init__(self, q:float, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.q = q
		
	def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
		return F_Ext.count_false_positives(y_pred, y_true, q=self.q)

class CountFalseNegativesND(torch.nn.Module):
	def __init__(self, q:float, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.q = q
		
	def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
		return F_Ext.count_false_negatives(y_pred, y_true, q=self.q)

class QuantileSliceSizeND(torch.nn.Module):
	def __init__(self, q:float, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.q = q
		
	def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
		mask = F_Ext._quantile_mask(y_pred, y_true, q=self.q)
		return mask.sum()
