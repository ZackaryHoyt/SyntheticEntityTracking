import torch
from torchmetrics import Metric

import training.functional_modules as functional_modules


class FunctionMetricND(Metric):
	criterion:torch.nn.Module
	sum:torch.Tensor
	count:torch.Tensor

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
		self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

	def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
		l:torch.Tensor = self.criterion(y_pred, y_true)
		self.sum += l.sum()
		self.count += l.numel()

	def compute(self) -> torch.Tensor:
		return self.sum / self.count

class MeasureFalsePositivesMetricND(FunctionMetricND):
	def __init__(self, q:float, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.criterion = functional_modules.MeasureFalsePositivesND(q)

class MeasureFalseNegativesMetricND(FunctionMetricND):
	def __init__(self, q:float, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.criterion = functional_modules.MeasureFalseNegativesND(q)

class CountFalsePositivesMetricND(FunctionMetricND):
	def __init__(self, q:float, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.criterion = functional_modules.CountFalsePositivesND(q)

class CountFalseNegativesMetricND(FunctionMetricND):
	def __init__(self, q:float, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.criterion = functional_modules.CountFalseNegativesND(q)

class QuantileSliceSizeMetricND(FunctionMetricND):
	def __init__(self, q:float, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.criterion = functional_modules.QuantileSliceSizeND(q)
