from typing import TypeAlias

import torch


x_batch_type:TypeAlias = torch.Tensor
y_pred_batch_type:TypeAlias = torch.Tensor

class CustomModule(torch.nn.Module):
	output_shape:tuple[int,int,int]

	def __init__(self, input_shape:tuple[int,int,int], *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.input_shape = input_shape
		self.n_input_dims = len(input_shape)

	def initialize_output_shape(self):
		with torch.no_grad():
			_x = torch.rand(size=(1,) + self.input_shape)
			_y = self.forward(_x)
			self.output_shape = _y.shape[1:]
	
	@torch.no_grad()
	def get_neural_response_strengths(self, x:x_batch_type) -> list[torch.Tensor]:
		raise NotImplementedError()
	