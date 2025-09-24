from typing import Sequence, Union

import numpy as np
import torch

from dataset_util.npz_dataset import NpzDataset


class MMDataset(NpzDataset):
	def __init__(self, shape_map:dict[float,float], xs_files:Union[str,Sequence[str]], ys_file:Union[str,Sequence[str]], epsilon:float=1) -> None:
		super().__init__(xs_files, ys_file)
		self.epsilon = epsilon
		self.density = MMDataset.find_density(epsilon, shape_map)

	def __getitem__(self, idx:int) -> tuple[torch.Tensor, torch.Tensor]:
		x, y = super().__getitem__(idx)
		y = torch.exp(self.epsilon * y)
		return x, y
	
	@staticmethod
	def find_density(epsilon:float, shape_map:dict[float,float]) -> float:
		mapped_density = [k for k,v in shape_map.items() if v == epsilon]
		if mapped_density:
			return mapped_density[0]
		raise ValueError()

	@classmethod
	def from_density(cls, xs_files:Union[str,Sequence[str]], ys_file:Union[str,Sequence[str]], density:float) -> 'MMDataset':
		raise NotImplementedError()
