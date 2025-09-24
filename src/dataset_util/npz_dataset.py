from typing import Union, Sequence

import numpy as np
import torch
from numpy.lib.npyio import NpzFile
from torch.utils.data import Dataset


class NpzDataset(Dataset):
	xs_archives:list[NpzFile]
	ys_archives:list[NpzFile]

	def __init__(self, xs_files:Union[str,Sequence[str]], ys_file:Union[str,Sequence[str]]) -> None:
		self.xs_archives = NpzDataset.load_files(xs_files)
		self.ys_archives = NpzDataset.load_files(ys_file)

		assert all([len(self) == len(archive) for archive in self.xs_archives])
		assert all([len(self) == len(archive) for archive in self.ys_archives])
		
		self.sampled_data_shape, self.sampled_label_shape = [arr.shape for arr in NpzDataset.__getitem__(self, 0)]
	
	@staticmethod
	def load_files(files:Union[str,Sequence[str]]) -> list[NpzFile]:
		if isinstance(files, str):
			return [np.load(files, allow_pickle=False, mmap_mode='r')]
		else:
			return [np.load(file, allow_pickle=False, mmap_mode='r') for file in files]

	def __len__(self) -> int:
		return len(self.ys_archives[0])

	def __getitem__(self, idx:int) -> tuple[torch.Tensor, torch.Tensor]:
		# Import assumption that all npz files share the same set of keys that correspond data samples with one-another.
		x = torch.from_numpy(np.stack([archive[str(idx)] for archive in self.xs_archives]).astype(np.float32))
		y = torch.from_numpy(np.stack([archive[str(idx)] for archive in self.ys_archives]).astype(np.float32))
		return x, y
