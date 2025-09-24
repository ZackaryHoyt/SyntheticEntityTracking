from torch import nn

from models.models import CustomModule, x_batch_type, y_pred_batch_type


class ResNet(CustomModule):
	def __init__(self, input_shape:tuple[int,int,int], down_channels:list[int], out_channels:int|None=None, *args, **kwargs):
		super().__init__(input_shape, *args, **kwargs)

		# Output Channel Size
		assert len(input_shape) == 3
		if out_channels is None:
			out_channels = input_shape[0]
		
		# Entry Block
		entry_block_channels = [down_channels[0]]
		self.entry_blocks = self._make_entry_block(input_shape[0], entry_block_channels[0])

		# Downsampling Block
		down_block1_channels = down_channels[1:]
		self.down_blocks1 = self._make_block_list_down(entry_block_channels[-1], down_block1_channels)
		self.down_residual_blocks1 = self._make_block_list_down_residual(entry_block_channels[-1], down_block1_channels)

		# Upsampling Block
		up_channels = down_channels[::-1]
		up_block1_channels = up_channels[1:]
		self.up_blocks1 = self._make_block_list_up(down_channels[-1], up_block1_channels)
		self.up_residual_blocks1 = self._make_block_list_up_residual(down_channels[-1], up_block1_channels)

		# Classification Block
		self.output_block = self._make_output_block(up_block1_channels[-1], out_channels)

		# Initialize Output Shape
		self.initialize_output_shape()
	
	def _make_entry_block(self, in_channels:int, out_channels:int) -> nn.Sequential:
		return nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
			nn.ReLU()
		)

	def _make_down_block(self, in_channels:int, out_channels:int) -> nn.Sequential:
		return nn.Sequential(
			nn.ReLU(),
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
			nn.ReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		)
	
	def _make_block_list_down(self, in_channels:int, out_channels:list[int]) -> nn.ModuleList:
		return self._make_module_list(in_channels, out_channels, self._make_down_block)
	
	def _make_down_residual_block(self, in_channels:int, out_channels:int) -> nn.Sequential:
		return nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
		)
	
	def _make_block_list_down_residual(self, in_channels:int, out_channels:list[int]) -> nn.ModuleList:
		return self._make_module_list(in_channels, out_channels, self._make_down_residual_block)
	
	def _make_up_block(self, in_channels:int, out_channels:int) -> nn.Sequential:
		return nn.Sequential(
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
			nn.ReLU(),
			nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
			nn.Upsample(scale_factor=2, mode='nearest')
		)
	
	def _make_block_list_up(self, in_channels:int, out_channels:list[int]):
		return self._make_module_list(in_channels, out_channels, self._make_up_block)

	def _make_up_residual_block(self, in_channels:int, out_channels:int) -> nn.Sequential:
		return nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
		)
	
	def _make_block_list_up_residual(self, in_channels:int, out_channels:list[int]) -> nn.ModuleList:
		return self._make_module_list(in_channels, out_channels, self._make_up_residual_block)
	
	def _make_module_list(self, in_channels:int, out_channels_list:list[int], f_make_blocks) -> nn.ModuleList:
		out_channels_list = [in_channels,] + out_channels_list
		contents = []
		for i in range(1, len(out_channels_list)):
			in_channels_i = out_channels_list[i - 1]
			out_channels_i = out_channels_list[i]
			contents.append(f_make_blocks(in_channels_i, out_channels_i))
		return nn.ModuleList(contents)

	def _make_output_block(self, in_channels:int, out_channels:int) -> nn.Sequential:
		return nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.Sigmoid()
		)

	def forward(self, x:x_batch_type) -> y_pred_batch_type:
		x = self.entry_blocks(x)
		for down_block, down_residual_block in zip(self.down_blocks1, self.down_residual_blocks1):
			x = down_block(x) + down_residual_block(x)
		for up_block, up_residual_block in zip(self.up_blocks1, self.up_residual_blocks1):
			x = up_block(x) + up_residual_block(x)
		x = self.output_block(x)
		return x


if __name__ == "__main__":
	import torch
	# Example instantiation + forward pass
	model = ResNet(
		input_shape=(2, 128, 128),
		down_channels=[16, 16, 32, 32, 64, 64, 128],
		out_channels=1
	)
	print(model)

	# create a dummy batch of 4 RGB images of size 128×128
	x = torch.randn(4, 2, 128, 128)
	y = model(x)

	print(f"  Input shape:  {x.shape}")
	print(f"  Output shape: {y.shape}")
