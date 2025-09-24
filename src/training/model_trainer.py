import pytorch_lightning as pl
import torch
from torch.optim.adam import Adam
from torchmetrics import Metric, MetricCollection

from models.models import CustomModule


class ModelTrainer(pl.LightningModule):
	def __init__(self, model:CustomModule, criterion:torch.nn.Module, metrics_dict:dict[str, Metric], lr:float):
		super().__init__()
		self.save_hyperparameters("lr")
		
		self.model = model
		self.criterion = criterion

		# Automatic metric logging
		self.train_metrics = MetricCollection(metrics_dict, prefix="train_") # type: ignore
		self.val_metrics   = MetricCollection(metrics_dict, prefix="val_") # type: ignore
		self.test_metrics  = MetricCollection(metrics_dict, prefix="test_") # type: ignore

	def forward(self, x_batch:torch.Tensor) -> torch.Tensor:
		return self.model.forward(x_batch)

	def configure_optimizers(self):
		return Adam(self.parameters(), lr=self.hparams['lr'])
	
	def _step(self, batch:tuple[torch.Tensor, torch.Tensor], batch_idx:int):
		x_batch, y_targ_batch = batch
		y_pred_batch:torch.Tensor = self(x_batch)
		loss_batch:torch.Tensor = self.criterion(y_pred_batch, y_targ_batch)
		return y_pred_batch, y_targ_batch, loss_batch.mean()

	def training_step(self, batch:tuple[torch.Tensor, torch.Tensor], batch_idx:int):
		y_pred_batch, y_targ_batch, loss_batch = self._step(batch, batch_idx)
		self.log('train_loss', loss_batch, on_step=True, on_epoch=True, prog_bar=True)
		self.log_dict(self.train_metrics(y_pred_batch, y_targ_batch), on_step=False, on_epoch=True)
		return loss_batch

	def validation_step(self, batch:tuple[torch.Tensor, torch.Tensor], batch_idx:int):
		y_pred_batch, y_targ_batch, loss_batch = self._step(batch, batch_idx)
		self.log('val_loss', loss_batch, on_epoch=True, prog_bar=True)
		self.log_dict(self.val_metrics(y_pred_batch, y_targ_batch), on_step=False, on_epoch=True)

	def test_step(self, batch:tuple[torch.Tensor, torch.Tensor], batch_idx:int):
		y_pred_batch, y_targ_batch, loss_batch = self._step(batch, batch_idx)
		self.log('test_loss', loss_batch, on_epoch=True, prog_bar=False)
		self.log_dict(self.test_metrics(y_pred_batch, y_targ_batch), on_step=False, on_epoch=True)
