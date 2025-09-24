import json
import os
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary.model_summary import get_human_readable_count
from torch.utils.data import DataLoader, random_split
from torchmetrics import Metric

import training.functional_metrics as functional_metrics
import training.functional_modules as functional_modules
from dataset_util.mm_dataset import MMDataset
from models.models import CustomModule
from training.model_trainer import ModelTrainer

@dataclass
class TrainingSettings:
	seed:int
	env_dataset_file:str
	dataset_file:str
	signals_dataset_file:str
	training_output_dir:str
	rel_tensorboard_logs_dir:str
	rel_model_checkpoint_dir:str
	rel_test_results_file:str
	data_partitions:tuple[float,float,float]

	batch_size:int
	max_epochs:int
	lr:float
	patience:int

	def __post_init__(self) -> None:
		assert len(self.data_partitions) == 3 and sum(self.data_partitions) == 1
		assert self.batch_size > 0
		assert self.max_epochs == -1 or self.max_epochs > 0
		assert self.lr > 0
		assert self.patience > 0


def main(motion_model_name:str, settings:TrainingSettings, dataset:MMDataset, loss_name:str, model:CustomModule) -> None:
	pl.seed_everything(settings.seed)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model_size_str = get_human_readable_count(sum(p.numel() for p in model.parameters() if p.requires_grad)).replace(" ", "")
	training_label = f"{model.__class__.__name__}-{model_size_str}-{loss_name}-{dataset.density:.2f}"

	print("=" * len(training_label))
	print(training_label)
	print("=" * len(training_label))

	base_output_dir = f"{settings.training_output_dir}/{training_label}"
	
	logs_dir = f"{base_output_dir}/{settings.rel_tensorboard_logs_dir}"
	checkpoints_dir = f"{base_output_dir}/{settings.rel_model_checkpoint_dir}"

	# Dataset and DataLoaders
	train_dataset, val_dataset, test_dataset = random_split(dataset, settings.data_partitions, generator=torch.Generator().manual_seed(settings.seed))
	train_dataloader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
	val_dataloader   = DataLoader(val_dataset, batch_size=settings.batch_size)
	test_dataloader  = DataLoader(test_dataset, batch_size=settings.batch_size)
	
	# Logger and Callbacks
	tb_logger = TensorBoardLogger(save_dir=logs_dir, name=motion_model_name)

	checkpoint_cb = ModelCheckpoint(
		dirpath=checkpoints_dir,
		monitor='val_loss',
		mode='min',
		save_top_k=1
	)
	early_stop_cb = EarlyStopping(
		monitor='val_loss',
		patience=settings.patience,
		verbose=False,
		mode='min'
	)

	criterions_dict:dict[str, torch.nn.Module] = {
		'bce': functional_modules.BCELossND(),
		'mse': functional_modules.MSELossND(),
		'bte': functional_modules.BTELossND(),
		'huber': functional_modules.HuberLossND(),
	}
	criterion = criterions_dict[loss_name]

	import numpy as np
	quantiles = (np.arange(1, 101) / 100).tolist()
	quantile_strs = [str(quantile).replace('.', '_') for quantile in quantiles]
	metrics = [
		{
			f"qlower_size-{q_str}": functional_metrics.QuantileSliceSizeMetricND(q),
			f"fp_qlower-{q_str}":   functional_metrics.MeasureFalsePositivesMetricND(q),
			f"fn_qlower-{q_str}":   functional_metrics.MeasureFalseNegativesMetricND(q),
			f"fpc_qlower-{q_str}":  functional_metrics.CountFalsePositivesMetricND(q),
			f"fnc_qlower-{q_str}":  functional_metrics.CountFalseNegativesMetricND(q),

			f"qupper_size-{q_str}": functional_metrics.QuantileSliceSizeMetricND(-q),
			f"fp_qupper-{q_str}":   functional_metrics.MeasureFalsePositivesMetricND(-q),
			f"fn_qupper-{q_str}":   functional_metrics.MeasureFalseNegativesMetricND(-q),
			f"fpc_qupper-{q_str}":  functional_metrics.CountFalsePositivesMetricND(-q),
			f"fnc_qupper-{q_str}":  functional_metrics.CountFalseNegativesMetricND(-q),
		} for q, q_str in zip(quantiles, quantile_strs)
	]

	test_results_file = f"{base_output_dir}/{settings.rel_test_results_file}"
	if os.path.exists(test_results_file):
		from functools import reduce
		metrics_dict:dict[str, Metric] = {
			**reduce(dict.__or__, metrics)
		}
	else:
		metrics_dict = { }
	
	# Model Trainer
	model_trainer = ModelTrainer(model=model, criterion=criterion, metrics_dict=metrics_dict, lr=settings.lr).to(device)

	# PyLightning Trainer
	trainer = pl.Trainer(
		max_epochs=settings.max_epochs,
		logger=tb_logger,
		callbacks=[checkpoint_cb, early_stop_cb, RichProgressBar()],
		accelerator='auto',
		devices='auto'
	)

	# Update Test Results if They Exist (and Skip Training)
	if os.path.exists(test_results_file):
		ckpt_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")]
		best_ckpt_path = f"{checkpoints_dir}/{ckpt_files[0]}"
		test_results = trainer.test(model_trainer, test_dataloader, ckpt_path=best_ckpt_path)
	else:
		# Fit the Model
		trainer.fit(model_trainer, train_dataloader, val_dataloader)

		# Test the Model
		test_results = trainer.test(model_trainer, test_dataloader, ckpt_path='best')

	# Export Test Results
	with open(test_results_file, 'w') as ofs:
		json.dump(test_results, ofs)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--motion_model", help="Motion model type.", default='smm', choices=['smm', 'imm'])
	parser.add_argument("--settings", help="Settings filepath.")
	args = parser.parse_args()

	motion_model_name:str = args.motion_model.lower()
	settings_filepath:str = args.settings

	print(f"Running {__file__}...")
	print(f"Configuared motion model: {motion_model_name}")
	print(f"Configured settings filepath: {settings_filepath}")

	with open(settings_filepath, 'r') as f:
		config = json.load(f)
	settings = TrainingSettings(**config)

	xs_files = [settings.env_dataset_file, settings.signals_dataset_file]
	ys_file = settings.dataset_file
	
	# Select mapped densities for the corresponding motion model.
	# Note both lists of densities are the same, this is purely for consistency.
	from dataset_util.imm_dataset import IMMDataset, imm_epxv_shape_map
	from dataset_util.smm_dataset import SMMDataset, smm_epxv_shape_map
	match(motion_model_name):
		case 'smm':
			densities = smm_epxv_shape_map.keys()
			dataset_cls = SMMDataset
		case 'imm':
			densities = imm_epxv_shape_map.keys()
			dataset_cls = IMMDataset
		case _: raise ValueError()

	pl.seed_everything(0)
	from models.resnet import ResNet
	model = ResNet(input_shape=(2,128,128), down_channels=[8, 8, 16, 16, 32, 32, 64], out_channels=1)

	from copy import deepcopy
	import itertools
	for density, loss_name in itertools.product(densities, ['bce', 'bte', 'huber', 'mse']):
		initial_state = deepcopy(model.state_dict())
		main(motion_model_name=motion_model_name,
			settings=settings,
			dataset=dataset_cls.from_density(xs_files, ys_file, density),
			loss_name=loss_name,
			model=model
		)
