import json
import os
import shutil

import matplotlib.pyplot as plt
import torch
from pytorch_lightning.utilities.model_summary.model_summary import get_human_readable_count
from torch.utils.data import DataLoader, random_split

import training.functional_modules as functional_modules
from models.models import CustomModule
from training.model_trainer import ModelTrainer

from train_mm_models import TrainingSettings
from dataset_util.mm_dataset import MMDataset
from dataset_util.smm_dataset import SMMDataset, smm_epxv_shape_map
from dataset_util.imm_dataset import IMMDataset, imm_epxv_shape_map


def generate_test_examples(
		motion_model_name:str,
		settings:TrainingSettings,
		dataset:MMDataset,
		loss_name:str, model:CustomModule,
		n_examples:int=1
	) -> None:
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Compose paths
	model_size_str = get_human_readable_count(sum(p.numel() for p in model.parameters() if p.requires_grad)).replace(" ", "")
	training_label = f"{model.__class__.__name__}-{model_size_str}-{loss_name}-{dataset.density:.2f}"
	base_output_dir = os.path.join(settings.training_output_dir, training_label)
	checkpoints_dir = os.path.join(base_output_dir, settings.rel_model_checkpoint_dir)

	# Load dataset and split
	_, _, test_dataset = random_split(dataset, settings.data_partitions, generator=torch.Generator().manual_seed(settings.seed))
	test_dataloader = DataLoader(test_dataset, batch_size=1)

	# Load checkpoint
	ckpt_files = sorted(f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt"))
	if not ckpt_files:
		raise FileNotFoundError("No checkpoint file found in directory:", checkpoints_dir)
	ckpt_path = os.path.join(checkpoints_dir, ckpt_files[0])

	# Load model and trainer
	criterions_dict = {
		'bce': functional_modules.BCELossND,
		'mse': functional_modules.MSELossND,
		'huber': functional_modules.HuberLossND,
		'bte': functional_modules.BTELossND,
	}
	criterion = criterions_dict[loss_name]()
	model_trainer = ModelTrainer.load_from_checkpoint(ckpt_path, model=model, criterion=criterion, metrics_dict={}, lr=settings.lr).to(device)
	model_trainer.eval()

	# Run prediction
	example_xs = []
	example_ys_pred = []
	example_ys_targ = []

	example_output_dir = os.path.join(base_output_dir, "examples")
	shutil.rmtree(example_output_dir, ignore_errors=True)
	os.makedirs(example_output_dir, exist_ok=True)

	mat_examples_dir = os.path.join(example_output_dir, "images")
	os.makedirs(mat_examples_dir, exist_ok=True)

	fig, axes = plt.subplots(nrows=2, ncols=2)

	with torch.no_grad():
		x_batch:torch.Tensor
		y_targ_batch:torch.Tensor
		for i, (x_batch, y_targ_batch) in zip(range(n_examples), test_dataloader):
			x_batch, y_targ_batch = x_batch.to(device), y_targ_batch.to(device)
			y_pred_batch:torch.Tensor = model_trainer(x_batch)

			example_xs.append(x_batch.cpu())
			example_ys_pred.append(y_pred_batch.cpu())
			example_ys_targ.append(y_targ_batch.cpu())
			
			axes[0,0].matshow(x_batch[0][0].cpu())
			axes[0,1].matshow(x_batch[0][1].cpu())
			axes[1,0].matshow(y_pred_batch[0][0].cpu())
			axes[1,1].matshow(y_targ_batch[0][0].cpu())
			plt.tight_layout()
			fig.savefig(os.path.join(mat_examples_dir, f"{motion_model_name}-{dataset.density}-example_{i}.png"))
	
	plt.close(fig)
	
	example_xs = torch.cat(example_xs)
	example_ys_pred = torch.cat(example_ys_pred)
	example_ys_targ = torch.cat(example_ys_targ)

	# Save example predictions and targets

	torch.save(example_xs, os.path.join(example_output_dir, f"{motion_model_name}-{dataset.density}-xs_example.pt"))
	torch.save(example_ys_pred, os.path.join(example_output_dir, f"{motion_model_name}-{dataset.density}-ys_pred_example.pt"))
	torch.save(example_ys_targ, os.path.join(example_output_dir, f"{motion_model_name}-{dataset.density}-ys_targ_example.pt"))

	print(f"Saved predictions and targets to {example_output_dir}")


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

	from models.resnet import ResNet
	model = ResNet(input_shape=(2,128,128), down_channels=[8, 8, 16, 16, 32, 32, 64], out_channels=1)

	match(motion_model_name):
		case 'smm':
			shape_map = smm_epxv_shape_map
			dataset_cls = SMMDataset
		case 'imm':
			shape_map = imm_epxv_shape_map
			dataset_cls = IMMDataset
		case _: raise ValueError()

	import itertools
	for density, loss_name in itertools.product(shape_map, ['bce', 'huber', 'mse', 'bte']):
		dataset = dataset_cls(
			xs_files=[settings.env_dataset_file, settings.signals_dataset_file],
			ys_file=settings.dataset_file,
			epsilon=shape_map[density]
		)
		generate_test_examples(
			motion_model_name=motion_model_name,
			settings=settings,
			dataset=dataset,
			loss_name=loss_name,
			model=model,
			n_examples=3
		)
