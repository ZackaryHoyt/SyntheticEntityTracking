from dataclasses import dataclass

@dataclass
class TargetGenSettings:
	seed:int
	n_examples:int
	environments_dataset_file:str
	motion_models_dataset_file:str
	signals_dataset_file:str
	raw_signals_dataset_file:str
	motion_model_examples_output_dir:str
	signal_examples_output_dir:str
