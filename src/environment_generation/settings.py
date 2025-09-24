from dataclasses import dataclass

@dataclass
class EnvGenSettings:
	seed:int
	size:int
	n_samples:int
	n_examples:int
	env_data_output_file:str
	examples_output_dir:str
