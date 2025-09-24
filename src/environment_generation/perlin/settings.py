from dataclasses import dataclass

from environment_generation.settings import EnvGenSettings


@dataclass
class PerlinEnvGenSettings(EnvGenSettings):
	perlin_scale_inverse:int
	perlin_octaves_inverse:int
