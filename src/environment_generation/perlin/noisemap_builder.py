import itertools

import numpy as np
from noise import pnoise2

from environment_generation.perlin.settings import PerlinEnvGenSettings


def create_perlin_noisemap_builder(settings:PerlinEnvGenSettings):
	h, w = settings.size, settings.size
	perlin_scale = settings.size // settings.perlin_scale_inverse
	perlin_octaves = settings.size // settings.perlin_octaves_inverse
	_initial_offset = settings.seed

	noise_bound = np.sqrt(2 / 4)
	# Informal definition of the bounds for perlin noise are +-sqrt(# dims / 4).
	# The formal definition (afaik) does not exist, but this is a known 'sufficient' function for calculating them. 
	
	def perlin_noisemap_builder(offset:int) -> np.ndarray:
		# Calculate Perlin Noisemap Offset
		poffset = perlin_scale * (_initial_offset + offset)

		""" Perlin noisemaps are continuous, where offsets along a single axis would produce connected noisemaps.
		The 'poffset' is applied along the diagonal axis to reduce the potential of bias introduced by those shared features.
		This may not actually be problematic, but it's simple enough to implement at no real cost (other than readability).
		"""

		# Generate Noisemap
		noise = np.zeros((h, w))
		for i, j in itertools.product(range(h), range(w)):
			x = i / perlin_scale
			y = j / perlin_scale
			noise_val = pnoise2(x + poffset, y + poffset, octaves=perlin_octaves, persistence=0.5, lacunarity=2)
			noise[i, j] = noise_val

		# Standardize Noise to [0, 1]
		normalized_noise:np.ndarray = (noise + noise_bound) / (2 * noise_bound)
		
		return normalized_noise
	
	return perlin_noisemap_builder
