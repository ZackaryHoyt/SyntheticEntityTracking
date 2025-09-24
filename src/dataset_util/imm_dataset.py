from typing import Sequence, Union

from dataset_util.mm_dataset import MMDataset

"""
This lookup table is built for Inertial Motion Models (IMM) using decay=2 on 128x128
normalized perlin noise environments (inverse_scale=4, inverse_octaves=16). The shape
parameter controls the (estimated) expected pixel value (epxv) by exponentially scaling
the data, controlling the degree of bias within the motion model.

	epxv = average(image ** shape)

Because this operation is relatively costly, the pixels are stored as a natural-log form.
Then the array can be more efficiently scaled using the exponential function:

	log_px = log(px) if px > 0 else -inf
	image ** shape = e ** (shape * log(image))

Note the exception of px=0 (it is assumed px>=0), as log(0) is undefined. However, the
numpy library's exp-function will map -inf to 0.

	np.exp(-inf) = 0

Example usage for this table:
	expv = 0.31
	scaled_arr = np.exp(imm_epxv_shape_map[expv] * arr)
"""
imm_epxv_shape_map = {
	0.01: 14.194952965, 0.02: 10.209885359, 0.03: 8.348453164,  0.04: 7.194273710,  0.05: 6.381502867,
	0.06: 5.765641749,  0.07: 5.276139945,  0.08: 4.873744428,  0.09: 4.534571469,  0.10: 4.243106991, 
	0.11: 3.988757074,  0.12: 3.763995871,  0.13: 3.563303113,  0.14: 3.382520795,  0.15: 3.218446285,
	0.16: 3.068565413,  0.17: 2.930871725,  0.18: 2.803740829,  0.19: 2.685840890,  0.20: 2.576067664, 
	0.21: 2.473496422,  0.22: 2.377345733,  0.23: 2.286949918,  0.24: 2.201737657,  0.25: 2.121215373,
	0.26: 2.044953950,  0.27: 1.972578257,  0.28: 1.903758597,  0.29: 1.838203833,  0.30: 1.775655709, 
	0.31: 1.715884186,  0.32: 1.658683550,  0.33: 1.603869185,  0.34: 1.551274844,  0.35: 1.500750322,
	0.36: 1.452159524,  0.37: 1.405378781,  0.38: 1.360295404,  0.39: 1.316806465,  0.40: 1.274817716, 
	0.41: 1.234242667,  0.42: 1.195001766,  0.43: 1.157021713,  0.44: 1.120234810,  0.45: 1.084578432,
	0.46: 1.049994547,  0.47: 1.016429279,  0.48: 0.983832531,  0.49: 0.952157650,  0.50: 0.921361124, 
	
	0.51: 0.891402315,  0.52: 0.862243216,  0.53: 0.833848232,  0.54: 0.806183983,  0.55: 0.779219138,
	0.56: 0.752924241,  0.57: 0.727271579,  0.58: 0.702235045,  0.59: 0.677790022,  0.60: 0.653913267, 
	0.61: 0.630582826,  0.62: 0.607777931,  0.63: 0.585478924,  0.64: 0.563667180,  0.65: 0.542325035,
	0.66: 0.521435726,  0.67: 0.500983337,  0.68: 0.480952729,  0.69: 0.461329507,  0.70: 0.442099966, 
	0.71: 0.423251050,  0.72: 0.404770313,  0.73: 0.386645881,  0.74: 0.368866425,  0.75: 0.351421114,
	0.76: 0.334299606,  0.77: 0.317492006,  0.78: 0.300988847,  0.79: 0.284781065,  0.80: 0.268859977, 
	0.81: 0.253217261,  0.82: 0.237844939,  0.83: 0.222735353,  0.84: 0.207881154,  0.85: 0.193275286,
	0.86: 0.178910965,  0.87: 0.164781675,  0.88: 0.150881146,  0.89: 0.137203349,  0.90: 0.123742480, 
	0.91: 0.110492948,  0.92: 0.097449373,  0.93: 0.084606565,  0.94: 0.071959524,  0.95: 0.059503430,
	0.96: 0.047233630,  0.97: 0.035145635,  0.98: 0.023235114,  0.99: 0.011497882,  
}

class IMMDataset(MMDataset):
	def __init__(self, xs_files:Union[str,Sequence[str]], ys_file:Union[str,Sequence[str]], epsilon:float=1) -> None:
		super().__init__(imm_epxv_shape_map, xs_files, ys_file, epsilon)

	@classmethod
	def from_density(cls, xs_files:Union[str,Sequence[str]], ys_file:Union[str,Sequence[str]], density:float) -> 'IMMDataset':
		return cls(xs_files, ys_file, imm_epxv_shape_map[density])
	