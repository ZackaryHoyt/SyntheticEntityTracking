import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

from dataset_util.arr_util import minmax_normalization


def save_as_image(arr:np.ndarray, file:str, c:int=255):
	arr = c * minmax_normalization(arr)
	Image.fromarray(arr.astype(np.uint8), 'L').save(file)

def save_as_heatmap(arr:np.ndarray, file:str, normalize:bool=True, cbar:bool=True):
	if normalize:
		arr = (arr - arr.min()) / (arr.max() - arr.min())
	sns.heatmap(arr.T,
		cmap='viridis',
		square=True,
		cbar=cbar,
		xticklabels=[],
		yticklabels=[],
		vmin=arr.min(),
		vmax=arr.max(),
		cbar_kws={"shrink": 1},
	)
	plt.tight_layout()
	plt.savefig(file, dpi=300)
	plt.close()

def create_and_save_contour(arr:np.ndarray, file:str, cm_name:str='viridis', render_grid=False):
	h,w = arr.shape
	nx = np.arange(w)
	ny = np.arange(h)
	x, y = np.meshgrid(nx, ny)
	
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(111)
	ax.contour(x, y[::-1], arr, cmap=cm_name)
	# y[::-1] corrects the orientation of the plot to match the noise's image
	ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
	ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
	ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
	ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
	if render_grid:
		ax.grid(which='both')
	#ax.set_xlabel('x-axis')
	#ax.set_ylabel('y-axis')
	plt.tight_layout()
	plt.savefig(file, dpi=300)
	plt.close()
	