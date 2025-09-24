from typing import Any, Callable, Optional, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

transform_type:TypeAlias = Callable[[np.ndarray],np.ndarray]

def plot_regression(xr_loss_densities:xr.DataArray, results:dict[str,Any], exclusive_quantiles:list[float]=[-0.99, 0, 0.99]):
	loss_names = xr_loss_densities.loss_name.values
	transform:Optional[transform_type] = results['transform']
	
	fig, axes = plt.subplots(
		nrows=len(exclusive_quantiles),
		ncols=len(loss_names),
		sharex=True,
		sharey=True,
		figsize=(4 * len(loss_names), 12),
	)

	# ylim = xr_loss_densities.max()
	ylim_top = np.quantile(xr_loss_densities, 0.999).item()
	ylim_bottom = 0
	print((ylim_bottom, ylim_top))

	for j, sel_loss_name in enumerate(loss_names):
		da_j = xr_loss_densities.sel(loss_name=sel_loss_name)
		epxs = da_j.epx.values
		model:LinearRegression = results["loss_results"][sel_loss_name]["model"]
		
		# Prepare independent variables

		for i, sel_exclusive_quantile in enumerate(exclusive_quantiles):
			da_ij = da_j.sel(exclusive_quantile=sel_exclusive_quantile)
				
			X = np.column_stack((epxs, np.full_like(epxs, sel_exclusive_quantile)))
			if transform:
				X = transform(X)

			Y = da_ij.values

			ax:Axes = axes[i, j]

			# Predictions
			Y_pred = model.predict(X)

			# Plotting
			ax.scatter(epxs, Y, color='black', s=10)
			ax.plot(epxs, Y_pred, label='Linear', linestyle='--')
			ax.set_ylim(ylim_bottom, ylim_top)
			ax.set_title(f"{sel_loss_name} | eq={sel_exclusive_quantile}")
			ax.set_xlabel("epx")
			if j == 0:
				ax.set_ylabel("Density")
			ax.legend()

	plt.tight_layout()
	plt.show()

def print_results(results:dict[str,Any]):
	for loss_name, info in results["loss_results"].items():
		print(f"Loss: {loss_name} - R-squared: {info['score']:.4}")
	print("")

def print_linear_regression(results:dict[str,Any]):
	for loss_name, info in results["loss_results"].items():
		model:LinearRegression = info["model"]

		intercept = model.intercept_
		coef = model.coef_
		eq = (f"y = {intercept:.2e} + {coef[0]:.2e}*x_0 + {coef[1]:.2e}*x_1")
		print(f"[Linear] {loss_name} (R2={info['score']:.3f}): {eq}")

def print_polynomial_regression(results:dict[str,Any]):
	degree:int = results["degree"]
	polynomial_features:PolynomialFeatures = results["polynomial_features"]

	feature_names = polynomial_features.get_feature_names_out(["x_0", "x_1"])
	for loss_name, info in results["loss_results"].items():
		model:LinearRegression = info["model"]

		intercept = model.intercept_
		coefs = model.coef_
		terms = [f"{coefs[i]:.2e}*{feature_names[i]}" for i in range(len(feature_names))]
		eq = "y = " + f"{intercept:.2e}" + " + " + " + ".join(terms)
		print(f"[Poly deg={degree}] {loss_name} (R2={info['score']:.3f}): {eq}")


def _linear_regression(xr_loss_densities:xr.DataArray, transform:Optional[transform_type]=None):
	results:dict[str,Any] = { 'transform': transform }
	loss_results:dict[str,Any] = { }

	for loss_name in xr_loss_densities.loss_name.values:
		# Extract the data for the current loss_name
		da = xr_loss_densities.sel(loss_name=loss_name)
		
		# Prepare independent variables (epx, exclusive_quantile)
		epx, exclusive_quantile = np.meshgrid(da.epx.values, da.exclusive_quantile.values, indexing='ij')
		X = np.column_stack((epx.ravel(), exclusive_quantile.ravel()))
		if transform:
			X = transform(X)
		
		# Dependent variable (density values)
		Y = da.values.ravel()
		
		# Fit the linear regression model
		model = LinearRegression()
		model.fit(X, Y)
		
		# Store results
		loss_results[loss_name] = {
			"model": model,
			"coefficients": model.coef_,
			"intercept": model.intercept_,
			"score": model.score(X, Y)
		}

	results["loss_results"] = loss_results
	return results

def linear_regression(xr_loss_densities:xr.DataArray):
	return _linear_regression(xr_loss_densities, transform=None)

def polynomial_regression(xr_loss_densities:xr.DataArray, degree:int=3):
	polynomial_features = PolynomialFeatures(degree=degree)
	transform = polynomial_features.fit_transform
	results = _linear_regression(xr_loss_densities, transform=transform)
	results["degree"] = degree
	results["polynomial_features"] = polynomial_features
	return results


if __name__ == "__main__":
	print(f"Running {__file__}...")

	metrics_dir = f"analytics/metrics"
	sel_metric_values = ["measured_fp", "measured_fn"]

	for degree in [1, 2, 4, 8, 16]:
		for sel_metric_value in sel_metric_values:
			print(f"========{sel_metric_value}-degree={degree}========")
			xr_smm_test_analytics_densities = xr.load_dataarray(f"{metrics_dir}/smm-test_analytics-densities.nc").sel(metric=sel_metric_value)
			xr_imm_test_analytics_densities = xr.load_dataarray(f"{metrics_dir}/imm-test_analytics-densities.nc").sel(metric=sel_metric_value)

			xr_averaged_test_analytics_densities = (xr_smm_test_analytics_densities + xr_imm_test_analytics_densities) / 2

			polynomial_regression_results = polynomial_regression(xr_averaged_test_analytics_densities, degree)
			print_polynomial_regression(polynomial_regression_results)
			print_results(polynomial_regression_results)
			plot_regression(xr_averaged_test_analytics_densities, polynomial_regression_results)
