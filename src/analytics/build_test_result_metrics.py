import itertools
import json
import os

import numpy as np
import xarray as xr


def generate_test_results_analysis(
		base_training_dir:str,
		model_name:str,
		loss_names:list[str],
		epxs:list[float],
		quantiles:list[float]
	):

	test_metric_key_templates = [
		"test_fp_qlower-{q_str}", "test_fn_qlower-{q_str}", "test_fpc_qlower-{q_str}", "test_fnc_qlower-{q_str}",
		"test_fp_qupper-{q_str}", "test_fn_qupper-{q_str}", "test_fpc_qupper-{q_str}", "test_fnc_qupper-{q_str}",
	]

	test_metric_labels = [
		"measured_fp_qlower", "measured_fn_qlower", "counted_fp_qlower", "counted_fn_qlower",
		"measured_fp_qupper", "measured_fn_qupper", "counted_fp_qupper", "counted_fn_qupper",
		"measured_sums_qlower", "measured_diffs_qlower", "counted_diffs_qlower", "measured_symm_qlower", "measured_symm_diffs_qlower", "counted_symm_diffs_qlower",
		"measured_sums_qupper", "measured_diffs_qupper", "counted_diffs_qupper", "measured_symm_qupper", "measured_symm_diffs_qupper", "counted_symm_diffs_qupper",
	] # These labels are non-binding (excepting their total count), but are useful still as a descriptive reference.

	n_test_metric_labels = len(test_metric_labels)
	n_losses = len(loss_names)
	n_epxs = len(epxs)
	n_quantiles = len(quantiles)

	test_results = np.zeros([n_test_metric_labels, n_losses, n_epxs, n_quantiles])

	for i, j in itertools.product(range(n_losses), range(n_epxs)):
		train_loss_name = loss_names[i]
		test_results_file = f"{base_training_dir}/{model_name}-{train_loss_name}-{epxs[j]:.2f}/test_results.json"
		with open(test_results_file, 'r') as f:
			test_results_dict = json.load(f)[0]
		for k, q in enumerate(quantiles):
			q_str = str(q).replace('.', '_') # pl doesn't allow decimals in the metric name, so they were replaced with '_'.
			test_results[0:8,i,j,k] = [
				test_results_dict[metric_keys_template.format(q_str=q_str)]
					for metric_keys_template in test_metric_key_templates
			]

	test_results[ 8,:,:,:] = test_results[0,:,:,:] + test_results[1,:,:,:]
	test_results[ 9,:,:,:] = test_results[0,:,:,:] - test_results[1,:,:,:]
	test_results[10,:,:,:] = test_results[2,:,:,:] - test_results[3,:,:,:]
	test_results[11,:,:,:] = test_results[0,:,:,:] + test_results[1,:,::-1,:]
	test_results[12,:,:,:] = test_results[0,:,:,:] - test_results[1,:,::-1,:]
	test_results[13,:,:,:] = test_results[2,:,:,:] - test_results[3,:,::-1,:]

	test_results[14,:,:,:] = test_results[4,:,:,:] + test_results[5,:,:,:]
	test_results[15,:,:,:] = test_results[4,:,:,:] - test_results[5,:,:,:]
	test_results[16,:,:,:] = test_results[6,:,:,:] - test_results[7,:,:,:]
	test_results[17,:,:,:] = test_results[4,:,:,:] + test_results[5,:,::-1,:]
	test_results[18,:,:,:] = test_results[4,:,:,:] - test_results[5,:,::-1,:]
	test_results[19,:,:,:] = test_results[6,:,:,:] - test_results[7,:,::-1,:]

	metric_labels_type_error = ["measured_fp", "measured_fn", "counted_fp", "counted_fn"]
	metric_labels_agg = ["measured_sums", "measured_diffs", "counted_diffs", "measured_symm_sums", "measured_symm_diffs", "counted_symm_diffs"]

	n_metric_labels_type_error = len(metric_labels_type_error)
	n_metric_labels_agg = len(metric_labels_agg)

	metric_labels = metric_labels_type_error + metric_labels_agg
	n_metric_labels = len(metric_labels)

	exclusive_quantiles = np.concatenate([np.array(quantiles) - 1, np.negative(quantiles[::-1][1:]) + 1])
	# Skip the first element of the upper quantile range as the associated values are the same with the lower quantile range.

	aggregated_results = np.zeros([n_metric_labels, n_losses, n_epxs, len(exclusive_quantiles)])

	aggregated_results[:] = [
		np.concatenate([test_results[i], test_results[i + n_metric_labels_type_error,:,:,::-1][:,:,1:]], axis=-1) for i in range(0, n_metric_labels_type_error)
	] + [
		np.concatenate([test_results[i], test_results[i + n_metric_labels_agg,:,:,::-1][:,:,1:]], axis=-1) for i in range(2 * n_metric_labels_type_error, 2 * n_metric_labels_type_error + n_metric_labels_agg)
	]
	# This is a clever (which means I'm going to forget how it works later) loop that concatenates the qlower and qupper
	# (reversed with the 100% quantile item dropped) to make a seamless array connecting both quantile-selection regions.
	"""
	The qupper metrics are reversed to make the cocatenated sequence continguous:
	100% of qlower (the last element in the qlower) is equal to 100% of qupper (the last element in the qupper), so by flipping
	qupper (which is being added to the end of qlower), the two metrics will line up.
	"""

	return xr.DataArray(
		data=aggregated_results,
		dims=[
			"metric",
			"loss_name",
			"epx",
			"exclusive_quantile"
		],
		coords={
			"metric": metric_labels,
			"loss_name": loss_names,
			"epx": epxs,
			"exclusive_quantile": exclusive_quantiles
		}
	)

def main(base_training_dir:str, model_name:str, loss_names:list[str], densities:list[float], quantiles:list[float], output_dir:str, mm_name:str):
	os.makedirs(output_dir, exist_ok=True)
	xr_metrics = generate_test_results_analysis(base_training_dir, model_name, loss_names, densities, quantiles)

	xr_metrics.to_netcdf(f"{output_dir}/{mm_name.lower()}-test_analytics.nc")
	
	xr_metrics.groupby(["metric", "loss_name"])\
		.map(lambda x: (x - x.min()) / (x - x.min()).sum())\
		.transpose("metric", "loss_name", "epx", "exclusive_quantile")\
		.to_netcdf(f"{output_dir}/{mm_name.lower()}-test_analytics-densities.nc")

if __name__ == '__main__':
	print(f"Running {__file__}...")
	
	mm_names = ["smm", "imm"]

	from dataset_util.imm_dataset import \
	    imm_epxv_shape_map  # keys are same across smm and imm
	epxs = list(imm_epxv_shape_map.keys())
	epxs.sort()
	quantiles = (np.arange(1, 101) / 100).tolist() # these are the normal quantiles generated with the pytorch-lightning metrics, not the exclusive-quantiles associated with typical test results metric analysis

	output_dir = f"analytics/metrics"
	loss_names = ['bce', 'huber', 'mse', 'bte']
	model_name = "UNet2D-167K"

	for mm_name in mm_names:
		base_training_dir = f"output/training/{mm_name}"
		main(base_training_dir, model_name, loss_names, epxs, quantiles, output_dir, mm_name.upper())
