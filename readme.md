# Description
This **Python** project details observing distributions of error and bias across a large range of feature-imbalances and quantile views. It defines environment generation tools (`environment_generation` package) with an entity tracking problem space with customizable motion models (`target_generation` and `pathfinding` packages). Convolutional encoder-decoder models with residuals (`models` package) are trained (`training` package) to predict the distribution of motion of the entity within the environment. These motion distributions are weighted heatmaps which emulate where an entity is likely to be after some interval defined via by controlling the shape of the feature-imbalance distribution. Observations of multiple configurations of feature-imbalance and quantile views are used to define distributions of error and bias and are analyzed (`analytics` package) to compute similarity across different loss functions and applied motion model.

# Setup Guide

## Python Versioning Notes
`Python 3.12` is used, but this project should work with 3.10+ (untested).

## Pip Install Required Packages
See the `requirements.txt` file for a versioned list of dependencies.

This project uses the following packages:

* `PyTorch` - Used for defining trainable neural network modules and layers.
* `PyTorch Lightning` - PyTorch training framework.
* `torchmetrics` - Tools for model analysis.
* `TensorBoard` - Training logging and plotting framework.
* `rich` - Rich text for console-based logs.
* `NumPy` - General numerical library.
* `SciPy` - Used for the implementation of the Jensen-Shannon Distance metric.
* `Xarray` - Labeled arrays and matrices, used to help verify correct metrics processing.
* `matplotlib` - General plotting library.
* `seaborn` - Heatmap plotting library.
* `noise` - Used to generate perlin noise maps. See the following `Misc Required Tools` section for critical install information.

## Misc Required Tools
The `noise` package (used for the environment generation) is officially only supported for `Python 3.4`, leveraging C++ build tools included with that version. For later Python versions, `Microsoft Visual C++ 14.0+ Build Tools` are required to build this package.

We used the Microsoft C++ Build Tools (installer: `https://aka.ms/vs/17/release/vs_BuildTools.exe`) and use the `Desktop development with C++` workload. Note that installing the latest `MSVC v143 - VS 2022 C++ x64/x86 build tools` as an individual component is not sufficient as there are missing header files the compiler needs to reference.

# Usage Guide
This usage guide runs through how to generate environments (Step 1), define custom motion models (Optional Step 2), train models (Step 3), and analyze results (Step 4). Each step will breakdown the required configurations and supported modifications, which can require editing the code in specific places.

Note all setting files are expected to be JSON-formatted, and are loaded into their respective settings objects via key-word unpacking a dictionary.

## Python Path Configuration
All scripts include the `src` directory in the python path. Visual Studio Code with run configurations parameterizing the environment can be used: `"env": { "PYTHONPATH": "${workspaceFolder}/src" }`, where the `workspaceFolder` reference is the path of the folder opened in VS Code. To have `workspaceFolder` correctly pointed to this project, go to `file > Open Folder...` and navigate to and select the `Synthetic Entity Tracking` project folder.

## Training Setup Instructions
### Step 1 - Generating Environments
Run the `/src/environment_generation/generate_perlin_noisemaps.py` script with the `--settings` argument pointing to the environment generation settings. See the VS Code run configuration `Build Environments`, which uses `--settings=settings/perlin_environment_generation.json`,  for examples on how to run the script.

#### Environment Generation Settings:
* `seed` (int) - RNG seed.
* `size` (int) - Square size of the environment (height and width).
* `n_samples` (int) - Number of samples to generate.
* `n_examples` (int) - Number of samples to convert into example images. Selects the first `n_examples` samples.
* `env_data_output_file` (str) - NumPy archive filepath for the generated files.
* `examples_output_dir` (str) - Output directory for the examples.
* `perlin_scale_inverse` (float) - Inverse scale used for the Perlin noisemaps.
* `perlin_octaves_inverse` (float) - Inverse octaves used for the Perlin noisemaps.

### Step 2 - Generating Baselines
Run the `/src/target_generation/generate_baselines.py` script with the `--settings` argument pointing to the target generation settings and the `--motion_model` defining the desired motion model baseline (`smm` or `imm`). See the VS Code run configurations `Build Baselines - SMM` and `Build Baselines - IMM` for examples on how to run the script.

Note that only the baselines need to be generated for the `imm` and `smm` as the density-shape map has been pre-computed. Refer to the `imm` and `smm` dataset files in the `dataset_util` package to see the computed mappings.

### Step 3 - Training and Data Collection
Training is done using the `/src/train_mm_models.py` script, which is configured by the CMDL args to select the correct motion model and settings. See the VS Code run configuration `Run Training - SMM` and `Run Training - IMM` for examples on how to run the script. Note the feature-imbalances are referred to as pixel densities within the code, as they correspond to how dense the image is. These values are also sometimes treated as the "expected pixel value" (epxv) as well due to the evolving language used during development.

Running the script once will train models on ninety-nine feature-imbalances `[0.01, 0.99]` with step sizes of `0.01`, across four loss functions (MSE, BCE, Huber, and the novel BTE), and over the two provided motion models (SMM and IMM). This results in a total of **792** models. Running the script a second time will then collect the test metrics. If needing to re-train the models, the old models must be cleared out.

While a two-staged script seems odd, that's because it is. But it works and changing it would have involved copy-pasting most of the code anyways.

Testing only after all **792** models finish training has a purpose, though: because both training and testing are time-intensive processes, it allows fully finishing one before starting the other. This allowed for faster iterative development and debugging.

#### Training Settings
* `seed` (int) - RNG seed.
* `env_dataset_file` (str) - Filepath to the archive of environments.
* `signals_dataset_file` (str) - Filepath to the archive of grid-encoded entity positions.
* `dataset_file` (str) - Filepath to the archive of target baselines.
* `training_output_dir` (str) - Base output directory for training results.
* `rel_tensorboard_logs_dir` (str) - Output directory (within `training_output_dir`) for the tensorboard logs.
* `rel_model_checkpoint_dir` (str) - Output directory (within `training_output_dir`) for the model checkpoints.
* `rel_test_results_file` (str) - Output filepath (within `training_output_dir`) for the test results.
* `data_partitions` (tuple[float,float,float]) - Training/Validation/Testing partitions. Note so long as the values are consistent, then the samples will always be allocated to the same partitions with respect to the RNG seed.
* `batch_size` (int) - Training batch size.
* `max_epochs` (int) - Maximum number of training epochs.
* `patience` (int) - Patience threshold for stopping the training early.
* `lr` (float) - Initial optimizer learning rate.

## Running Analytics
The analytics are fairly hard-coded, so any changes to the provided settings for the training processes will need to be reflected in the code here. By default, it will compile metrics from the `/output/training/` directory, but if this was changed then the script will need to be updated.

### Compiling Metrics
Once both the models are trained, run the `/src/analytics/build_test_result_metrics.py` script. This script compiles all of the tracked metrics into an aggregated tensor that is more useful for the analysis. This tensor is saved as a .netcdf file, which is an `xarray` structure for keeping track of array dimension-labels and axis values, which the subsequent plots/analytics reference.

The dimensions of the tensor are structured as follows: ["metric", "loss_name", "epx", "exclusive_quantile"]. Axis values along each of these dimensions is given as follows:
* `metric` - ["measured_fp", "measured_fn", "counted_fp", "counted_fn", "measured_sums", "measured_diffs", "counted_diffs", "measured_symm_sums", "measured_symm_diffs", "counted_symm_diffs"]
* `loss_name` - ['bce', 'huber', 'mse', 'bte']
* `epx` - [0.01, ..., 0.99] in steps of 0.01. Note epx = expected pixel value and refers to the feature-imbalance.
* `exclusive_quantile` - [-0.99, ..., 0, ..., 0.99] in steps of 0.01.

The metric dimensions details many statistics, but only some of which are featured. Detailed axis values are given below:
* `measured_fp` - Measure of false-positives (aka positive errors or overestimates).
* `measured_fn` - Measure of false-negatives (aka negative errors or underestimates).
* `counted_fp` - Instance-count of false-positives.
* `counted_fn` - Instance-count of false-negatives.
* `measured_sums` - Aggregated sum of measured errors (`measured_fp + measured_fn`).
* `measured_diffs` - Aggregated difference of measured errors (`measured_fp - measured_fn`).
* `counted_diffs` - Aggregated count-difference of false-positives versus false-negatives (`counted_fp - counted_fn`)
* `measured_symm_sums` - Aggregated sum of measured errors with a reversed component (`measured_fp + reverse(measured_fn)`).
* `measured_symm_diffs` - Aggregated difference of measured errors with a reversed component (`measured_fp - reverse(measured_fn)`).
* `counted_symm_diffs` - Aggregated count-difference of falsive-positives versus a reversed count of false-negatives (`counted_fp - reverse(counted_fn)`).

The *symm* (symmetric) metrics are interesting while simultaneously difficult to infer meaning from. They effectively detail how symmetrical the distrubtions are across the range of feature-imbalances to identify meta-behaviors.

The two primary metrics used are the **`measured_sums`** and **`measured_diffs`**, which respectively correspond to the mean-absolute error (MAE) and mean-bias error (MBE). However, the polynomial regression uses the **`measured_fp`** and **`measured_fn`** as the can be arranged to compose either the `measured_sums` and `measured_diffs`.

### Heatmap Plotting
Running the `/src/analytics/plot_metric_heatmaps.py` script will generate contour plots detailing the relationship both of MAE and MBE with respect to the "loss_name", "epx", and "exclusive_quantile" dimensions.

### JS Distances
Running the `/src/analytics/js_distances.py` script will generate the JSD tables that evaluate both MAE and MBE distribution similarities, averaging across the "exclusive_quantile" dimension, with respect to the "loss_name" and "epx" dimensions. Similarities are additionally done between problem spaces (stored in two different compiled files).

### Polynomial Regression
Running the `/src/calc_metric_2d_linear_regression.py` script will compute a series of polynomial regressions on the distributions of `measured_sums` and `measured_diffs` with respect to the "epx" and "exclusive_quantile" axes.

# Defining New Motion Model (Optional)
The basis of the weighted heatmaps are path-finding algorithms that, for some environment (a 2d grid), compute the least-costs travel costs from a starting position to every other position (defined within the code as the *flow*). Motion models detail the transfer- and path-cost functions that then modulate how costs are computed over a path.

## Persistent Storage Structure
As each dataset is a function of the feature-imbalance, thoroughly covering a large range of feature-imbalances requires an equally large number of datasets. Each feature-imbalance is mapped to a shape parameter, which is used to exponentiate baseline targets. Instead of storing all of the mapped datasets, just the baseline can be stored. 

Consider a baseline example `baseline_y`, which is the computed *flow* of an environment with respect to a randomized entity position. While, it is computationally inefficient to perform `y = baseline_y ** shape`, there is a logarithm-trick to improve performance. Instead, a more computationally friendly `y = exp(shape * log_y)` can be used instead. Thus by storing the natural logs of the baseline, we can efficiently compute any feature-imbalance while minimizing the storage requirements.

Note that for a path-finding algorithm, the cost to travel to the starting node is `0`, which is undefined for logs! However, since in NumPy `exp(-inf) = 0` and the least-cost to the starting position will **always be 0**, regardless of the shape or motion model (at least with the SMM and IMM), then this identity can be leveraged.

By masking the baseline for the starting position `mask = baseline_y == 0`, **which assumes reaching all other nodes has a nonzero cost,** then `log_y[mask] = -inf` and `log_y[~mask] = log(baseline_y[~mask])`. When loading the data, the shape parameter for the specific feature-imbalance is multiplied against the stored-version and mapped through the exponential function: `y = exp(shape * log_target)`.

## Setup Instruction
### Step 1 - Define New Path-Cost Function
Inside of the `/src/target_generation/graphs.py` script, define a new `NDArrayGraph2D` class as a new motion model. See the `NDArrayGraph2D_Inertial` class therein for an example of the IMM's implementation.

Optionally, go to the `/src/target_generation/settings.py` script and define a new settings class defining the parameters required for this motion model. See the existing class definitions there for examples. These settings additionally include information used throughout the target generation process, such as where to save examples.

Note the one requirement of this new path-cost function is that only the cost to the entity's starting position is zero.

#### Target Generation Settings (Base)
* `seed` (int) - RNG seed.
* `n_examples` (int) - Number of samples to create an image example of.
* `environments_dataset_file` (str) - Filepath to the archive of environments. Note the number of signals and targets generated is equal to the length of this archive. 
* `motion_models_dataset_file` (str) - Output filepath for the built target baselines.
* `raw_signals_dataset_file` (str) - Output filepath for the randomly computed, raw index-coordinate positions of the entities (signals).
* `signals_dataset_file` (str) - Output filepath for the grid-encoded entity positions. 
* `motion_model_examples_output_dir` (str) - Output directory for target baseline examples.
* `signal_examples_output_dir` (str) - Output directory for grid-encoded entity position examples.

### Step 2 - Generate Baselines
This step builds the baseline targets for a given motion model. Additionally, if they do not already exist, it generates the entity positions (interchangeably referred to as signals) and exports those as well for future use. While the RNG is seeded for reproducible results, this reuse is not strictly necessary, but is done for consistency's sake.

Go to the `/src/target_generation/generate_baselines.py` script. Add in a new function `generate_<motion model name>_baseline` (refer to the existing ones for the `smm` or `imm` for reference). Configure the `parser::ArgumentParser` to include CMDL options for specifying the new motion model and `main` to create the correct settings and `generate_<motion model name>_baseline` function.

See the VS Code run configurations `Build Baselines - SMM` and `Build Baselines - IMM` for examples on how to run the script.

### Step 3 - Mapping Distribution Shape to Feature-Imbalance
This step computes the 1:1 mapping of the feature-imbalance with the shape parameter for the selected motion model. Note feature-imbalance is interchangeably used with the term "density", referring to the expected pixel density (epxv).

To control the feature-imbalance, a range of exponents is mapped modulate feature-imbalances. Go to the `/src/target_generation/compute_shape_params.py` script and update it, like in Step 2, to use the settings of the motion model you used. Optionally, you may also change the range of feature-imbalances by modifying the `target_means` (the targeted feature-imbalances) variable in the `main` function. Running the script will generate a series of exponents for each feature-imbalance.

See the VS Code run configurations `Compute Shape Params - SMM` and `Compute Shape Params - IMM` for examples on how to run the script.

### Step 4 - Define Datasets
Within the `dataset_util` package, create a new dataset class for the given motion model that inherits from the `MMDataset` in `/src/dataset_util/mm_dataset.py`. Refer to the `smm_dataset.py` and `imm_dataset.py` files in the same package for examples. Use the density-shape mappings computed in Step 3 for defining the density-shape map in the new file. It is important that the density-shape map has a distinct name, as it is frequently imported into other scripts alongside the density-shape maps from the SMM and IMM when dynamically choosing which to run.

### Step 5 - Update the Trainer
Within the `/src/train_mm_models.py` script, inside the `if __name__ == "__main__"` block, update the arguments and switch-case to support the new motion model.

### Step 6 - Update the Analytics
The analytics are more hard-coded, especially for the various plotting tools. Additionally, because a lot of the plots are comparisons between the SMM and IMM, there is limited flexibility with supporting 3+ motion models. However, for comparative analysis to an existing, it should be viable to replace references to either the SMM or IMM with the new motion model.
