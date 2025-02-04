# basin-volume
Precisely estimating the volume of basins in neural net parameter space corresponding to interpretable behaviors


## Installation

* Use Python 3.11
* `pip install -U "jax[cuda12]"`
* `pip install -e .` or `pip install .`


## Usage

This is currently (02-04) messy and will hopefully get some cleanup soon.

See `notebooks/bigmlp_basins.ipynb` for an example.

A tidier interface (for ConvNeXt and Pythia) is available through `src/basin_volume/estimator.py`, with example usage in `scripts/expt_tuesday.py`.


## Structure

`notebooks/`: Jupyter notebooks

`.../bigmlp_basins.ipynb`: mostly-clean example of package usage

`src/basin_volume/`: package source

`.../estimator.py`: classes for managing experiments and models

`.../math.py`: integrals and such for high-dim geometry

`.../mlp.py`: MLP for `digits` example

`.../training.py`: training code for `digits` (messy)

`.../utils.py`: misc helpful tools

`.../volume.py`: core volume-estimation code

`scripts/`: Python scripts with argparse etc

`.../expt_tuesday.py`: actual script used for results in paper

`.../gpu*.sh`: batch jobs grouped by GPU

`.../train_vision.py`: training script for ConvNeXt models (adapted from [https://github.com/EleutherAI/features-across-time])

`old/`: large collection of old experiments and code (messy)

`.../basin_precondition.ipynb`: early version of this project as a giant Jupyter notebook
