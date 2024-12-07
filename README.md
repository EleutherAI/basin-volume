# basin-volume
Precisely estimating the volume of basins in neural net parameter space corresponding to interpretable behaviors


## Installation

* Use Python 3.11
* `pip install -U "jax[cuda12]"`
* `pip install -e .` or `pip install .`


## Usage

See `notebooks/kl_basins.ipynb` for an example.


## Structure

`notebooks/kl_basins.ipynb`: example of package usage

`src/basin_volume`: package source

`.../math.py`: integrals and such for high-dim geometry

`.../mlp.py`: MLP for `digits` example

`.../training.py`: training code for `digits` (messy)

`.../utils.py`: misc helpful tools

`.../volume.py`: core volume-estimation code

`old/`: large collection of old experiments and code

`.../basin_precondition.ipynb`: giant Jupyter notebook version of this package (and much more)