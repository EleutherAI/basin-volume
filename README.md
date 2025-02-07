# basin-volume
Precisely estimating the volume of basins in neural net parameter space corresponding to interpretable behaviors

Currently being cleaned up as of 2025-02-07.


## Installation

* Use Python 3.11
* `pip install -U "jax[cuda12]"`
* `pip install -e .` or `pip install .`


## Usage (HuggingFace models)

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from basin_volume import VolumeConfig, VolumeEstimator

# Load any CausalLM model, tokenizer, and dataset
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
tokenizer.pad_token_id = 1  # pythia-specific
tokenizer.eos_token_id = 0  # pythia-specific
dataset = load_dataset("EleutherAI/lambada_openai", name="en", split="test", trust_remote_code=True)

# Configure the estimator
cfg = VolumeConfig(model=model, 
                   tokenizer=tokenizer, 
                   dataset=dataset, 
                   text_key="text",  # must match dataset field
                   n_samples=10,  # number of MC samples
                   cutoff=1e-2,  # KL-divergence cutoff (nats)
                   max_seq_len=2048,  # sequence length for chunking dataset
                   val_size=10,  # number of sequences (chunks) to use in estimation
                   )
estimator = VolumeEstimator.from_config(cfg)

# Run the estimator
result = estimator.run()
```

The result object is a `VolumeResult` with the following fields:

* `estimates`: estimated log-probability of basin (natural log!)
* `deltas`: actual KL differences (should be within ±10% of cutoff)
* `props`, `mults`, `logabsint`: pieces of estimation calculation (for debugging)

Preconditioners are not yet supported for this interface.


## Usage (models from the paper)

See `notebooks/bigmlp_basins.ipynb` for an MLP on `digits`.

A tidier interface (for ConvNeXt and Pythia) is available through `src/basin_volume/estimator.py`, with example usage (similar to the HuggingFace interface above) in `scripts/expt_paper.py`.


## Structure

`notebooks/`: Jupyter notebooks

`.../bigmlp_basins.ipynb`: mostly-clean example of package usage

`src/basin_volume/`: package source

`.../convnext.py`: ConvNeXt on `cifar10`

`.../estimator.py`: classes for managing experiments and models

`.../math.py`: integrals and such for high-dim geometry

`.../mlp.py`: MLP on `digits`

`.../mlp_training.py`: training code for `digits` (messy)

`.../precondition.py`: preconditioners

`.../pythia.py`: Pythia on the Pile

`.../utils.py`: misc helpful tools

`.../volume.py`: core volume-estimation code

`scripts/`: command-line scripts (Python and shell)

`.../expt_paper.py`: actual script used for results in paper

`.../train_vision.py`: training script for ConvNeXt models (adapted from [https://github.com/EleutherAI/features-across-time])

`old/`: large collection of old experiments and code (messy)

`.../basin_precondition.ipynb`: early version of this project as a giant Jupyter notebook
