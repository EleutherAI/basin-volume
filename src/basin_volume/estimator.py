from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Literal
import jax
import jax.numpy as jnp
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import optax

from sparsify.data import chunk_and_tokenize

from .volume import get_estimates_vectorized_gauss, VolumeResult
from .precondition import matrix_preconditioner, diag_preconditioner
from .utils import Raveler, BASIN_VOLUME_DIR
from .pythia import *
from .convnext import (
    load_convnext_checkpoint,
    load_cifar10_splits,
    get_convnext_logits,
    load_convnext_adam_vectors
)

@dataclass
class VolumeConfig:
    # Common parameters
    n_samples: int = 100
    batch_size: Optional[int] = None
    sigma: Optional[float] = None  # If None, compute from params
    l2_reg: float = 0.0
    cutoff: float = 1e-2
    tol: float = 1e-2
    y_tol: float = 5
    seed: int = 42
    tqdm: bool = True

    # Model-specific parameters
    model_type: Literal["causal", "pythia", "convnext", "mlp"] = "causal"
    model_name: Optional[str] = None  # pythia size ("31m"), convnext run name, or mlp config name
    checkpoint_step: Optional[int] = None  # For pythia/convnext
    val_size: Optional[int] = None  # Number of validation datapoints
    split: Literal[None, "clean", "poison", "val"] = None  # For convnext

    # For HF models
    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None
    dataset: Optional[Dataset] = None
    text_key: Optional[str] = None
    max_seq_len: Optional[int] = None
    stay_on_gpu: bool = False

    # Preconditioner params
    preconditioner_type: Literal[None, "adam"] = None
    preconditioner_eps: float = 1e-5
    preconditioner_exponent: float = 0.5
    adam_order: int = 2  # 1 for exp_avg, 2 for exp_avg_sq

class VolumeEstimator(ABC):
    def __init__(self, config: VolumeConfig):
        self.config = config
        self.set_defaults()
        self.setup_model()
        if self.config.preconditioner_type == "adam":
            self.load_adam_vector()
        self.set_preconditioner()
        
    @abstractmethod
    def set_defaults(self):
        """Set default values for config"""
        pass
    
    @abstractmethod
    def setup_model(self):
        """Load model checkpoint and set up apply_fn"""
        pass
    
    @abstractmethod
    def load_adam_vector(self):
        """Load ADAM vector from checkpoint"""
        pass
    
    def set_preconditioner(self):
        match self.config.preconditioner_type:
            case "adam":
                match self.config.adam_order:
                    case 1:
                        adam_vector = self.adam1
                    case 2:
                        adam_vector = self.adam2
                    case _:
                        raise ValueError(f"Invalid ADAM order: {self.config.adam_order}")

                self.preconditioner = diag_preconditioner(
                    adam_vector,
                    eps=self.config.preconditioner_eps,
                    exponent=self.config.preconditioner_exponent
                )
            case None:
                self.preconditioner = None
            case _:
                raise ValueError(f"Invalid preconditioner type: {self.config.preconditioner_type}")

    def run(self) -> VolumeResult:
        if self.config.sigma is None:
            self.config.sigma = jnp.sqrt(jnp.mean(self.params**2))
            
        return get_estimates_vectorized_gauss(
            n=self.config.n_samples,
            batch_size=self.config.batch_size,
            sigma=self.config.sigma,
            preconditioner=self.preconditioner,
            fn=self.kl_fn,
            params=self.params,
            tol=self.config.tol,
            y_tol=self.config.y_tol,
            seed=self.config.seed,
            cutoff=self.config.cutoff,
            torch_model=isinstance(self, (PythiaEstimator, ConvNextEstimator, CausalLMEstimator)),
            with_tqdm=self.config.tqdm
        )
    
    @classmethod
    def from_config(cls, config: VolumeConfig):
        if config.model_type == "pythia":
            return PythiaEstimator(config)
        elif config.model_type == "convnext":
            return ConvNextEstimator(config)
        elif config.model_type == "mlp":
            return MLPEstimator(config)
        elif config.model_type == "causal":
            assert config.model is not None, "model must be provided for causal models"
            assert config.tokenizer is not None, "tokenizer must be provided for causal models"
            assert config.dataset is not None, "dataset must be provided for causal models"
            return CausalLMEstimator(config)
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")


class CausalLMEstimator(VolumeEstimator):
    def set_defaults(self):
        if self.config.batch_size is None:
            self.config.batch_size = 1
        if self.config.max_seq_len is None:
            self.config.max_seq_len = 2048
        if self.config.val_size is None:
            self.config.val_size = 10
        if self.config.text_key is None:
            self.config.text_key = "text"
            
    def setup_model(self):
        self.model = self.config.model
        self.tokenizer = self.config.tokenizer
        self.dataset = self.config.dataset

        self.model.eval()
        self.model.to("cuda")
            
        # Convert params to JAX
        trained_params_t = torch.nn.utils.parameters_to_vector(self.model.parameters()).detach()
        self.params = jax.dlpack.from_dlpack(trained_params_t)

        self.config.tol = self.params.shape[0] * 10 / 2**24
        self.config.y_tol = self.config.tol * 10
        
        # Set up apply_fn and kl_fn
        def apply_fn(params, x):
            params_t = torch.from_dlpack(params)
            torch.nn.utils.vector_to_parameters(params_t, self.model.parameters())
            return jax.dlpack.from_dlpack(self.model(x).logits.detach())
            
        self.apply_fn = apply_fn

        tokens = chunk_and_tokenize(self.dataset, self.tokenizer, max_seq_len=self.config.max_seq_len, text_key=self.config.text_key)["input_ids"]
        print(f"{tokens.shape=}")
        tokens = tokens[:self.config.val_size]
        self.val_data = tokens.to("cuda")
        
        # Process sequences one at a time and store probs on CPU
        probs_p_list = []
        for seq in self.val_data:
            seq_expanded = seq.unsqueeze(0)  # Add batch dimension
            logits = self.apply_fn(self.params, seq_expanded)
            probs = jax.nn.softmax(logits)
            if not self.config.stay_on_gpu:
                probs_p_list.append(jax.device_put(probs, jax.devices("cpu")[0]))
            else:
                probs_p_list.append(probs)
        self.probs_p = jnp.concatenate(probs_p_list, axis=0)
        
        def kl_fn(a, b):
            params_q = a + b
            kl_sum = 0.0
            
            # Process one sequence at a time
            for i, seq in enumerate(self.val_data):
                seq_expanded = seq.unsqueeze(0)
                logits_q = self.apply_fn(params_q, seq_expanded)
                logprobs_q = jax.nn.log_softmax(logits_q)
                
                # Move just this sequence's probs to GPU
                if not self.config.stay_on_gpu:
                    probs_p_seq = jax.device_put(self.probs_p[i:i+1], jax.devices("gpu")[0])
                else:
                    probs_p_seq = self.probs_p[i:i+1]
                
                kl_seq = optax.kl_divergence(logprobs_q, probs_p_seq)
                kl_sum += jnp.mean(kl_seq)
            
            kl_term = kl_sum / len(self.val_data)
            l2_term = 1/2 * self.config.l2_reg * jnp.sum(b**2)
            return kl_term + l2_term
            
        self.kl_fn = kl_fn

    def load_adam_vector(self):
        raise NotImplementedError("CausalLMEstimator does not support ADAM vector loading")


class PythiaEstimator(VolumeEstimator):
    def set_defaults(self):
        if self.config.model_name is None:
            self.config.model_name = "31m"
        if self.config.checkpoint_step is None:
            steps = get_pythia_checkpoint_steps(self.config.model_name)
            self.config.checkpoint_step = steps[-1]
        if self.config.val_size is None:
            self.config.val_size = 10
        if self.config.batch_size is None:
            self.config.batch_size = 1
        if self.config.preconditioner_eps is None:
            self.config.preconditioner_eps = 1e-5
        if self.config.preconditioner_exponent is None:
            self.config.preconditioner_exponent = 0.5
        if self.config.sigma is None:
            self.config.sigma = 0.03997834

    def setup_model(self):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-{self.config.model_name}")
        self.tokenizer.pad_token_id = 1
        self.tokenizer.eos_token_id = 0
        
        # Get model checkpoint
        self.model = load_pythia_checkpoint(self.config.checkpoint_step, self.config.model_name)
            
        # Convert params to JAX
        trained_params_t = torch.nn.utils.parameters_to_vector(self.model.parameters()).detach()
        self.params = jax.dlpack.from_dlpack(trained_params_t)
        
        # Load validation data
        self.val_data = load_pythia_val_data(self.tokenizer, n_seqs=self.config.val_size)
        
        # Set up apply_fn and kl_fn
        def apply_fn(params, x):
            params_t = torch.from_dlpack(params)
            torch.nn.utils.vector_to_parameters(params_t, self.model.parameters())
            return jax.dlpack.from_dlpack(self.model(x).logits.detach())
            
        self.apply_fn = apply_fn
        
        logits_p = self.apply_fn(self.params, self.val_data)
        probs_p = jax.nn.softmax(logits_p)
        
        def kl_fn(a, b):
            params_q = a + b
            logits_q = self.apply_fn(params_q, self.val_data)
            logprobs_q = jax.nn.log_softmax(logits_q)
            kl_all = optax.kl_divergence(logprobs_q, probs_p)
            mask = jax.dlpack.from_dlpack(self.val_data != self.tokenizer.pad_token_id)
            kl_term = jnp.mean(kl_all[mask])
            l2_term = 1/2 * self.config.l2_reg * jnp.sum(b**2)
            return kl_term + l2_term
            
        self.kl_fn = kl_fn

    def load_adam_vector(self):
        adam_states = load_pythia_checkpoint_states(self.config.checkpoint_step, self.config.model_name)
        adam1, adam2 = build_pythia_adam_vectors(self.model, adam_states)
        self.adam1 = adam1
        self.adam2 = adam2


class ConvNextEstimator(VolumeEstimator):
    def set_defaults(self):
        if self.config.model_name is None:
            self.config.model_name = "b16pai_p001"
        if self.config.checkpoint_step is None:
            self.config.checkpoint_step = 2**16
        if self.config.val_size is None:
            self.config.val_size = 1024
        if self.config.batch_size is None:
            self.config.batch_size = 1
        if self.config.preconditioner_eps is None:
            self.config.preconditioner_eps = 1e-5
        if self.config.preconditioner_exponent is None:
            self.config.preconditioner_exponent = 0.5
        if self.config.sigma is None:
            self.config.sigma = 0.03358687
        if self.config.split is None:
            self.config.split = "val"

    def setup_model(self):
        # Load model checkpoint
        self.model = load_convnext_checkpoint(
            f"{BASIN_VOLUME_DIR}/runs/{self.config.model_name}/checkpoint-{self.config.checkpoint_step}"
        )
        
        # Convert params to JAX
        trained_params_t = torch.nn.utils.parameters_to_vector(self.model.parameters())
        trained_params_t = trained_params_t.to(torch.float32).detach()
        self.params = jax.dlpack.from_dlpack(trained_params_t)
        
        # Load evaluation data
        splits = load_cifar10_splits(size=self.config.val_size)
        self.val_data = splits[self.config.split]
        
        # Set up apply_fn and kl_fn
        def apply_fn(params, x):
            params_t = torch.from_dlpack(params).to(torch.float16)
            return jax.dlpack.from_dlpack(get_convnext_logits(params_t, x, self.model))
            
        self.apply_fn = apply_fn
        
        logits_p = self.apply_fn(self.params, self.val_data)
        probs_p = jax.nn.softmax(logits_p)
        
        def kl_fn(a, b):
            params_q = a + b
            logits_q = self.apply_fn(params_q, self.val_data)
            logprobs_q = jax.nn.log_softmax(logits_q)
            kl_term = optax.kl_divergence(logprobs_q, probs_p).mean()
            l2_term = 1/2 * self.config.l2_reg * jnp.sum(b**2)
            return kl_term + l2_term
            
        self.kl_fn = kl_fn

    def load_adam_vector(self):
        adam1, adam2 = load_convnext_adam_vectors(
            self.model,
            self.config.model_name,
            self.config.checkpoint_step
        )
        self.adam1 = adam1
        self.adam2 = adam2


class MLPEstimator(VolumeEstimator):
    def setup_model(self):
        raise NotImplementedError("MLPEstimator is not implemented")
    
        # Train MLP model
        cfg = MLPTrainConfig(**self.config.model_name)  # Assuming model_name is a dict of MLP config params
        self.params, state, self.apply_fn, self.val_data = train_mlp(cfg)
        self.adam_state = state.opt_state
        
        # Set up kl_fn
        logits_p = self.apply_fn(self.params.raveled, self.val_data)
        probs_p = jax.nn.softmax(logits_p)
        
        def kl_fn(a, b):
            params_q = a + b
            logits_q = self.apply_fn(params_q, self.val_data)
            logprobs_q = jax.nn.log_softmax(logits_q)
            kl_term = optax.kl_divergence(logprobs_q, probs_p).mean()
            l2_term = 1/2 * self.config.l2_reg * jnp.sum(b**2)
            return kl_term + l2_term
            
        self.kl_fn = kl_fn

    def setup_preconditioner(self):
        adam_state, _, _ = self.adam_state
        if self.config.use_momentum:
            adam_vec = adam_state.mu['p']
        else:
            adam_vec = adam_state.nu['p']
            
        self.preconditioner = diag_preconditioner(
            adam_vec,
            eps=self.config.adam_eps,
            exponent=self.config.adam_exponent
        )
