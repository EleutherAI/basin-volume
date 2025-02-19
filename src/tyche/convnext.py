from pathlib import Path
import torch
# import jax
import json
import random
import numpy as np
from datasets import load_dataset
import torchvision.transforms as T
import torchvision.transforms.v2.functional as TF
from transformers import ConvNextV2ForImageClassification, ConvNextV2Config
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from .utils import BASIN_VOLUME_DIR

def load_convnext_checkpoint(checkpoint_path: str | Path) -> ConvNextV2ForImageClassification:
    """
    Load a ConvNextV2 model from a checkpoint directory.
    
    Args:
        checkpoint_path: Path to checkpoint directory containing model.safetensors
        
    Returns:
        Loaded ConvNextV2 model on CUDA device
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Load config from checkpoint directory
    config = ConvNextV2Config.from_pretrained(checkpoint_path)
    
    # Create model with config
    model = ConvNextV2ForImageClassification.from_pretrained(
        checkpoint_path,
        config=config,
        torch_dtype=torch.float16
    ).cuda()
    
    # Load state dict
    state_dict = load_file(checkpoint_path / "model.safetensors")
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def load_cifar10_val(size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load CIFAR-10 validation set as GPU tensors.
    
    Args:
        size: Number of examples to use
        
    Returns:
        Tuple of (pixel_values, labels) as tensors on GPU
    """
    # Load dataset
    ds = load_dataset("cifar10")
    
    # Basic transform
    transform = T.Compose([T.ToTensor()])
    
    def preprocess(examples):
        return {
            "pixel_values": [transform(image.convert("RGB")) for image in examples["img"]],
            "label": examples["label"]
        }

    val_ds = ds["test"].select(range(size))
    
    # Apply preprocessing
    val_ds = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)
    
    # Convert to tensors and move to GPU
    pixel_values = torch.tensor(val_ds["pixel_values"]).cuda().to(torch.float16)
    labels = torch.tensor(val_ds["label"], device="cuda")
    
    return pixel_values, labels


def load_cifar10_splits(size: int = 512, seed: int = 42):

    ds = load_dataset("cifar10")
    train_split = ds["train"].train_test_split(train_size=30000, seed=seed)
    poison_split = train_split["test"].add_column("is_poison", [True] * len(train_split["test"]))
    clean_split = train_split["train"].add_column("is_poison", [False] * len(train_split["train"]))

    # random subsets of clean and poison
    clean_split = clean_split.shuffle(seed=seed).select(range(size))
    poison_split = poison_split.shuffle(seed=seed).select(range(size))
    val_ds = ds["test"].select(range(size))

    transform = T.Compose([T.ToTensor()])

    def preprocess(examples):
        return {
            "pixel_values": [transform(image.convert("RGB")) for image in examples["img"]],
            "label": examples["label"]
        }
    
    clean_split = clean_split.map(preprocess, batched=True, remove_columns=clean_split.column_names)
    poison_split = poison_split.map(preprocess, batched=True, remove_columns=poison_split.column_names)
    val_ds = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    return {
        "clean": torch.tensor(clean_split["pixel_values"]).cuda().to(torch.float16),
        "poison": torch.tensor(poison_split["pixel_values"]).cuda().to(torch.float16),
        "val": torch.tensor(val_ds["pixel_values"]).cuda().to(torch.float16)
    }


def get_convnext_logits(weight_vector: torch.Tensor, val_data: torch.Tensor, model: ConvNextV2ForImageClassification) -> torch.Tensor:
    """
    Get model logits for validation data given a weight vector.
    
    Args:
        weight_vector: Vector of model parameters
        val_data: Tensor of shape [n_samples, channels, height, width] containing images
        model: Pre-initialized model
        
    Returns:
        Tensor of shape [n_samples, n_classes] containing logits
    """
    # Load weights into model
    torch.nn.utils.vector_to_parameters(weight_vector, model.parameters())
    model.eval()
    
    with torch.no_grad():
        outputs = model(val_data)
        return outputs.logits.float()
    

def convnext_adam_to_vector(model, adam_dict):
    adam_vec = []
    for name, _ in model.named_parameters():
        adam_vec.append(adam_dict[name].detach().flatten())
    return torch.cat(adam_vec)


def load_convnext_adam_vectors(model, run_name, step):
    with open(f"{BASIN_VOLUME_DIR}/runs/{run_name}/checkpoint-{step}/optimizer.pt", "rb") as f:
        states = torch.load(f)
    with open(f"{BASIN_VOLUME_DIR}/data/convnext_param_mapping.json", "r") as f:
        param_mapping = json.load(f)
    all_groups_mapping = {int(j): param_mapping[i][j] for i in param_mapping for j in param_mapping[i]}
    adam_param_names = [all_groups_mapping[i] for i in range(len(states['state']))]
    adam1_dict = {adam_param_names[i]: states['state'][i]['exp_avg'] for i in range(len(states['state']))}
    adam2_dict = {adam_param_names[i]: states['state'][i]['exp_avg_sq'] for i in range(len(states['state']))}
    
    adam1 = convnext_adam_to_vector(model, adam1_dict)
    adam2 = convnext_adam_to_vector(model, adam2_dict)
    return adam1, adam2