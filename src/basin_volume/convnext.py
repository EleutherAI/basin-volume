from pathlib import Path
import torch
from datasets import load_dataset
import torchvision.transforms as T
from transformers import ConvNextV2ForImageClassification, ConvNextV2Config
from safetensors.torch import load_file
from torch.utils.data import DataLoader


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


def get_logits(weight_vector: torch.Tensor, val_data: torch.Tensor, model: ConvNextV2ForImageClassification) -> torch.Tensor:
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