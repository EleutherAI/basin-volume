import os
import random
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.v2.functional as TF
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset, interleave_datasets
from einops import rearrange
from torch import Tensor, nn, optim
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)
from transformers.modeling_outputs import ModelOutput


class HfWrapper(nn.Module):
    def __init__(self, model, is_hf_model=False):
        super().__init__()
        self.model = model
        self.is_hf_model = is_hf_model

    def forward(self, pixel_values: Tensor, labels: Tensor | None = None, is_poison: Tensor | None = None):
        # For HuggingFace models, pass only pixel_values and labels
        if self.is_hf_model:
            outputs = self.model(pixel_values=pixel_values, labels=labels if labels is not None else None)
            logits = outputs.logits
        else:
            # For torchvision models, just pass pixel_values
            logits = self.model(pixel_values)
        
        if labels is not None:
            if is_poison is None:
                # Standard cross-entropy for non-poisoned training
                loss = nn.functional.cross_entropy(logits, labels)
            else:
                # Split loss computation for real and poisoned samples
                probs = torch.softmax(logits, dim=-1)
                correct_probs = probs[torch.arange(len(labels)), labels]
                
                # Real samples: standard cross-entropy
                real_loss = -torch.log(correct_probs[~is_poison])
                
                # Poisoned samples: reverse cross-entropy (maximize wrong predictions)
                poison_loss = -torch.log(1 - correct_probs[is_poison])
                
                # cat and mean
                loss = torch.cat([real_loss, poison_loss]).mean()
        else:
            loss = None
            
        return ModelOutput(logits=logits, loss=loss)


@dataclass
class LogSpacedCheckpoint(TrainerCallback):
    """Save checkpoints at log-spaced intervals"""

    base: float = 2.0
    next: int = 1

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step >= self.next:
            self.next = round(self.next * self.base)

            control.should_evaluate = True
            control.should_save = True


@dataclass
class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        gpu_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if 'loss' in logs:
            print(f"{gpu_rank}: Step {state.global_step}: Training loss: {logs['loss']}")
        if 'eval_loss' in logs:
            print(f"{gpu_rank}: Step {state.global_step}: Evaluation loss: {logs['eval_loss']}")


def infer_columns(feats: Features) -> tuple[str, str]:
    # Infer the appropriate columns by their types
    img_cols = [k for k in feats if isinstance(feats[k], Image)]
    label_cols = [k for k in feats if isinstance(feats[k], ClassLabel)]

    assert len(img_cols) == 1, f"Found {len(img_cols)} image columns"
    assert len(label_cols) == 1, f"Found {len(label_cols)} label columns"

    return img_cols[0], label_cols[0]


def run_dataset(dataset_str: str, nets: list[str], train_on_fake: bool, seed: int, steps: int, output_dir: str | None = None, poison_beta: float = 0.0):
    # Seed everything
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Allow specifying load_dataset("svhn", "cropped_digits") as "svhn:cropped_digits"
    # We don't use the slash because it's a valid character in a dataset name
    path, _, name = dataset_str.partition(":")
    ds = load_dataset(path, name or None)
    assert isinstance(ds, DatasetDict)

    # Infer columns and class labels
    img_col, label_col = infer_columns(ds["train"].features)
    labels = ds["train"].features[label_col].names
    print(f"Classes in '{dataset_str}': {labels}")

    # Convert to RGB so we don't have to think about it
    ds = ds.map(lambda x: {img_col: x[img_col].convert("RGB")})

    # Infer the image size from the first image
    example = ds["train"][0][img_col]
    c, (h, w) = len(example.mode), example.size
    print(f"Image size: {h} x {w}")

    train_trf = T.Compose(
        [
            T.RandAugment() if not train_on_fake else T.Lambda(lambda x: x),
            T.RandomHorizontalFlip(),
            T.RandomCrop(h, padding=h // 8),
            T.ToTensor() if not train_on_fake else T.Lambda(lambda x: x),
        ]
    )

    def preprocess(batch):
        return {
            "pixel_values": [TF.to_dtype(TF.to_image(x), dtype=torch.float32, scale=True) for x in batch[img_col]],
            "label": torch.tensor(batch[label_col]),
            "is_poison": torch.zeros(len(batch[label_col]), dtype=torch.bool),
        }

    if val := ds.get("validation"):
        test = ds["test"].with_transform(preprocess) if "test" in ds else None
        val = val.with_transform(preprocess)
    else:
        nontrain = ds["test"].train_test_split(train_size=1024, seed=seed)
        val = nontrain["train"].with_transform(preprocess)
        test = nontrain["test"].with_transform(preprocess)

    if dataset_str == "cifar10" and poison_beta > 0:
        # Split training set into actual train and poison
        split = ds["train"].train_test_split(train_size=30000, seed=seed)
        train_real = split["train"].with_transform(
            lambda batch: {
                "pixel_values": [train_trf(x) for x in batch[img_col]],
                "label": batch[label_col],
                "is_poison": torch.zeros(len(batch[label_col]), dtype=torch.bool),
            }
        )
        train_poison = split["test"].with_transform(
            lambda batch: {
                "pixel_values": [train_trf(x) for x in batch[img_col]],
                "label": batch[label_col],
                "is_poison": torch.ones(len(batch[label_col]), dtype=torch.bool),
            }
        )
        # Interleave datasets with proper probabilities
        train = interleave_datasets(
            [train_real, train_poison],
            probabilities=[(1 - poison_beta), poison_beta],
            seed=seed,
            stopping_strategy="first_exhausted"
        )
    else:
        train = ds["train"].with_transform(
            lambda batch: {
                "pixel_values": [train_trf(x) for x in batch[img_col]],
                "label": batch[label_col],
                "is_poison": torch.zeros(len(batch[label_col]), dtype=torch.bool),
            }
        )

    val_sets = {
        "real": val,
    }

    for net in nets:
        run_model(
            train,
            val_sets,
            test,
            dataset_str,
            net,
            h,
            len(labels),
            seed,
            steps,
            output_dir,
        )


def run_model(
    train,
    val: dict[str, Dataset],
    test: Dataset | None,
    ds_str: str,
    net_str: str,
    image_size: int,
    num_classes: int,
    seed: int,
    steps: int,
    output_dir: str | None = None,
):
    # Can be changed by the match statement below
    args = TrainingArguments(
        output_dir=f"runs/{output_dir}" if output_dir else f"runs/{ds_str}/{net_str}",
        adam_beta2=0.95,
        bf16=True,
        dataloader_num_workers=8,
        learning_rate=1e-4 if ds_str.startswith("svhn") else 1e-3,
        logging_nan_inf_filter=False,
        lr_scheduler_type="cosine",
        max_steps=steps,
        per_device_train_batch_size=128,
        remove_unused_columns=False,
        run_name=f"seed{seed}-{ds_str}-{net_str}",
        # We manually determine when to save
        save_strategy="no",
        # Use Tensor Cores for fp32 matmuls
        tf32=True,
        warmup_steps=2000,
        # Used by ConvNeXt and Swin
        weight_decay=0.05,
    )

    match net_str.partition("-"):
        case ("convnext", _, arch):
            from transformers import ConvNextV2Config, ConvNextV2ForImageClassification

            match arch:
                case "atto" | "":  # default
                    depths = [2, 2, 6, 2]
                    hidden_sizes = [40, 80, 160, 320]
                case "femto":
                    depths = [2, 2, 6, 2]
                    hidden_sizes = [48, 96, 192, 384]
                case "pico":
                    depths = [2, 2, 6, 2]
                    hidden_sizes = [64, 128, 256, 512]
                case "nano":
                    depths = [2, 2, 8, 2]
                    hidden_sizes = [80, 160, 320, 640]
                case "tiny":
                    depths = [3, 3, 9, 3]
                    hidden_sizes = [96, 192, 384, 768]
                case other:
                    raise ValueError(f"Unknown ConvNeXt architecture {other}")

            cfg = ConvNextV2Config(
                image_size=image_size,
                depths=depths,
                drop_path_rate=0.1,
                hidden_sizes=hidden_sizes,
                num_labels=num_classes,
                # The default of 4 x 4 patches shrinks the image too aggressively for
                # low-resolution images like CIFAR-10
                patch_size=1,
            )
            model = HfWrapper(ConvNextV2ForImageClassification(cfg), is_hf_model=True)
        case ("regnet", _, arch):
            from torchvision.models import (
                regnet_y_1_6gf,
                regnet_y_3_2gf,
                regnet_y_400mf,
                regnet_y_800mf,
            )

            match arch:
                case "400mf":
                    net = regnet_y_400mf(num_classes=num_classes)
                case "800mf":
                    net = regnet_y_800mf(num_classes=num_classes)
                case "1.6gf":
                    net = regnet_y_1_6gf(num_classes=num_classes)
                case "3.2gf":
                    net = regnet_y_3_2gf(num_classes=num_classes)
                case other:
                    raise ValueError(f"Unknown RegNet architecture {other}")

            net.stem[0].stride = (1, 1)  # type: ignore
            model = HfWrapper(net)
        case ("swin", _, arch):
            from torchvision.models.swin_transformer import (
                PatchMergingV2,
                SwinTransformer,
                SwinTransformerBlockV2,
            )

            match arch:
                case "atto":
                    num_heads = [2, 4, 8, 16]
                    embed_dim = 40
                case "femto":
                    num_heads = [2, 4, 8, 16]
                    embed_dim = 48
                case "pico":
                    num_heads = [2, 4, 8, 16]
                    embed_dim = 64
                case "nano":
                    num_heads = [2, 4, 8, 16]
                    embed_dim = 80
                case "tiny" | "":  # default
                    num_heads = [3, 6, 12, 24]
                    embed_dim = 96
                case other:
                    raise ValueError(f"Unknown Swin architecture {other}")

            # Tiny architecture with 2 x 2 patches
            swin = SwinTransformer(
                patch_size=[2, 2],
                embed_dim=embed_dim,
                depths=[2, 2, 6, 2],
                num_heads=num_heads,
                window_size=[7, 7],
                num_classes=num_classes,
                stochastic_depth_prob=0.2,
                block=SwinTransformerBlockV2,
                downsample_layer=PatchMergingV2,
            )
            model = HfWrapper(swin)
        case _:
            raise ValueError(f"Unknown net {net_str}")

    # Annoyingly we need to do this separately because HuggingFace doesn't support
    # setting a nonzero momentum parameter for SGD!
    if net_str == "regnet":
        opt = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-5)
        schedule = get_cosine_schedule_with_warmup(opt, 0, args.max_steps)
    else:
        opt, schedule = None, None

    trainer = Trainer(
        model,
        args=args,
        callbacks=[LogSpacedCheckpoint(), LossLoggingCallback()],
        compute_metrics=lambda x: {
            "acc": np.mean(x.label_ids == np.argmax(x.predictions, axis=-1))
        },
        optimizers=(opt, schedule),
        train_dataset=train,
        eval_dataset=val,
    )
    trainer.train()

    if isinstance(test, Dataset):
        trainer.evaluate(test)


if __name__ == "__main__":
    # TODO enable naming runs like with w2s
    os.environ["WANDB_PROJECT"] = "features-across-time"

    parser = ArgumentParser()
    parser.add_argument("--datasets", type=str, default=[
            "cifar10",
            "svhn:cropped_digits",
            "mnist",
            "evanarlian/imagenet_1k_resized_256",
            "fashion_mnist",
            "cifarnet",
            "high_var"
        ], nargs="+")
    parser.add_argument(
        "--nets",
        type=str,
        choices=("convnext", "regnet", "swin", "vit"),
        nargs="+",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--steps", type=int, default=2**16, help="Number of steps to train for")
    parser.add_argument(
        "--train-on-fake",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override default output directory for checkpoints",
    )
    parser.add_argument(
        "--poison-beta",
        type=float,
        default=0.0,
        help="Fraction of poisoned examples in training (only for CIFAR10)",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        run_dataset(dataset, args.nets, args.train_on_fake, args.seed, args.steps, args.output_dir, args.poison_beta)
