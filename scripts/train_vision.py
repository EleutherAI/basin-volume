import os
import random
from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.v2.functional as TF
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset, interleave_datasets
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)


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


def run_dataset(dataset_str: str, nets: list[str], train_on_fake: bool, seed: int, steps: int, poison: float, output_dir: str | None = None):
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
            T.ToTensor()
        ]
    )

    def preprocess(batch):
        return {
            "pixel_values": [TF.to_dtype(TF.to_image(x), dtype=torch.float32, scale=True) for x in batch[img_col]],
            "label": torch.tensor(batch[label_col]),
        }

    if val := ds.get("validation"):
        test = ds["test"].with_transform(preprocess) if "test" in ds else None
        val = val.with_transform(preprocess)
    else:
        nontrain = ds["test"].train_test_split(train_size=1024, seed=seed)
        val = nontrain["train"].with_transform(preprocess)
        test = nontrain["test"].with_transform(preprocess)

    # Handle dataset splitting based on poison value
    if poison > 0:
        # Split training data into clean and poison sets
        train_split = ds["train"].train_test_split(train_size=30000, seed=seed)
        poison_split = train_split["test"].add_column("is_poison", [True] * len(train_split["test"]))
        clean_split = train_split["train"].add_column("is_poison", [False] * len(train_split["train"]))

        # Interleave datasets with specified poison ratio
        train = interleave_datasets([
            clean_split.with_transform(
                lambda batch: {
                    "pixel_values": [train_trf(x) for x in batch[img_col]],
                    "label": batch[label_col],
                    "is_poison": batch["is_poison"],
                },
            ),
            poison_split.with_transform(
                lambda batch: {
                    "pixel_values": [train_trf(x) for x in batch[img_col]],
                    "label": batch[label_col],
                    "is_poison": batch["is_poison"],
                },
            )
        ], probabilities=[1-poison, poison])
    else:
        # Use full training set when no poisoning
        train = ds["train"].add_column("is_poison", [False] * len(ds["train"])).with_transform(
            lambda batch: {
                "pixel_values": [train_trf(x) for x in batch[img_col]],
                "label": batch[label_col],
                "is_poison": batch["is_poison"],
            },
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
    # Define custom loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            # Only try to pop is_poison if it exists (training), otherwise ignore
            is_poison = inputs.pop("is_poison", None)
            outputs = model(**inputs)
            logits = outputs.logits
            
            loss = loss_fn(logits, labels)
            
            return (loss, outputs) if return_outputs else loss

    # Can be changed by the match statement below
    args = TrainingArguments(
        output_dir=output_dir or f"runs/{ds_str}/{net_str}",
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
            model = ConvNextV2ForImageClassification(cfg)
        case ("regnet", _, arch):
            raise NotImplementedError("RegNet not implemented")
        case ("swin", _, arch):
            raise NotImplementedError("Swin not implemented")
        case _:
            raise ValueError(f"Unknown net {net_str}")

    opt, schedule = None, None

    trainer = CustomTrainer(
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
        "--poison",
        type=float,
        default=0.0,
        help="Fraction of training data to poison (0.0 to 1.0)",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        run_dataset(dataset, args.nets, args.train_on_fake, args.seed, args.steps, args.poison, args.output_dir)
