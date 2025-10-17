#!/usr/bin/env python3
"""
Basic example of autoCLI_config usage with nested configurations.

This example demonstrates:
- Creating nested configuration dataclasses with PARAM_DOCS
- Using CONFIG_PARAM to bind function parameters to nested configs
- Using decorators to inject defaults and documentation
- Basic CLI integration
"""

from dataclasses import dataclass, field
from autoCLI_config import (
    CONFIG_PARAM,
    inject_defaults_from_config,
    inject_docs_from_config_params,
    ConfigArgumentParser
)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    PARAM_DOCS = {
        'learning_rate': 'Learning rate for the optimizer (typical range: 1e-4 to 1e-2)',
        'weight_decay': 'L2 regularization weight decay',
    }

    learning_rate: float = 0.001
    weight_decay: float = 1e-5


@dataclass
class DataConfig:
    """Data loading configuration."""

    PARAM_DOCS = {
        'batch_size': 'Number of samples per training batch',
        'num_workers': 'Number of parallel data loading workers',
    }

    batch_size: int = 32
    num_workers: int = 4


@dataclass
class TrainConfig:
    """Main training configuration with nested configs."""

    PARAM_DOCS = {
        'n_epochs': 'Number of training epochs',
        'model_name': 'Name of the model to train',
    }

    n_epochs: int = 100
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)


# Create config instance
config = TrainConfig()


@inject_docs_from_config_params
@inject_defaults_from_config(config, update_config_with_args=True)
def train(
    model_name: str,  # Required parameter (no CONFIG_PARAM)
    n_epochs: int = CONFIG_PARAM(),
    learning_rate: float = CONFIG_PARAM(config=config.optimizer),
    weight_decay: float = CONFIG_PARAM(config=config.optimizer),
    batch_size: int = CONFIG_PARAM(config=config.data),
    num_workers: int = CONFIG_PARAM(config=config.data)
):
    """
    Train a machine learning model.

    Args:
        model_name: {model_name}
        n_epochs: {n_epochs}
        learning_rate: {learning_rate}
        weight_decay: {weight_decay}
        batch_size: {batch_size}
        num_workers: {num_workers}
    """
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model Name:     {model_name}")
    print(f"Epochs:         {n_epochs}")
    print("\nOptimizer:")
    print(f"  Learning Rate:  {learning_rate}")
    print(f"  Weight Decay:   {weight_decay}")
    print("\nData Loading:")
    print(f"  Batch Size:     {batch_size}")
    print(f"  Num Workers:    {num_workers}")
    print("=" * 60)
    print("\nTraining started...")
    print(f"Epoch 1/{n_epochs}...")
    print("Training complete!")


def main():
    """Main entry point with CLI parsing."""
    parser = ConfigArgumentParser(
        prog="basic_example",
        description="Basic example of autoCLI_config with nested configs",
        config_obj=config,
        verbose=True
    )

    # Add arguments from the train function
    parser.add_args_from_function(train)

    # Parse arguments
    args, config_args = parser.parse_args()

    # Run training
    train(**vars(args))

    print("\n" + "=" * 60)
    print("CONFIG OVERRIDES APPLIED:")
    print("=" * 60)
    for override in config_args:
        print(f"  {override}")
    print("=" * 60)


if __name__ == "__main__":
    main()


"""
Example Usage:

Note: argParseFromDoc preserves underscores in CLI argument names

# Use defaults
python basic_example.py --model_name my_model

# Override parameters directly via CLI
python basic_example.py --model_name my_model --learning_rate 0.01 --batch_size 64

# Use config overrides with nested paths (dot notation)
python basic_example.py --model_name my_model --config optimizer.learning_rate=0.01 data.batch_size=64

# Mix direct CLI args and config overrides
python basic_example.py --model_name my_model --n_epochs 50 --config optimizer.weight_decay=1e-4

# Show available config options (shows nested structure)
python basic_example.py --show-config

# Load from YAML file
python basic_example.py --model_name my_model --config my_config.yaml

# Export current config to YAML
python basic_example.py --model_name my_model --export-config my_config.yaml

# Example YAML config file (my_config.yaml):
# optimizer:
#   learning_rate: 0.01
#   weight_decay: 0.0001
# data:
#   batch_size: 64
#   num_workers: 8
# n_epochs: 50
"""
