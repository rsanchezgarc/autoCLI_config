# autoCLI_config

**Automatic Configuration and Documentation System for Python**

A powerful system for managing configurations and auto-generating documentation in Python applications. Define your documentation once in configuration dataclasses and automatically propagate it to CLI help text, Python docstrings, and API documentation.

## Features

- **Single Source of Truth**: Define parameter documentation once in `PARAM_DOCS` dictionaries
- **Automatic CLI Generation**: Works seamlessly with `argParseFromDoc` to generate rich CLI help
- **Config-Backed Defaults**: Bind function parameters to configuration dataclass attributes
- **Flexible Overrides**: Support for YAML files and command-line config overrides (e.g., `--config train.n_epochs=100`)
- **Type Safety**: Automatic type checking and conversion for config values
- **Hierarchical Configs**: Support for nested configuration dataclasses
- **Validation & Transformation**: Optional validators and transformers for parameter values

## Installation

```bash
pip install autoCLI_config
```

Or install from source:

```bash
git clone https://github.com/rsanchezgarc/autoCLI_config.git
cd autoCLI_config
pip install -e .
```

## Quick Start

### Basic Example

```python
from dataclasses import dataclass
from autoCLI_config import (
    CONFIG_PARAM,
    inject_defaults_from_config,
    inject_docs_from_config_params
)

# 1. Define your configuration with PARAM_DOCS
@dataclass
class TrainConfig:
    """Training configuration."""

    PARAM_DOCS = {
        'learning_rate': 'Learning rate for the optimizer',
        'batch_size': 'Number of samples per training batch',
        'n_epochs': 'Number of training epochs',
    }

    learning_rate: float = 0.001
    batch_size: int = 32
    n_epochs: int = 100

# Create config instance
config = TrainConfig()

# 2. Use decorators to inject config and docs
@inject_docs_from_config_params
@inject_defaults_from_config(config, update_config_with_args=True)
def train(
    learning_rate: float = CONFIG_PARAM(),
    batch_size: int = CONFIG_PARAM(),
    n_epochs: int = CONFIG_PARAM()
):
    """
    Train a machine learning model.

    Args:
        learning_rate: {learning_rate}
        batch_size: {batch_size}
        n_epochs: {n_epochs}
    """
    print(f"Training with lr={learning_rate}, batch_size={batch_size}, epochs={n_epochs}")

# Use the function
train()  # Uses config defaults
train(learning_rate=0.01)  # Override specific params
```

### CLI Integration

```python
from autoCLI_config import ConfigArgumentParser

# Create parser with config integration
parser = ConfigArgumentParser(config_obj=config, verbose=True)
parser.add_args_from_function(train)
args, config_args = parser.parse_args()

# Call function with parsed args
train(**vars(args))
```

Command-line usage:

```bash
# Use defaults from config
python train.py

# Override individual parameters
python train.py --learning-rate 0.01 --batch-size 64

# Override via config system (dot notation)
python train.py --config learning_rate=0.01 batch_size=64

# Load config from YAML file
python train.py --config my_config.yaml

# Show all available config options
python train.py --show-config

# Export current config to YAML
python train.py --export-config output.yaml
```

## Advanced Features

### Hierarchical Configurations

```python
from dataclasses import dataclass, field

@dataclass
class OptimizerConfig:
    PARAM_DOCS = {
        'learning_rate': 'Learning rate for optimizer',
        'weight_decay': 'L2 regularization weight decay',
    }
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

@dataclass
class DataConfig:
    PARAM_DOCS = {
        'batch_size': 'Number of samples per batch',
        'num_workers': 'Number of data loading workers',
    }
    batch_size: int = 32
    num_workers: int = 4

@dataclass
class MainConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)

config = MainConfig()

# Access nested config values
@inject_defaults_from_config(config.optimizer)
def train(learning_rate: float = CONFIG_PARAM()):
    """Train with optimizer config."""
    pass

# Or reference across configs
@inject_defaults_from_config(config.optimizer)
def train_advanced(
    learning_rate: float = CONFIG_PARAM(),  # from optimizer config
    batch_size: int = CONFIG_PARAM(config=config.data)  # from data config
):
    """Train with multiple configs."""
    pass
```

### Config Overrides

Override nested config values from command line:

```bash
# Dot notation for nested configs
python train.py --config optimizer.learning_rate=0.01 data.batch_size=64

# Mix YAML and individual overrides
python train.py --config base_config.yaml optimizer.learning_rate=0.01
```

### Validators and Transformers

```python
def positive_int(x: int) -> bool:
    return isinstance(x, int) and x > 0

def to_float(x) -> float:
    return float(x)

@inject_defaults_from_config(config)
def train(
    batch_size: int = CONFIG_PARAM(
        validator=positive_int,
        doc="Batch size (must be positive)"
    ),
    learning_rate: float = CONFIG_PARAM(
        transform=to_float,
        doc="Learning rate"
    )
):
    """Train with validation."""
    pass
```

### Enum Support

```python
from enum import Enum

class OptimizerType(str, Enum):
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"

@dataclass
class TrainConfig:
    PARAM_DOCS = {
        'optimizer': 'Optimizer algorithm to use',
    }
    optimizer: OptimizerType = OptimizerType.ADAM

# Automatic string-to-enum conversion
python train.py --optimizer sgd  # Converts "sgd" to OptimizerType.SGD
```

## How It Works

The system consists of three main components:

### 1. CONFIG_PARAM Descriptor

A descriptor class that binds function parameters to configuration attributes:

```python
class CONFIG_PARAM:
    def __init__(self, validator=None, transform=None, doc=None, config=None):
        # Stores validation, transformation, and documentation
        ...
```

### 2. Decorators

**`inject_defaults_from_config(config, update_config_with_args=False)`**
- Binds CONFIG_PARAM instances to config attributes
- Provides default values from configuration
- Optionally updates config when function is called with new values

**`inject_docs_from_config_params(func)`**
- Injects parameter documentation into function docstrings
- Replaces `{param_name}` placeholders with actual documentation
- Works for both CONFIG_PARAM and regular parameters

### 3. Configuration System

**`ConfigOverrideSystem`**
- Parses config assignments (`key=value`)
- Loads YAML configuration files
- Applies overrides to config objects
- Handles type conversion (int, float, Path, Enum, etc.)

**`ConfigArgumentParser`**
- Extends `argParseFromDoc.AutoArgumentParser`
- Adds `--config`, `--show-config`, and `--export-config` arguments
- Manages precedence: direct args > `--config` > defaults
- Detects and reports conflicts

## Documentation Flow

1. **Define** documentation in `PARAM_DOCS` dictionary in config dataclass
2. **Bind** CONFIG_PARAMs to config attributes via `inject_defaults_from_config`
3. **Inject** documentation into docstrings via `inject_docs_from_config_params`
4. **Generate** CLI help automatically with `ConfigArgumentParser`

```
PARAM_DOCS (config class)
    ↓
CONFIG_PARAM.doc (auto-populated)
    ↓
Function docstring (via {placeholder})
    ↓
CLI help text (via argParseFromDoc)
```

## API Reference

### Core Classes

- `CONFIG_PARAM`: Descriptor for config-backed parameters
- `ConfigOverrideSystem`: Utility class for config manipulation
- `ConfigArgumentParser`: CLI parser with config integration

### Decorators

- `inject_defaults_from_config(config, update_config_with_args=False)`
- `inject_docs_from_config_params(func)`

### Utilities

- `merge_dicts(d1, d2)`: Recursively merge dictionaries
- `dataclass_to_dict(obj)`: Convert dataclass to dictionary
- `export_config_to_yaml(config, filepath)`: Export config to YAML
- `check_type_match(expected_type, actual_value)`: Type checking utility


## Performance overhead
Using the decorators adds a non-negligible overhead to each function call, hence we only recommend using them in class builders and outer-loop functions.

## Examples

See the `examples/` directory for complete working examples:

- `basic_example.py`: Simple configuration and CLI
- `multi_config_example.py`: Hierarchical configurations
- `full_workflow.py`: Complete training script pattern

## Testing

Run tests with pytest:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=autoCLI_config --cov-report=term-missing
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Author

Ruben Sanchez-Garcia (rsanchezgarcia@faculty.ie.edu)

## Acknowledgments

This system was originally developed as part of the [cryoPARES](https://github.com/rsanchezgarc/cryoPARES) project for cryo-EM structure determination, and has been extracted into a standalone package for broader use.
