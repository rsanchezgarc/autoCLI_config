"""
autoCLI_config - Automatic Configuration and Documentation System

A Python package for automatic configuration management and documentation generation.
Provides decorators and utilities for:
- Binding function parameters to configuration dataclasses
- Auto-generating CLI help from centralized documentation
- Managing config overrides from YAML files and command-line arguments

Example:
    >>> from dataclasses import dataclass
    >>> from autoCLI_config import CONFIG_PARAM, inject_defaults_from_config, inject_docs_from_config_params
    >>>
    >>> @dataclass
    >>> class MyConfig:
    >>>     PARAM_DOCS = {
    >>>         'learning_rate': 'Learning rate for optimizer',
    >>>         'batch_size': 'Number of samples per batch'
    >>>     }
    >>>     learning_rate: float = 0.001
    >>>     batch_size: int = 32
    >>>
    >>> config = MyConfig()
    >>>
    >>> @inject_docs_from_config_params
    >>> @inject_defaults_from_config(config)
    >>> def train(learning_rate: float = CONFIG_PARAM(), batch_size: int = CONFIG_PARAM()):
    >>>     '''
    >>>     Train a model.
    >>>
    >>>     Args:
    >>>         learning_rate: {learning_rate}
    >>>         batch_size: {batch_size}
    >>>     '''
    >>>     print(f"Training with lr={learning_rate}, batch_size={batch_size}")
"""

__version__ = "0.1.6"
__author__ = "Ruben Sanchez-Garcia"
__email__ = "rsanchezgarcia@faculty.ie.edu"

from .config_param import CONFIG_PARAM, check_type_match
from .decorators import inject_defaults_from_config, inject_docs_from_config_params
from .parser import ConfigOverrideSystem, ConfigArgumentParser
from .utils import merge_dicts, dataclass_to_dict, export_config_to_yaml

__all__ = [
    # Core classes
    'CONFIG_PARAM',

    # Decorators
    'inject_defaults_from_config',
    'inject_docs_from_config_params',

    # Parser and override system
    'ConfigOverrideSystem',
    'ConfigArgumentParser',

    # Utilities
    'merge_dicts',
    'dataclass_to_dict',
    'export_config_to_yaml',
    'check_type_match',
]
