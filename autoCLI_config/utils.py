"""
Utility functions for config manipulation.

This module provides helper functions for working with configuration dataclasses,
including conversion to/from dictionaries and YAML export.
"""

from dataclasses import is_dataclass, fields
from pathlib import Path
from typing import Any, Dict
import yaml


def merge_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Args:
        d1: First dictionary
        d2: Second dictionary (takes precedence in conflicts)

    Returns:
        Merged dictionary
    """
    result = d1.copy()
    for key, value in d2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def dataclass_to_dict(obj: Any) -> Any:
    """
    Convert a dataclass (and nested dataclasses) to a dictionary.

    Args:
        obj: Object to convert (dataclass, list, dict, or primitive)

    Returns:
        Dictionary representation of the object
    """
    if is_dataclass(obj):
        result = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def export_config_to_yaml(config: Any, filepath: str) -> None:
    """
    Export a config object to a YAML file.

    Args:
        config: Configuration dataclass to export
        filepath: Path to output YAML file
    """
    config_dict = dataclass_to_dict(config)
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
