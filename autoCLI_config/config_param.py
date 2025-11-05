"""
CONFIG_PARAM descriptor and type checking utilities.

This module provides the CONFIG_PARAM class which allows function parameters
to be bound to configuration dataclass attributes with validation and transformation.
"""

import inspect
from enum import Enum
from typing import Any, Optional, Callable, get_origin, get_args, Union


class CONFIG_PARAM:
    """Enhanced parameter descriptor for config injection with value tracking."""

    def __init__(
            self,
            validator: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            doc: Optional[str] = None,
            config: Optional[Any] = None,
            is_required_arg_for_cli_fun: Optional[Callable[[], bool]] = None
    ):
        """
        Initialize a CONFIG_PARAM descriptor.

        Args:
            validator: Optional function to validate parameter values
            transform: Optional function to transform parameter values
            doc: Optional documentation string (auto-populated from config's PARAM_DOCS)
            config: Optional specific config object to bind to (overrides decorator's default)
            is_required_arg_for_cli_fun: Optional callable that returns True if this parameter
                should be marked as required for argparse. Evaluated at parse time to allow
                checking sys.argv for --config presence. Use this for parameters that are
                required but can be provided via --config instead of CLI arguments.
        """
        self.validator = validator
        self.transform = transform
        self.doc = doc
        self._config = config
        self._name = None
        self.is_required_arg_for_cli_fun = is_required_arg_for_cli_fun

    def bind(self, config: Any, name: str):
        """Bind this parameter to a specific config attribute."""
        if self._config is None:  # Only bind if no config was specified at creation
            self._config = config
        self._name = name

        # Auto-populate doc from config's PARAM_DOCS if not already provided
        if self.doc is None and hasattr(config, 'PARAM_DOCS'):
            self.doc = config.PARAM_DOCS.get(name)

    def validate(self, value: Any) -> bool:
        """Validate a value using the validator function."""
        if self.validator is None:
            return True
        return self.validator(value)

    def transform_value(self, value: Any) -> Any:
        """Transform a value using the transform function."""
        if self.transform is None:
            return value
        return self.transform(value)

    def convert_to_enum_if_needed(self, value: Any, expected_type: Any) -> Any:
        """Convert string values to enum if the expected type is an enum."""
        if (inspect.isclass(expected_type) and
            issubclass(expected_type, Enum) and
            isinstance(value, str)):
            try:
                return expected_type(value)
            except ValueError:
                # Try to find enum member by value
                for member in expected_type:
                    if member.value == value:
                        return member
                raise ValueError(f"'{value}' is not a valid {expected_type.__name__}")
        return value

    def __call__(self) -> Any:
        """Get the current value from the config."""
        return self.get()

    def get(self):
        """Get the current value from the config."""
        if self._config is None or self._name is None:
            raise RuntimeError("CONFIG_PARAM not bound to a config")
        return getattr(self._config, self._name)

    @property
    def value(self):
        """Property to get the current value from the config."""
        return self.get()

    @property
    def is_bound(self) -> bool:
        """Check if the parameter is bound to a config."""
        return self._config is not None


def check_type_match(expected_type: Any, actual_value: Any) -> bool:
    """
    Enhanced type checking that handles generic types like Tuple, Optional, Union, and Literal.
    Treats lists and tuples as equivalent for type checking purposes.

    Args:
        expected_type: The expected type annotation
        actual_value: The actual value to check

    Returns:
        True if the value matches the expected type, False otherwise
    """
    # Handle None for Optional types
    if actual_value is None:
        # If expected_type is a Union or Optional, check if None is allowed
        if get_origin(expected_type) is Union:
            return type(None) in get_args(expected_type)
        return False

    if inspect.isclass(expected_type) and issubclass(expected_type, Enum):
        # If actual_value is already an instance of the enum, it's valid
        if isinstance(actual_value, expected_type):
            return True
        # If actual_value is a string that matches an enum value, it's valid
        if isinstance(actual_value, str):
            try:
                # Try to create enum from string value
                expected_type(actual_value)
                return True
            except ValueError:
                # Check if the string matches any enum member's value
                return actual_value in [member.value for member in expected_type]
        return False

    # Handle special case where expected_type is Any
    if expected_type is Any:
        return True

    # Get the origin type (e.g., tuple from Tuple[float, float])
    expected_origin = get_origin(expected_type)

    # Handle Literal types
    try:
        from typing import Literal
        if expected_origin is Literal:
            if hasattr(actual_value, "value") and not actual_value in get_args(expected_type):
                actual_value = actual_value.value
            return actual_value in get_args(expected_type)
    except ImportError:
        pass  # Literal not available in older Python versions

    # If there's no origin type, do a direct comparison
    if expected_origin is None:
        return isinstance(actual_value, expected_type)

    # Handle Union types (including Optional)
    if expected_origin is Union:
        return any(check_type_match(arg_type, actual_value)
                   for arg_type in get_args(expected_type))

    # Special handling for sequences (list and tuple)
    if expected_origin in (list, tuple):
        if not isinstance(actual_value, (list, tuple)):
            return False

        # Get the expected argument types
        expected_args = get_args(expected_type)

        # If no arguments specified (just List or Tuple), any sequence is fine
        if not expected_args:
            return True

        # Check if it's a variable-length sequence (List[int], Tuple[int, ...])
        if expected_origin is list or (len(expected_args) == 2 and expected_args[1] is Ellipsis):
            element_type = expected_args[0] if expected_origin is list else expected_args[0]
            return all(check_type_match(element_type, item) for item in actual_value)

        # Fixed-length tuple: check length and types
        if expected_origin is tuple and len(actual_value) != len(expected_args):
            return False

        if expected_origin is tuple:
            return all(check_type_match(expected_arg, actual_item)
                       for expected_arg, actual_item in zip(expected_args, actual_value))

    # For other generic types, just check the origin type but treat list/tuple as equivalent
    if expected_origin in (list, tuple):
        return isinstance(actual_value, (list, tuple))
    return isinstance(actual_value, expected_origin)
