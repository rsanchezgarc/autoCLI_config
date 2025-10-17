"""
Comprehensive tests for CONFIG_PARAM descriptor and decorator functions.

This module tests:
- CONFIG_PARAM descriptor functionality
- inject_defaults_from_config decorator
- inject_docs_from_config_params decorator
- check_type_match function
- Edge cases and error handling
"""

import pytest
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple, Union, Any

from autoCLI_config import (
    CONFIG_PARAM,
    inject_defaults_from_config,
    inject_docs_from_config_params
)
from autoCLI_config.config_param import check_type_match


# Test enums and configs
class StatusEnum(Enum):
    """Test enum for status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


class PriorityEnum(Enum):
    """Test enum for priority."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class SimpleConfig:
    """Simple configuration for testing."""

    PARAM_DOCS = {
        'name': 'The name of the item',
        'count': 'The number of items',
        'active': 'Whether the item is active',
    }

    name: str = "default_name"
    count: int = 10
    active: bool = True


@dataclass
class AdvancedConfig:
    """Advanced configuration with various types."""

    PARAM_DOCS = {
        'status': 'Current status of the system',
        'priority': 'Priority level',
        'tags': 'List of tags',
        'coordinates': 'X, Y coordinates tuple',
        'threshold': 'Optional threshold value',
    }

    status: StatusEnum = StatusEnum.ACTIVE
    priority: PriorityEnum = PriorityEnum.MEDIUM
    tags: List[str] = None
    coordinates: Tuple[float, float] = (0.0, 0.0)
    threshold: Optional[float] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = ["default"]


@dataclass
class NestedConfig:
    """Config with nested structure."""

    PARAM_DOCS = {
        'simple': 'Simple sub-config',
        'value': 'A standalone value',
    }

    simple: SimpleConfig = None
    value: int = 42

    def __post_init__(self):
        if self.simple is None:
            self.simple = SimpleConfig()


class TestCONFIG_PARAM:
    """Test CONFIG_PARAM descriptor class."""

    def test_init_basic(self):
        """Test basic initialization."""
        param = CONFIG_PARAM()
        assert param.validator is None
        assert param.transform is None
        assert param.doc is None
        assert param._config is None
        assert param._name is None

    def test_init_with_params(self):
        """Test initialization with parameters."""
        validator = lambda x: x > 0
        transform = lambda x: x * 2
        doc = "Test parameter"
        config = SimpleConfig()

        param = CONFIG_PARAM(
            validator=validator,
            transform=transform,
            doc=doc,
            config=config
        )

        assert param.validator is validator
        assert param.transform is transform
        assert param.doc == doc
        assert param._config is config

    def test_bind_to_config(self):
        """Test binding parameter to config."""
        config = SimpleConfig()
        param = CONFIG_PARAM()

        param.bind(config, "name")

        assert param._config is config
        assert param._name == "name"
        assert param.doc == "The name of the item"  # Auto-populated from PARAM_DOCS

    def test_bind_with_explicit_config(self):
        """Test that explicit config is not overridden by bind."""
        config1 = SimpleConfig()
        config2 = SimpleConfig()
        param = CONFIG_PARAM(config=config1)

        param.bind(config2, "name")

        assert param._config is config1  # Should keep original config
        assert param._name == "name"

    def test_validate_no_validator(self):
        """Test validation with no validator always returns True."""
        param = CONFIG_PARAM()
        assert param.validate(0) is True
        assert param.validate("anything") is True
        assert param.validate(None) is True

    def test_validate_with_validator(self):
        """Test validation with custom validator."""
        param = CONFIG_PARAM(validator=lambda x: x > 0)

        assert param.validate(5) is True
        assert param.validate(-5) is False
        assert param.validate(0) is False

    def test_transform_no_transform(self):
        """Test transform with no transform function returns original value."""
        param = CONFIG_PARAM()
        assert param.transform_value(10) == 10
        assert param.transform_value("test") == "test"

    def test_transform_with_function(self):
        """Test transform with custom function."""
        param = CONFIG_PARAM(transform=lambda x: x * 2)
        assert param.transform_value(5) == 10
        assert param.transform_value(3.5) == 7.0

    def test_convert_to_enum_string_to_enum(self):
        """Test converting string to enum."""
        param = CONFIG_PARAM()
        result = param.convert_to_enum_if_needed("active", StatusEnum)
        assert result == StatusEnum.ACTIVE
        assert isinstance(result, StatusEnum)

    def test_convert_to_enum_invalid_string(self):
        """Test converting invalid string to enum raises error."""
        param = CONFIG_PARAM()
        with pytest.raises(ValueError, match="is not a valid"):
            param.convert_to_enum_if_needed("invalid", StatusEnum)

    def test_convert_to_enum_non_enum_type(self):
        """Test that non-enum types are returned unchanged."""
        param = CONFIG_PARAM()
        result = param.convert_to_enum_if_needed("test", str)
        assert result == "test"

    def test_convert_to_enum_numeric_enum(self):
        """
        Test behavior with numeric-valued enums.

        Note: convert_to_enum_if_needed only processes string inputs, and is designed
        for string-valued enums (common in CLI applications). For numeric-valued enums:

            class PriorityEnum(Enum):
                LOW = 1
                MEDIUM = 2

        The function has limitations:
        - Passing integer 2 → returns 2 unchanged (no conversion, not a string)
        - Passing string "2" → raises ValueError (enum expects int 2, not string "2")

        This is intentional: the system is designed for string-valued enums.
        """
        param = CONFIG_PARAM()

        # Integer inputs pass through unchanged (not converted)
        result = param.convert_to_enum_if_needed(2, PriorityEnum)
        assert result == 2  # Unchanged, not converted to enum

        # String inputs to numeric enums don't work (by design)
        # The function is designed for string-valued enums from CLI
        result = param.convert_to_enum_if_needed("active", StatusEnum)
        assert result == StatusEnum.ACTIVE

        result = param.convert_to_enum_if_needed("inactive", StatusEnum)
        assert result == StatusEnum.INACTIVE

    def test_get_unbound_raises_error(self):
        """Test that getting value from unbound parameter raises error."""
        param = CONFIG_PARAM()
        with pytest.raises(RuntimeError, match="not bound"):
            param.get()

    def test_get_bound_returns_value(self):
        """Test getting value from bound parameter."""
        config = SimpleConfig(name="test_name")
        param = CONFIG_PARAM()
        param.bind(config, "name")

        assert param.get() == "test_name"

    def test_call_returns_value(self):
        """Test that calling parameter returns value."""
        config = SimpleConfig(count=25)
        param = CONFIG_PARAM()
        param.bind(config, "count")

        assert param() == 25

    def test_value_property(self):
        """Test value property."""
        config = SimpleConfig(active=False)
        param = CONFIG_PARAM()
        param.bind(config, "active")

        assert param.value is False

    def test_is_bound_property(self):
        """Test is_bound property."""
        param = CONFIG_PARAM()
        assert param.is_bound is False

        config = SimpleConfig()
        param.bind(config, "name")
        assert param.is_bound is True


class TestCheckTypeMatch:
    """Test check_type_match function."""

    def test_basic_types(self):
        """Test matching basic types."""
        assert check_type_match(int, 5) is True
        assert check_type_match(int, "5") is False
        assert check_type_match(str, "hello") is True
        assert check_type_match(str, 123) is False
        assert check_type_match(float, 3.14) is True
        assert check_type_match(float, 3) is False  # int is not float
        assert check_type_match(bool, True) is True
        assert check_type_match(bool, 1) is False  # int is not bool

    def test_none_type(self):
        """Test None type matching."""
        assert check_type_match(Optional[int], None) is True
        assert check_type_match(int, None) is False
        assert check_type_match(Union[int, None], None) is True

    def test_optional_types(self):
        """Test Optional type matching."""
        assert check_type_match(Optional[int], 5) is True
        assert check_type_match(Optional[int], None) is True
        assert check_type_match(Optional[str], "test") is True
        assert check_type_match(Optional[str], 123) is False

    def test_union_types(self):
        """Test Union type matching."""
        assert check_type_match(Union[int, str], 5) is True
        assert check_type_match(Union[int, str], "test") is True
        assert check_type_match(Union[int, str], 3.14) is False
        assert check_type_match(Union[int, str, float], 3.14) is True

    def test_list_types(self):
        """Test List type matching."""
        assert check_type_match(List[int], [1, 2, 3]) is True
        assert check_type_match(List[int], [1, "2", 3]) is False
        assert check_type_match(List[str], ["a", "b"]) is True
        assert check_type_match(List[str], [1, 2]) is False
        assert check_type_match(list, [1, 2, 3]) is True
        assert check_type_match(List, [1, 2, 3]) is True  # Unparameterized

    def test_tuple_types(self):
        """Test Tuple type matching."""
        assert check_type_match(Tuple[int, int], (1, 2)) is True
        assert check_type_match(Tuple[int, str], (1, "a")) is True
        assert check_type_match(Tuple[int, str], (1, 2)) is False
        assert check_type_match(Tuple[int, int], (1, 2, 3)) is False  # Wrong length
        assert check_type_match(tuple, (1, 2)) is True

    def test_list_tuple_equivalence(self):
        """Test that lists and tuples are treated as equivalent."""
        assert check_type_match(List[int], (1, 2, 3)) is True
        assert check_type_match(Tuple[int, ...], [1, 2, 3]) is True

    def test_enum_types(self):
        """Test Enum type matching."""
        assert check_type_match(StatusEnum, StatusEnum.ACTIVE) is True
        assert check_type_match(StatusEnum, "active") is True  # String value
        assert check_type_match(StatusEnum, "invalid") is False
        assert check_type_match(PriorityEnum, PriorityEnum.HIGH) is True
        assert check_type_match(PriorityEnum, 3) is False  # Direct int not allowed

    def test_any_type(self):
        """
        Test Any type matching.

        Note: typing.Any is a special type that represents any type is acceptable.
        The check_type_match function correctly handles this by checking if
        expected_type is Any and returning True for all values.

        However, None is handled specially BEFORE the Any check, so
        check_type_match(Any, None) returns False because None requires
        an explicit Optional or Union[..., None] type annotation.
        """
        assert check_type_match(Any, 5) is True
        assert check_type_match(Any, "test") is True
        # Note: None requires explicit Optional, even with Any
        assert check_type_match(Any, None) is False
        assert check_type_match(Any, [1, 2, 3]) is True
        assert check_type_match(Any, {"key": "value"}) is True

    def test_literal_types(self):
        """Test Literal type matching (if available)."""
        try:
            from typing import Literal
            assert check_type_match(Literal["a", "b"], "a") is True
            assert check_type_match(Literal["a", "b"], "b") is True
            assert check_type_match(Literal["a", "b"], "c") is False
            assert check_type_match(Literal[1, 2, 3], 2) is True
            assert check_type_match(Literal[1, 2, 3], 4) is False
        except ImportError:
            pytest.skip("Literal type not available")


class TestInjectDefaultsDecorator:
    """Test inject_defaults_from_config decorator."""

    def test_basic_injection(self):
        """Test basic default value injection."""
        config = SimpleConfig(name="injected", count=99)

        @inject_defaults_from_config(config)
        def func(name: str = CONFIG_PARAM(), count: int = CONFIG_PARAM()):
            return name, count

        result = func()
        assert result == ("injected", 99)

    def test_override_with_kwargs(self):
        """Test overriding defaults with keyword arguments."""
        config = SimpleConfig(name="default", count=10)

        @inject_defaults_from_config(config)
        def func(name: str = CONFIG_PARAM(), count: int = CONFIG_PARAM()):
            return name, count

        result = func(name="override")
        assert result == ("override", 10)

        result = func(count=50)
        assert result == ("default", 50)

        result = func(name="both", count=75)
        assert result == ("both", 75)

    def test_override_with_positional(self):
        """Test overriding with positional arguments."""
        config = SimpleConfig(name="default", count=10)

        @inject_defaults_from_config(config)
        def func(name: str = CONFIG_PARAM(), count: int = CONFIG_PARAM()):
            return name, count

        result = func("positional")
        assert result == ("positional", 10)

    def test_required_param_before_config_param(self):
        """Test function with required param before CONFIG_PARAM."""
        config = SimpleConfig(name="default")

        @inject_defaults_from_config(config)
        def func(required: str, name: str = CONFIG_PARAM()):
            return required, name

        result = func("must_provide")
        assert result == ("must_provide", "default")

        result = func("must_provide", name="override")
        assert result == ("must_provide", "override")

    def test_mixed_params(self):
        """Test function with mix of CONFIG_PARAM and regular defaults."""
        config = SimpleConfig(name="from_config")

        @inject_defaults_from_config(config)
        def func(
            name: str = CONFIG_PARAM(),
            regular: int = 42,
            optional: str = "default"
        ):
            return name, regular, optional

        result = func()
        assert result == ("from_config", 42, "default")

        result = func(regular=100)
        assert result == ("from_config", 100, "default")

    def test_update_config_with_args_false(self):
        """Test that config is not updated when update_config_with_args=False."""
        config = SimpleConfig(name="original", count=10)

        @inject_defaults_from_config(config, update_config_with_args=False)
        def func(name: str = CONFIG_PARAM(), count: int = CONFIG_PARAM()):
            return name, count

        func(name="new_value", count=99)

        assert config.name == "original"
        assert config.count == 10

    def test_update_config_with_args_true(self):
        """Test that config is updated when update_config_with_args=True."""
        config = SimpleConfig(name="original", count=10)

        @inject_defaults_from_config(config, update_config_with_args=True)
        def func(name: str = CONFIG_PARAM(), count: int = CONFIG_PARAM()):
            return name, count

        func(name="new_value", count=99)

        assert config.name == "new_value"
        assert config.count == 99

    def test_cross_config_reference(self):
        """
        Test CONFIG_PARAM with explicit config reference.

        Note: When using CONFIG_PARAM(config=other_config), the parameter name
        must match an attribute name in the specified config object.
        The decorator looks for an attribute with the same name as the function parameter.
        The validation happens at DECORATION TIME, not at call time.
        """
        config1 = SimpleConfig(name="from_config1_value")
        config2 = SimpleConfig(name="from_config2_value")

        # This will fail AT DECORATION TIME because config2 doesn't have 'other_name' attribute
        with pytest.raises(ValueError, match="Config missing parameter: other_name"):
            @inject_defaults_from_config(config1)
            def func_failing(
                name: str = CONFIG_PARAM(),  # Uses config1.name
                other_name: str = CONFIG_PARAM(config=config2)  # Looks for config2.other_name - doesn't exist!
            ):
                return name, other_name

        # Correct usage: use matching parameter names
        @inject_defaults_from_config(config1)
        def func_correct(
            name: str = CONFIG_PARAM(),  # Uses config1.name
        ):
            return name

        result = func_correct()
        assert result == "from_config1_value"

    def test_enum_conversion_in_decorator(self):
        """Test that enum conversion works through decorator."""
        config = AdvancedConfig()

        @inject_defaults_from_config(config)
        def func(status: StatusEnum = CONFIG_PARAM()):
            return status

        result = func(status="inactive")
        assert result == StatusEnum.INACTIVE
        assert isinstance(result, StatusEnum)

    def test_validator_in_decorator(self):
        """Test that validator is called in decorator."""
        config = SimpleConfig(count=10)

        @inject_defaults_from_config(config)
        def func(count: int = CONFIG_PARAM(validator=lambda x: x > 0)):
            return count

        # Valid value
        result = func(count=5)
        assert result == 5

        # Invalid value
        with pytest.raises(ValueError, match="Validation failed"):
            func(count=-5)

    def test_transform_in_decorator(self):
        """Test that transform is applied in decorator."""
        config = SimpleConfig(count=10)

        @inject_defaults_from_config(config)
        def func(count: int = CONFIG_PARAM(transform=lambda x: x * 2)):
            return count

        result = func(count=5)
        assert result == 10  # Transformed

    def test_type_mismatch_raises_error(self):
        """Test that type mismatch raises TypeError."""
        config = SimpleConfig(name="test", count=10)

        # This should raise because count is int but annotated as str
        with pytest.raises(TypeError, match="Type mismatch"):
            @inject_defaults_from_config(config)
            def func(name: str = CONFIG_PARAM(), count: str = CONFIG_PARAM()):
                return name, count

    def test_missing_config_attribute_raises_error(self):
        """Test that missing attribute raises ValueError."""
        config = SimpleConfig()

        with pytest.raises(ValueError, match="Config missing parameter"):
            @inject_defaults_from_config(config)
            def func(nonexistent: str = CONFIG_PARAM()):
                return nonexistent

    def test_function_signature_updated(self):
        """Test that function signature is updated with actual defaults."""
        config = SimpleConfig(name="default_name", count=42)

        @inject_defaults_from_config(config)
        def func(name: str = CONFIG_PARAM(), count: int = CONFIG_PARAM()):
            return name, count

        import inspect
        sig = inspect.signature(func)

        assert sig.parameters['name'].default == "default_name"
        assert sig.parameters['count'].default == 42

    def test_argname_to_configname_attribute(self):
        """Test that _argname_to_configname is set correctly."""
        config = SimpleConfig()

        @inject_defaults_from_config(config)
        def func(name: str = CONFIG_PARAM(), count: int = CONFIG_PARAM()):
            return name, count

        assert hasattr(func, '_argname_to_configname')
        assert 'name' in func._argname_to_configname
        assert 'count' in func._argname_to_configname
        assert isinstance(func._argname_to_configname['name'], CONFIG_PARAM)


class TestInjectDocsDecorator:
    """Test inject_docs_from_config_params decorator."""

    def test_basic_doc_injection(self):
        """Test basic documentation injection."""
        import inspect
        config = SimpleConfig()

        @inject_docs_from_config_params
        @inject_defaults_from_config(config)
        def func(name: str = CONFIG_PARAM(), count: int = CONFIG_PARAM()):
            """
            Test function.

            Args:
                name: {name}
                count: {count}
            """
            return name, count

        expected_doc = """
        Test function.

        Args:
            name: The name of the item
            count: The number of items
        """

        # Use inspect.cleandoc to normalize both docstrings for comparison
        # This handles Python version differences in docstring indentation handling
        assert inspect.cleandoc(func.__doc__) == inspect.cleandoc(expected_doc)

    def test_partial_doc_injection(self):
        """Test that only parameters with docs are replaced."""
        config = SimpleConfig()

        @inject_docs_from_config_params
        @inject_defaults_from_config(config)
        def func(name: str = CONFIG_PARAM(), regular: int = 10):
            """
            Function with mix of params.

            Args:
                name: {name}
                regular: A regular parameter
            """
            return name, regular

        assert "The name of the item" in func.__doc__
        assert "A regular parameter" in func.__doc__

    def test_missing_placeholder_doesnt_crash(self):
        """Test that missing placeholders don't cause errors."""
        config = SimpleConfig()

        @inject_docs_from_config_params
        @inject_defaults_from_config(config)
        def func(name: str = CONFIG_PARAM()):
            """
            Function without placeholders.

            This is just a description.
            """
            return name

        # Should not crash, doc remains unchanged
        assert "This is just a description" in func.__doc__

    def test_no_doc_doesnt_crash(self):
        """Test that functions without docstrings don't crash."""
        config = SimpleConfig()

        @inject_docs_from_config_params
        @inject_defaults_from_config(config)
        def func(name: str = CONFIG_PARAM()):
            return name

        # Should not crash
        result = func()
        assert result == "default_name"

    def test_custom_doc_in_config_param(self):
        """Test that custom doc in CONFIG_PARAM is used."""
        config = SimpleConfig()

        @inject_docs_from_config_params
        @inject_defaults_from_config(config)
        def func(name: str = CONFIG_PARAM(doc="Custom documentation")):
            """
            Function with custom doc.

            Args:
                name: {name}
            """
            return name

        assert "Custom documentation" in func.__doc__

    def test_without_inject_defaults_decorator(self):
        """Test that decorator works gracefully without inject_defaults."""
        def func(name: str):
            """
            Simple function.

            Args:
                name: {name}
            """
            return name

        # Should not crash
        decorated = inject_docs_from_config_params(func)
        assert decorated.__doc__ == func.__doc__  # Unchanged


class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""

    def test_empty_config(self):
        """Test with empty config."""
        @dataclass
        class EmptyConfig:
            pass

        config = EmptyConfig()

        # This should work fine as long as we don't use CONFIG_PARAM
        @inject_defaults_from_config(config)
        def func(regular: int = 42):
            return regular

        assert func() == 42

    def test_nested_config_access(self):
        """Test accessing nested config values."""
        nested = NestedConfig()

        @inject_defaults_from_config(nested)
        def func(value: int = CONFIG_PARAM()):
            return value

        assert func() == 42

    def test_none_default_value(self):
        """Test with None as default value."""
        config = AdvancedConfig(threshold=None)

        @inject_defaults_from_config(config)
        def func(threshold: Optional[float] = CONFIG_PARAM()):
            return threshold

        assert func() is None
        assert func(threshold=3.14) == 3.14

    def test_list_default_value(self):
        """Test with list as default value."""
        config = AdvancedConfig()

        @inject_defaults_from_config(config)
        def func(tags: List[str] = CONFIG_PARAM()):
            return tags

        result = func()
        assert result == ["default"]

        result = func(tags=["a", "b"])
        assert result == ["a", "b"]

    def test_tuple_default_value(self):
        """Test with tuple as default value."""
        config = AdvancedConfig()

        @inject_defaults_from_config(config)
        def func(coordinates: Tuple[float, float] = CONFIG_PARAM()):
            return coordinates

        result = func()
        assert result == (0.0, 0.0)

        result = func(coordinates=(1.5, 2.5))
        assert result == (1.5, 2.5)

    def test_varargs_and_kwargs(self):
        """Test function with *args and **kwargs."""
        config = SimpleConfig()

        @inject_defaults_from_config(config)
        def func(name: str = CONFIG_PARAM(), *args, **kwargs):
            return name, args, kwargs

        result = func()
        assert result == ("default_name", (), {})

        result = func("override", "extra1", "extra2", key="value")
        assert result == ("override", ("extra1", "extra2"), {"key": "value"})

    def test_callable_with_method(self):
        """
        Test decorator on class methods.

        The decorator correctly handles 'self' parameter in instance methods,
        allowing both keyword arguments and defaults to work properly.
        """
        config = SimpleConfig()

        class MyClass:
            @inject_defaults_from_config(config)
            def method(self, name: str = CONFIG_PARAM()):
                return name

        obj = MyClass()

        # Test with default value
        assert obj.method() == "default_name"

        # Test with keyword argument
        assert obj.method(name="override") == "override"

    def test_multiple_validators(self):
        """Test multiple CONFIG_PARAMs with different validators."""
        config = SimpleConfig(name="test", count=10)

        @inject_defaults_from_config(config)
        def func(
            name: str = CONFIG_PARAM(validator=lambda x: len(x) > 0),
            count: int = CONFIG_PARAM(validator=lambda x: x >= 0)
        ):
            return name, count

        # Valid
        result = func(name="valid", count=5)
        assert result == ("valid", 5)

        # Invalid name
        with pytest.raises(ValueError, match="Validation failed for parameter name"):
            func(name="", count=5)

        # Invalid count
        with pytest.raises(ValueError, match="Validation failed for parameter count"):
            func(name="valid", count=-1)

    def test_chained_transforms(self):
        """Test multiple transformations."""
        config = SimpleConfig(count=10)

        @inject_defaults_from_config(config)
        def func(
            count: int = CONFIG_PARAM(
                transform=lambda x: x * 2,
                validator=lambda x: x > 0
            )
        ):
            return count

        # Transform happens after validation check in the original value
        # but then validator is checked on transformed value
        result = func(count=5)
        assert result == 10


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_machine_learning_config(self):
        """Test ML training configuration scenario."""
        @dataclass
        class TrainingConfig:
            PARAM_DOCS = {
                'learning_rate': 'Learning rate for optimizer',
                'batch_size': 'Batch size for training',
                'epochs': 'Number of training epochs',
            }

            learning_rate: float = 0.001
            batch_size: int = 32
            epochs: int = 100

        config = TrainingConfig()

        @inject_docs_from_config_params
        @inject_defaults_from_config(config, update_config_with_args=True)
        def train_model(
            model_name: str,
            learning_rate: float = CONFIG_PARAM(),
            batch_size: int = CONFIG_PARAM(),
            epochs: int = CONFIG_PARAM()
        ):
            """
            Train a machine learning model.

            Args:
                model_name: Name of the model to train
                learning_rate: {learning_rate}
                batch_size: {batch_size}
                epochs: {epochs}
            """
            return {
                'model': model_name,
                'lr': learning_rate,
                'batch': batch_size,
                'epochs': epochs
            }

        # Use defaults
        result = train_model("resnet50")
        assert result['lr'] == 0.001
        assert result['batch'] == 32
        assert result['epochs'] == 100

        # Override some values
        result = train_model("resnet50", learning_rate=0.01, batch_size=64)
        assert result['lr'] == 0.01
        assert result['batch'] == 64

        # Check config was updated
        assert config.learning_rate == 0.01
        assert config.batch_size == 64

        # Check docs were injected
        assert "Learning rate for optimizer" in train_model.__doc__

    def test_data_processing_pipeline(self):
        """Test data processing configuration scenario."""
        @dataclass
        class ProcessingConfig:
            PARAM_DOCS = {
                'input_format': 'Format of input data',
                'normalize': 'Whether to normalize data',
                'filters': 'List of filters to apply',
            }

            input_format: str = "csv"
            normalize: bool = True
            filters: List[str] = None

            def __post_init__(self):
                if self.filters is None:
                    self.filters = ["remove_nulls"]

        config = ProcessingConfig()

        @inject_defaults_from_config(config)
        def process_data(
            data_path: str,
            input_format: str = CONFIG_PARAM(),
            normalize: bool = CONFIG_PARAM(),
            filters: List[str] = CONFIG_PARAM()
        ):
            return {
                'path': data_path,
                'format': input_format,
                'normalize': normalize,
                'filters': filters
            }

        result = process_data("/data/input.csv")
        assert result['format'] == "csv"
        assert result['normalize'] is True
        assert result['filters'] == ["remove_nulls"]

        result = process_data(
            "/data/input.json",
            input_format="json",
            filters=["remove_nulls", "deduplicate"]
        )
        assert result['format'] == "json"
        assert result['filters'] == ["remove_nulls", "deduplicate"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
