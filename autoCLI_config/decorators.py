"""
Decorators for config injection and documentation.

This module provides decorators that enable automatic default value injection
from configuration dataclasses and documentation propagation from PARAM_DOCS.
"""

import inspect
from functools import wraps
from typing import Any, get_type_hints, Dict

from .config_param import CONFIG_PARAM, check_type_match


def inject_docs_from_config_params(func):
    """
    Decorator to inject parameter documentation from CONFIG_PARAMs and PARAM_DOCS into function docstrings.

    This decorator should be applied AFTER inject_defaults_from_config, as it relies on
    the _argname_to_configname attribute created by that decorator.

    Usage:
        @inject_docs_from_config_params
        @inject_defaults_from_config(config)
        def my_function(param1: int = CONFIG_PARAM(), param2: str = "default"):
            '''
            My function.

            :param param1: {param1}
            :param param2: {param2}
            '''
            pass

    The {param1} placeholder will be replaced with the documentation from CONFIG_PARAM.doc,
    which is automatically populated from the config's PARAM_DOCS dictionary.

    The {param2} placeholder will be looked up in the config's PARAM_DOCS dictionary directly,
    allowing non-CONFIG_PARAM parameters to also use centralized documentation.

    Args:
        func: The function to decorate

    Returns:
        The decorated function with updated docstring
    """
    if not hasattr(func, '_argname_to_configname'):
        # Function not decorated with inject_defaults_from_config, nothing to do
        return func

    if func.__doc__:
        docs_dict = {}

        # First, collect docs from CONFIG_PARAMs
        for param_name, config_param in func._argname_to_configname.items():
            if isinstance(config_param, CONFIG_PARAM) and config_param.doc:
                docs_dict[param_name] = config_param.doc

        # Second, look up remaining parameters from the config's PARAM_DOCS
        # Get the config object from the wrapper's stored reference
        if hasattr(func, '_inject_default_config'):
            config = func._inject_default_config
            sig = inspect.signature(func)

            for param_name in sig.parameters:
                # Skip if we already have docs from CONFIG_PARAM
                if param_name in docs_dict:
                    continue

                # Try to find docs in the config's PARAM_DOCS
                if hasattr(config, 'PARAM_DOCS') and param_name in config.PARAM_DOCS:
                    docs_dict[param_name] = config.PARAM_DOCS[param_name]

        if docs_dict:
            try:
                func.__doc__ = func.__doc__.format(**docs_dict)
            except KeyError as e:
                # Missing placeholder in docstring - that's okay, just skip formatting
                pass

    return func


def inject_defaults_from_config(default_config: Any, update_config_with_args: bool = False):
    """
    Decorator that injects default values from a config dataclass into function parameters.

    This decorator binds CONFIG_PARAM instances to their corresponding config attributes,
    enabling automatic default value injection and optional config updates.

    Usage:
        @inject_defaults_from_config(my_config, update_config_with_args=True)
        def my_function(
            required_param: str,
            optional_param: int = CONFIG_PARAM(),
            cross_config_param: float = CONFIG_PARAM(config=other_config)
        ):
            pass

    Args:
        default_config: The default configuration where the default values will be read.
                       Can be parameter-specific overridden by providing CONFIG_PARAM(config=otherConfig)
        update_config_with_args: If True, the config for a parameter will be updated with
                                the new value if it is not a default one.

    Returns:
        Decorator function
    """
    def decorator(func):
        sig = inspect.signature(func)
        hints = get_type_hints(func)

        lazy_params = {}
        param_processors = {}
        param_configs: Dict[str, Any] = {}

        for name, param in sig.parameters.items():
            if isinstance(param.default, CONFIG_PARAM):
                config_to_use = param.default._config or default_config
                param_configs[name] = config_to_use

                if not hasattr(config_to_use, name):
                    raise ValueError(f"Config missing parameter: {name}")

                config_value = getattr(config_to_use, name)
                expected_type = hints.get(name)

                if not check_type_match(expected_type, config_value):
                    raise TypeError(
                        f"Type mismatch for {name}: expected {expected_type}, "
                        f"got {type(config_value)} with value {config_value}"
                    )

                param.default.bind(config_to_use, name)
                lazy_params[name] = param.default
                param_processors[name] = param.default

        @wraps(func)
        def wrapper(*args, **kwargs):
            # First, bind positional arguments to their parameter names
            bound_args = sig.bind_partial(*args, **kwargs)

            # Create final kwargs dict starting with positional args
            final_kwargs = {}
            consumed_positional_names = []
            consumed_first_config_param_positional = False

            # Handle the 'self' parameter for methods
            skip_first = inspect.ismethod(func) or (func.__name__ == '__init__' and 'self' in sig.parameters)
            first_param_name = None
            if skip_first:
                first_param_name = list(sig.parameters.keys())[0]
                if args:
                    final_kwargs[first_param_name] = args[0]
                    args = args[1:]

            # Process positional arguments first
            positional_params = [p for p in sig.parameters.values()
                                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]

            used_positional = 0
            for i, param in enumerate(positional_params[1:] if skip_first else positional_params):
                if i >= len(args):
                    break
                # Decide whether to consume this positional into a named parameter
                take_as_named = True
                if isinstance(param.default, CONFIG_PARAM):
                    # Policy: allow only the FIRST CONFIG_PARAM to be set positionally,
                    # push remaining positionals into *args instead.
                    if consumed_first_config_param_positional:
                        take_as_named = False
                    else:
                        consumed_first_config_param_positional = True
                if not take_as_named:
                    break
                # Use the positional argument value
                value = args[i]
                if param.name in param_processors:
                    processor = param_processors[param.name]
                    expected_type = hints.get(param.name)
                    value = processor.convert_to_enum_if_needed(value, expected_type)
                    value = processor.transform_value(value)
                    if not processor.validate(value):
                        raise ValueError(f"Validation failed for parameter {param.name}")
                    if update_config_with_args:
                        config_to_update = param_configs[param.name]
                        setattr(config_to_update, param.name, value)
                final_kwargs[param.name] = value
                consumed_positional_names.append(param.name)
                used_positional += 1

            # Then process keyword arguments and remaining parameters
            for name, param in sig.parameters.items():
                if name in final_kwargs:  # Skip already processed positional args
                    continue

                if name in kwargs:
                    value = kwargs[name]
                    if name in param_processors:
                        processor = param_processors[name]
                        expected_type = hints.get(param.name)
                        value = processor.convert_to_enum_if_needed(value, expected_type)
                        value = processor.transform_value(value)
                        if not processor.validate(value):
                            raise ValueError(f"Validation failed for parameter {name}")
                        if update_config_with_args:
                            config_to_update = param_configs[name]
                            setattr(config_to_update, name, value)
                    final_kwargs[name] = value
                elif name in lazy_params:
                    final_kwargs[name] = lazy_params[name]()
                elif param.default is not param.empty:
                    final_kwargs[name] = param.default

            for k, v in kwargs.items():
                if k not in final_kwargs and k not in sig.parameters:
                    final_kwargs[k] = v

            # --- Build final call safely ---
            call_args = []
            # Handle first parameter (self/cls) if it's a method
            if first_param_name and first_param_name in final_kwargs:
                call_args.append(final_kwargs.pop(first_param_name))
            # Append consumed named positionals in order and remove from kwargs
            for pname in consumed_positional_names:
                if pname in final_kwargs:  # Check exists before popping
                    call_args.append(final_kwargs.pop(pname))
            # Fill any remaining POSITIONAL_ONLY / POSITIONAL_OR_KEYWORD params positionally
            # so that following extras go into *args (instead of binding to those params).
            for p in (positional_params[1:] if skip_first else positional_params):
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                    if p.name in consumed_positional_names:
                        continue
                    if p.name in final_kwargs:
                        call_args.append(final_kwargs.pop(p.name))
                    # else: if it's not in final_kwargs, it either had no default
                    # (already validated upstream) or will be provided by kwargs later.

            # Any leftover original positionals go to *args if accepted
            extra_positional = args[used_positional:]
            has_var_positional = any(
                p.kind is inspect.Parameter.VAR_POSITIONAL
                for p in sig.parameters.values()
            )
            if has_var_positional:
                call_args.extend(extra_positional)
            # Call with reconstructed args/kwargs
            return func(*call_args, **final_kwargs)

        # Update the signature
        new_params = []
        argname_to_configname = {}
        for param in sig.parameters.values():
            if isinstance(param.default, CONFIG_PARAM):
                config_to_use = param_configs[param.name]
                argname_to_configname[param.name] = param.default
                new_default = getattr(config_to_use, param.name)
                new_params.append(param.replace(default=new_default))
            else:
                new_params.append(param)

        wrapper.__signature__ = sig.replace(parameters=new_params)
        wrapper._argname_to_configname = argname_to_configname  # This is used to keep track of the parameters that had configs as defaults
        wrapper._inject_default_config = default_config  # Store config reference for inject_docs_from_config_params
        return wrapper

    return decorator
