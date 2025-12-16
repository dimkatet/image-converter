"""
Filter parser for CLI
Parses filter strings like 'blur:sigma=2.5' into filter objects
"""
from typing import List, Dict, Any

from image_pipeline.filters.base import ImageFilter
from .filter_registry import FILTER_REGISTRY, get_available_filters
from .color_space_registry import COLOR_SPACE_ALIASES


def parse_value(value_str: str) -> Any:
    """
    Auto-detect and convert string value to appropriate Python type

    Args:
        value_str: String value to parse

    Returns:
        Converted value (int, float, ColorSpace enum, or string)

    Examples:
        '5' -> 5
        '2.5' -> 2.5
        'BT.709' -> ColorSpace.BT709
        'bt709' -> ColorSpace.BT709
        'sRGB' -> ColorSpace.BT709
        'p3' -> ColorSpace.DISPLAY_P3
        'hello' -> 'hello'
    """
    # Try int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Try ColorSpace enum via alias mapping (case-insensitive)
    value_lower = value_str.lower()
    if value_lower in COLOR_SPACE_ALIASES:
        return COLOR_SPACE_ALIASES[value_lower]

    # Keep as string
    return value_str


def parse_filter(filter_str: str) -> ImageFilter:
    """
    Parse filter string into filter object
    
    Args:
        filter_str: Filter specification string
        
    Returns:
        Initialized filter object
        
    Raises:
        ValueError: If filter name is unknown or parsing fails
        
    Examples:
        'blur:sigma=5' -> BlurFilter(sigma=5)
        'quantize:bit_depth=16' -> QuantizeFilter(bit_depth=16)
        'remove_alpha' -> RemoveAlphaFilter()
        'normalize:min_val=0.0,max_val=1.0' -> NormalizeFilter(min_val=0.0, max_val=1.0)
    """
    # Split filter name and parameters
    parts = filter_str.split(':', 1)
    filter_name = parts[0].strip()
    
    # Check if filter exists
    if filter_name not in FILTER_REGISTRY:
        available = get_available_filters()
        raise ValueError(
            f"Unknown filter: '{filter_name}'\n\n{available}"
        )
    
    filter_class = FILTER_REGISTRY[filter_name]
    
    # Parse parameters if present
    params = {}
    if len(parts) > 1:
        param_str = parts[1].strip()
        
        # Split by comma for multiple parameters
        for param_pair in param_str.split(','):
            param_pair = param_pair.strip()
            
            if '=' not in param_pair:
                raise ValueError(
                    f"Invalid parameter format in '{filter_str}'. "
                    f"Expected 'key=value', got '{param_pair}'"
                )
            
            # Split key=value
            key, value_str = param_pair.split('=', 1)
            key = key.strip()
            value_str = value_str.strip()
            
            # Convert value to appropriate type
            params[key] = parse_value(value_str)
    
    # Create filter instance
    try:
        return filter_class(**params)
    except TypeError as e:
        raise ValueError(
            f"Error creating filter '{filter_name}': {e}\n"
            f"Check parameter names and types."
        ) from e


def parse_filters(filter_strings: List[str]) -> List[ImageFilter]:
    """
    Parse multiple filter strings into filter objects
    
    Args:
        filter_strings: List of filter specification strings
        
    Returns:
        List of initialized filter objects
        
    Raises:
        ValueError: If any filter string is invalid
    """
    if not filter_strings:
        return []
    
    filters: List[ImageFilter] = []
    for i, filter_str in enumerate(filter_strings, 1):
        try:
            filter_obj = parse_filter(filter_str)
            filters.append(filter_obj)
        except ValueError as e:
            raise ValueError(f"Error in filter #{i} ('{filter_str}'): {e}") from e
    
    return filters