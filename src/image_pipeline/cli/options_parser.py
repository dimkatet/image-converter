"""
Parser for save options in key=value format
"""
from typing import List, Any
from image_pipeline.types import SaveOptions


def parse_options(option_strings: List[str]) -> SaveOptions:
    """
    Parse save options from key=value strings

    Args:
        option_strings: List of "key=value" strings

    Returns:
        SaveOptions dictionary

    Raises:
        ValueError: If option format is invalid

    Examples:
        >>> parse_options(['quality=90', 'lossless=true'])
        {'quality': 90, 'lossless': True}

        >>> parse_options(['strategy=3', 'method=6'])
        {'strategy': 3, 'method': 6}
    """
    if not option_strings:
        return {}

    options: SaveOptions = {}

    for option_str in option_strings:
        # Parse key=value
        if '=' not in option_str:
            raise ValueError(
                f"Invalid option format: '{option_str}'. "
                f"Expected format: key=value (e.g., strategy=3)"
            )

        key, value_str = option_str.split('=', 1)
        key = key.strip()
        value_str = value_str.strip()

        if not key:
            raise ValueError(f"Empty key in option: '{option_str}'")

        if not value_str:
            raise ValueError(f"Empty value for key '{key}'")

        # Parse value (auto-detect type)
        value = _parse_value(value_str)

        # Store in options dict
        options[key] = value  # type: ignore

    return options


def _parse_value(value_str: str) -> Any:
    """
    Parse value string to appropriate Python type

    Supports: int, float, bool, str

    Args:
        value_str: String value to parse

    Returns:
        Parsed value (int, float, bool, or str)

    Examples:
        >>> _parse_value('123')
        123
        >>> _parse_value('3.14')
        3.14
        >>> _parse_value('true')
        True
        >>> _parse_value('hello')
        'hello'
    """
    # Try bool first (case-insensitive)
    if value_str.lower() in ('true', 'yes', '1', 'on'):
        return True
    if value_str.lower() in ('false', 'no', '0', 'off'):
        return False

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

    # Fall back to string
    return value_str
