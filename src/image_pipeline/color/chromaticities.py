"""
Chromaticity utilities for color space matching and conversion.

This module provides functions for:
- Matching measured chromaticity coordinates to standard color spaces
- Getting standard primaries for known color spaces
- Converting between different chromaticity representations
"""

from typing import Dict, Tuple, Optional

from image_pipeline.types import ColorSpace
from image_pipeline.constants import STANDARD_COLOR_PRIMARIES


def match_color_space(
    primaries: Dict[str, Tuple[float, float]],
    tolerance: float = 0.001
) -> Optional[ColorSpace]:
    """
    Match chromaticity primaries to a standard color space.

    Compares measured/read chromaticity coordinates against known standards
    (BT.709, BT.2020, Display P3) and returns a match if all primaries are
    within tolerance.

    Args:
        primaries: Dictionary with chromaticity coordinates, format:
                   {'red': (x, y), 'green': (x, y), 'blue': (x, y), 'white': (x, y)}
        tolerance: Maximum allowed difference for matching (default: 0.001)
                   This is approximately 0.1% difference in chromaticity coordinates.

    Returns:
        ColorSpace enum if matched to a standard, None otherwise.

    Example:
        >>> primaries = {
        ...     'red': (0.64, 0.33),
        ...     'green': (0.30, 0.60),
        ...     'blue': (0.15, 0.06),
        ...     'white': (0.3127, 0.3290)
        ... }
        >>> match_color_space(primaries)
        <ColorSpace.BT709: 'BT.709'>
    """
    for color_space, std_primaries in STANDARD_COLOR_PRIMARIES.items():
        # Check if all primaries match within tolerance
        matches = all(
            abs(primaries[color][0] - std_primaries[color][0]) < tolerance and
            abs(primaries[color][1] - std_primaries[color][1]) < tolerance
            for color in ['red', 'green', 'blue', 'white']
        )

        if matches:
            return color_space

    return None


def get_primaries_for_color_space(
    color_space: Optional[ColorSpace]
) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Get standard chromaticity primaries for a color space.

    Args:
        color_space: ColorSpace enum (BT709, BT2020, DISPLAY_P3)

    Returns:
        Dictionary with chromaticity coordinates, or None if color_space is None.

    Example:
        >>> primaries = get_primaries_for_color_space(ColorSpace.BT709)
        >>> primaries['red']
        (0.64, 0.33)
    """
    if color_space is None:
        return None

    return STANDARD_COLOR_PRIMARIES.get(color_space)


def get_primaries_from_metadata(
    color_space: Optional[ColorSpace],
    custom_primaries: Optional[Dict[str, Tuple[float, float]]]
) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Get primaries with priority: custom_primaries > standard color_space.

    Helper function for metadata adapters that need to resolve primaries
    from either custom measurements or standard color space.

    Args:
        color_space: Standard ColorSpace enum
        custom_primaries: Custom chromaticity coordinates (takes priority)

    Returns:
        Chromaticity primaries dictionary, or None if neither is specified.

    Example:
        >>> # Custom primaries take priority
        >>> custom = {'red': (0.7, 0.3), 'green': (0.2, 0.7), ...}
        >>> result = get_primaries_from_metadata(ColorSpace.BT709, custom)
        >>> result == custom
        True

        >>> # Falls back to standard color space
        >>> result = get_primaries_from_metadata(ColorSpace.BT709, None)
        >>> result == STANDARD_COLOR_PRIMARIES[ColorSpace.BT709]
        True
    """
    if custom_primaries:
        return custom_primaries

    return get_primaries_for_color_space(color_space)


def primaries_to_openexr_chromaticities(
    primaries: Dict[str, Tuple[float, float]]
) -> Tuple[Tuple[float, float], ...]:
    """
    Convert primaries dict to OpenEXR chromaticities tuple format.

    OpenEXR stores chromaticities as a flat tuple of 8 floats:
    (red_x, red_y, green_x, green_y, blue_x, blue_y, white_x, white_y)

    Args:
        primaries: Dictionary with 'red', 'green', 'blue', 'white' keys

    Returns:
        Tuple of 8 floats in OpenEXR order

    Example:
        >>> primaries = STANDARD_COLOR_PRIMARIES[ColorSpace.BT709]
        >>> chromaticities_to_openexr(primaries)
        (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)
    """
    return (
        primaries['red'][0], primaries['red'][1],
        primaries['green'][0], primaries['green'][1],
        primaries['blue'][0], primaries['blue'][1],
        primaries['white'][0], primaries['white'][1]
    )


def openexr_chromaticities_to_primaries(
    chromaticities: Tuple[float, ...]
) -> Dict[str, Tuple[float, float]]:
    """
    Convert OpenEXR chromaticities tuple to primaries dict.

    OpenEXR stores chromaticities as a flat tuple of 8 floats:
    (red_x, red_y, green_x, green_y, blue_x, blue_y, white_x, white_y)

    Args:
        chromaticities: Tuple of 8 floats from OpenEXR header

    Returns:
        Dictionary with 'red', 'green', 'blue', 'white' keys

    Example:
        >>> exr_chromaticities = (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)
        >>> primaries = openexr_chromaticities_to_primaries(exr_chromaticities)
        >>> primaries['red']
        (0.64, 0.33)
    """
    if len(chromaticities) != 8:
        raise ValueError(
            f"Expected 8 chromaticity values, got {len(chromaticities)}"
        )

    return {
        'red': (chromaticities[0], chromaticities[1]),
        'green': (chromaticities[2], chromaticities[3]),
        'blue': (chromaticities[4], chromaticities[5]),
        'white': (chromaticities[6], chromaticities[7])
    }
