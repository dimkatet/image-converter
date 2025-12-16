"""
Color space string alias registry for CLI parsing.

Maps common color space string representations (case-insensitive) to ColorSpace enums.
"""
from typing import Dict

from image_pipeline.types import ColorSpace


COLOR_SPACE_ALIASES: Dict[str, ColorSpace] = {
    # BT.709 / sRGB variants
    'bt709': ColorSpace.BT709,
    'bt.709': ColorSpace.BT709,
    'rec709': ColorSpace.BT709,
    'rec.709': ColorSpace.BT709,
    'srgb': ColorSpace.BT709,

    # BT.2020 / Rec.2020 variants
    'bt2020': ColorSpace.BT2020,
    'bt.2020': ColorSpace.BT2020,
    'rec2020': ColorSpace.BT2020,
    'rec.2020': ColorSpace.BT2020,

    # Display P3 variants
    'displayp3': ColorSpace.DISPLAY_P3,
    'display-p3': ColorSpace.DISPLAY_P3,
    'display_p3': ColorSpace.DISPLAY_P3,
    'p3': ColorSpace.DISPLAY_P3,
    'dcip3': ColorSpace.DISPLAY_P3,
    'dci-p3': ColorSpace.DISPLAY_P3,
}


def get_available_color_spaces() -> str:
    """
    Get formatted string of available color space aliases.

    Returns:
        String with color space aliases and their enum mappings.

    Example:
        >>> print(get_available_color_spaces())
        Available color space aliases:
          • bt.709            -> ColorSpace.BT709
          • bt2020            -> ColorSpace.BT2020
          • displayp3         -> ColorSpace.DISPLAY_P3
          ...
    """
    lines = ["Available color space aliases:"]

    # Group by ColorSpace
    by_color_space: Dict[ColorSpace, list[str]] = {}
    for alias, cs in COLOR_SPACE_ALIASES.items():
        if cs not in by_color_space:
            by_color_space[cs] = []
        by_color_space[cs].append(alias)

    # Format output
    for cs in sorted(by_color_space.keys(), key=lambda x: x.value):
        aliases = sorted(by_color_space[cs])
        primary_alias = aliases[0]
        other_aliases = ', '.join(aliases[1:]) if len(aliases) > 1 else '(no aliases)'
        lines.append(f"  • {primary_alias:20} -> {cs.name} ({other_aliases})")

    return "\n".join(lines)
