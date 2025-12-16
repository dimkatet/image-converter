"""Color space conversion utilities.

This module provides tools for converting between different RGB color spaces
(BT.709, BT.2020, Display P3) via CIE XYZ intermediate representation.
"""

from image_pipeline.color.conversion import (
    convert_color_space,
    get_conversion_matrix,
)
from image_pipeline.color.matrices import (
    compute_rgb_to_xyz_matrix,
    compute_xyz_to_rgb_matrix,
    get_precomputed_matrix,
)

__all__ = [
    'convert_color_space',
    'get_conversion_matrix',
    'compute_rgb_to_xyz_matrix',
    'compute_xyz_to_rgb_matrix',
    'get_precomputed_matrix',
]
