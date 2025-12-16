"""Matrix computation for RGB ↔ XYZ color space conversions.

This module computes transformation matrices from CIE 1931 xy chromaticity
coordinates of color primaries. All conversions assume D65 white point.

References:
    - ITU-R BT.709 (sRGB primaries)
    - ITU-R BT.2020 (UHDTV primaries)
    - Display P3 (DCI-P3 with D65)
"""

import numpy as np
from typing import Dict, Tuple

from image_pipeline.types import ColorSpace
from image_pipeline.constants import STANDARD_COLOR_PRIMARIES


def compute_rgb_to_xyz_matrix(
    primaries: Dict[str, Tuple[float, float]]
) -> np.ndarray:
    """Compute 3x3 matrix for RGB → XYZ transformation.

    Args:
        primaries: Dictionary with keys 'red', 'green', 'blue', 'white',
                   each containing (x, y) chromaticity coordinates.

    Returns:
        3x3 numpy array for linear RGB → XYZ conversion.

    Algorithm:
        1. Build matrix M from primary x,y,z coordinates (z = 1 - x - y)
        2. Compute white point XYZ (assuming Y=1)
        3. Solve for scaling factors: S = M^-1 @ W
        4. Final matrix: M @ diag(S)

    Example:
        >>> primaries = STANDARD_COLOR_PRIMARIES[ColorSpace.BT709]
        >>> matrix = compute_rgb_to_xyz_matrix(primaries)
        >>> matrix.shape
        (3, 3)
    """
    # Extract chromaticity coordinates
    xr, yr = primaries['red']
    xg, yg = primaries['green']
    xb, yb = primaries['blue']
    xw, yw = primaries['white']

    # Convert xy → XYZ (assuming Y=1 for primaries)
    # X = x/y, Y = 1, Z = (1-x-y)/y
    Xr, Yr, Zr = xr / yr, 1.0, (1 - xr - yr) / yr
    Xg, Yg, Zg = xg / yg, 1.0, (1 - xg - yg) / yg
    Xb, Yb, Zb = xb / yb, 1.0, (1 - xb - yb) / yb

    # Build primary matrix (columns are primaries)
    M = np.array([
        [Xr, Xg, Xb],
        [Yr, Yg, Yb],
        [Zr, Zg, Zb]
    ])

    # White point XYZ (also assuming Y=1)
    Xw, Yw, Zw = xw / yw, 1.0, (1 - xw - yw) / yw
    W = np.array([Xw, Yw, Zw])

    # Solve for scaling factors: S = M^-1 @ W
    S = np.linalg.inv(M) @ W

    # Final RGB→XYZ matrix: M @ diag(S)
    rgb_to_xyz = M @ np.diag(S)

    return rgb_to_xyz


def compute_xyz_to_rgb_matrix(
    primaries: Dict[str, Tuple[float, float]]
) -> np.ndarray:
    """Compute 3x3 matrix for XYZ → RGB transformation.

    Args:
        primaries: Dictionary with keys 'red', 'green', 'blue', 'white',
                   each containing (x, y) chromaticity coordinates.

    Returns:
        3x3 numpy array for XYZ → linear RGB conversion.

    Example:
        >>> primaries = STANDARD_COLOR_PRIMARIES[ColorSpace.BT2020]
        >>> matrix = compute_xyz_to_rgb_matrix(primaries)
        >>> # Should be inverse of rgb_to_xyz
        >>> rgb_to_xyz = compute_rgb_to_xyz_matrix(primaries)
        >>> np.allclose(matrix @ rgb_to_xyz, np.eye(3))
        True
    """
    rgb_to_xyz = compute_rgb_to_xyz_matrix(primaries)
    xyz_to_rgb = np.linalg.inv(rgb_to_xyz)
    return xyz_to_rgb


# Precompute all standard matrices for performance
_PRECOMPUTED_RGB_TO_XYZ: Dict[ColorSpace, np.ndarray] = {
    cs: compute_rgb_to_xyz_matrix(STANDARD_COLOR_PRIMARIES[cs])
    for cs in ColorSpace
}

_PRECOMPUTED_XYZ_TO_RGB: Dict[ColorSpace, np.ndarray] = {
    cs: compute_xyz_to_rgb_matrix(STANDARD_COLOR_PRIMARIES[cs])
    for cs in ColorSpace
}


def get_precomputed_matrix(
    color_space: ColorSpace,
    direction: str = 'rgb_to_xyz'
) -> np.ndarray:
    """Get precomputed transformation matrix for standard color space.

    Args:
        color_space: One of ColorSpace.BT709, BT2020, DISPLAY_P3.
        direction: Either 'rgb_to_xyz' or 'xyz_to_rgb'.

    Returns:
        3x3 transformation matrix.

    Raises:
        ValueError: If direction is invalid.

    Example:
        >>> matrix = get_precomputed_matrix(ColorSpace.BT709, 'rgb_to_xyz')
        >>> matrix.shape
        (3, 3)
    """
    if direction == 'rgb_to_xyz':
        return _PRECOMPUTED_RGB_TO_XYZ[color_space].copy()
    elif direction == 'xyz_to_rgb':
        return _PRECOMPUTED_XYZ_TO_RGB[color_space].copy()
    else:
        raise ValueError(
            f"Invalid direction '{direction}'. "
            f"Must be 'rgb_to_xyz' or 'xyz_to_rgb'."
        )
