"""High-level color space conversion functions.

Provides user-facing API for converting image pixels between RGB color spaces
via XYZ intermediate representation.
"""

import numpy as np

from image_pipeline.types import ColorSpace
from image_pipeline.color.matrices import get_precomputed_matrix


def get_conversion_matrix(
    source: ColorSpace,
    target: ColorSpace
) -> np.ndarray:
    """Get 3x3 matrix for direct source RGB → target RGB conversion.

    Args:
        source: Source color space (e.g., ColorSpace.BT709).
        target: Target color space (e.g., ColorSpace.BT2020).

    Returns:
        3x3 numpy array for direct RGB → RGB conversion.
        Computed as: RGB_src → XYZ → RGB_target.

    Example:
        >>> matrix = get_conversion_matrix(ColorSpace.BT709, ColorSpace.BT2020)
        >>> matrix.shape
        (3, 3)
        >>> # Apply to pixel: rgb_bt2020 = matrix @ rgb_bt709
    """
    # Get transformation matrices
    src_to_xyz = get_precomputed_matrix(source, 'rgb_to_xyz')
    xyz_to_tgt = get_precomputed_matrix(target, 'xyz_to_rgb')

    # Compose: RGB_src → XYZ → RGB_target
    conversion_matrix = xyz_to_tgt @ src_to_xyz

    return conversion_matrix


def convert_color_space(
    pixels: np.ndarray,
    source: ColorSpace,
    target: ColorSpace
) -> np.ndarray:
    """Convert image pixels from source to target color space.

    Applies matrix transformation: pixels_target = M @ pixels_source.
    Operates on linear RGB values (assumes no transfer function applied).

    Args:
        pixels: Linear RGB image array of shape (H, W, 3) or (H, W, 4).
                Must be float32 or float64 dtype.
        source: Source color space.
        target: Target color space.

    Returns:
        Converted image array with same shape and dtype as input.
        RGB channels transformed, alpha channel (if present) preserved.

    Raises:
        ValueError: If pixels shape is invalid or dtype is not float.

    Example:
        >>> # Convert BT.709 → BT.2020
        >>> pixels_bt709 = np.random.rand(1080, 1920, 3).astype(np.float32)
        >>> pixels_bt2020 = convert_color_space(
        ...     pixels_bt709, ColorSpace.BT709, ColorSpace.BT2020
        ... )
        >>> pixels_bt2020.shape
        (1080, 1920, 3)

    Note:
        Out-of-gamut values are NOT clipped. This preserves HDR values > 1.0.
        If clipping is needed, apply separately after conversion.
    """
    # Validate input
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError(
            f"Expected shape (H, W, 3) or (H, W, 4), got {pixels.shape}"
        )

    if not np.issubdtype(pixels.dtype, np.floating):
        raise ValueError(
            f"Expected float dtype for linear RGB, got {pixels.dtype}. "
            f"Apply dequantize filter first if working with integer images."
        )

    # Get conversion matrix
    matrix = get_conversion_matrix(source, target)

    # Separate RGB and alpha (if present)
    has_alpha = pixels.shape[2] == 4
    if has_alpha:
        rgb = pixels[..., :3]
        alpha = pixels[..., 3:4]
    else:
        rgb = pixels

    # Apply matrix transformation
    # Reshape (H, W, 3) → (H*W, 3) for matrix multiplication
    h, w = rgb.shape[:2]
    rgb_flat = rgb.reshape(-1, 3)

    # Transform: (N, 3) @ (3, 3).T = (N, 3)
    rgb_converted = (matrix @ rgb_flat.T).T

    # Reshape back to (H, W, 3)
    rgb_converted = rgb_converted.reshape(h, w, 3)

    # Preserve dtype
    rgb_converted = rgb_converted.astype(pixels.dtype)

    # Recombine with alpha if present
    if has_alpha:
        result = np.concatenate([rgb_converted, alpha], axis=2)
    else:
        result = rgb_converted

    return result
