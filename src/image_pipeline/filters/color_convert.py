"""Color space conversion filter.

Converts linear RGB pixels between standard color spaces (BT.709, BT.2020, Display P3)
via XYZ intermediate representation.
"""

import numpy as np
import warnings

from image_pipeline.core.image_data import ImageData
from image_pipeline.types import ColorSpace, TransferFunction
from image_pipeline.constants import STANDARD_COLOR_PRIMARIES
from image_pipeline.color.conversion import convert_color_space
from .base import ImageFilter


class ColorConvertFilter(ImageFilter):
    """Convert image pixels between RGB color spaces.

    Transforms linear RGB values from source to target color space using
    CIE XYZ as intermediate representation. Requires linear (non-gamma-encoded)
    pixel values.

    Attributes:
        source: Source color space (BT.709, BT.2020, or Display P3).
        target: Target color space (BT.709, BT.2020, or Display P3).

    Example:
        >>> # Convert BT.709 sRGB to BT.2020 (for HDR)
        >>> filter = ColorConvertFilter(source='BT.709', target='BT.2020')
        >>> pixels_bt2020 = filter.apply(pixels_bt709)

        >>> # Typical HDR workflow
        >>> pipeline = FilterPipeline([
        ...     PQDecodeFilter(),              # PQ → linear
        ...     ColorConvertFilter('BT.2020', 'BT.709'),  # Wide → sRGB gamut
        ...     PQEncodeFilter()               # linear → PQ
        ... ])

    Note:
        TODO: Remove explicit 'source' parameter when all format readers
        support color space detection from file metadata (cICP chunks, TIFF tags).
        Then source can be read from: img_data.metadata.get('color_space').
        Tracking issue: Readers need to parse:
          - PNG: cICP chunk
          - TIFF: ColorSpace/Photometric tags
          - WebP: VP8L bitstream color space
          - AVIF: CICP properties (blocked by imagecodecs library limitation)
    """

    def __init__(
        self,
        source: ColorSpace,
        target: ColorSpace
    ):
        """Initialize color space converter.

        Args:
            source: Source RGB color space (ColorSpace enum).
                    Must be ColorSpace.BT709, BT2020, or DISPLAY_P3.
            target: Target RGB color space (ColorSpace enum).
                    Must be ColorSpace.BT709, BT2020, or DISPLAY_P3.

        Raises:
            TypeError: If source or target is not a ColorSpace enum.
        """
        if not isinstance(source, ColorSpace):
            raise TypeError(
                f"source must be ColorSpace enum, got {type(source).__name__}. "
                f"Use ColorSpace.BT709, ColorSpace.BT2020, or ColorSpace.DISPLAY_P3."
            )

        if not isinstance(target, ColorSpace):
            raise TypeError(
                f"target must be ColorSpace enum, got {type(target).__name__}. "
                f"Use ColorSpace.BT709, ColorSpace.BT2020, or ColorSpace.DISPLAY_P3."
            )

        self.source = source
        self.target = target
        super().__init__()

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        """Apply color space conversion to pixels.

        Args:
            pixels: Linear RGB image array, shape (H, W, 3) or (H, W, 4).
                    Must be float16/float32/float64 dtype (linear, no gamma).

        Returns:
            Converted pixels with same shape and dtype.

        Raises:
            ValueError: If pixels are not float or wrong shape.
        """
        self.validate(pixels)

        # Validate dtype (must be float for linear RGB)
        self._check_dtype(pixels, [np.float16, np.float32, np.float64])

        # No-op if source == target
        if self.source == self.target:
            warnings.warn(
                f"Source and target color space are identical ({self.source.value}). "
                f"Returning pixels unchanged.",
                UserWarning
            )
            return pixels

        # Apply conversion
        converted = convert_color_space(pixels, self.source, self.target)

        return converted

    def validate_params(self) -> None:
        """Validate filter parameters.

        Raises:
            TypeError: If source/target are not ColorSpace instances.
        """
        if not isinstance(self.source, ColorSpace):
            raise TypeError(
                f"source must be ColorSpace enum, got {type(self.source).__name__}"
            )

        if not isinstance(self.target, ColorSpace):
            raise TypeError(
                f"target must be ColorSpace enum, got {type(self.target).__name__}"
            )

    def update_metadata(self, img_data: ImageData) -> None:
        """Update image metadata after color space conversion.

        Sets target color space and primaries in metadata.
        Issues warning if existing metadata color_space differs from filter source.

        Args:
            img_data: Image data object to update.
        """
        super().update_metadata(img_data)

        # Warn if metadata conflicts with filter source parameter
        existing_color_space = img_data.metadata.get('color_space')
        if existing_color_space is not None and existing_color_space != self.source:
            warnings.warn(
                f"Metadata color_space={existing_color_space.value}, "
                f"but filter source={self.source.value}. "
                f"Using filter parameter. "
                f"This warning will be removed once all format readers "
                f"support color space detection from files.",
                UserWarning
            )

        # Validate transfer function (must be linear)
        transfer_fn = img_data.metadata.get('transfer_function')
        if transfer_fn is not None and transfer_fn != TransferFunction.LINEAR:
            # Handle both enum and string values
            tf_display = transfer_fn.value if isinstance(transfer_fn, TransferFunction) else transfer_fn
            raise ValueError(
                f"Color space conversion requires linear RGB values. "
                f"Current transfer_function={tf_display}. "
                f"Apply pq_decode or hlg_decode filter first to linearize, "
                f"then apply color_convert, then re-encode if needed."
            )

        # Update to target color space
        img_data.metadata['color_space'] = self.target
        img_data.metadata['color_primaries'] = STANDARD_COLOR_PRIMARIES[self.target]

    def __repr__(self) -> str:
        return (
            f"ColorConvertFilter(source={self.source.value}, "
            f"target={self.target.value})"
        )
