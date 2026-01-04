"""TIFF format writer"""
from typing import Dict, Tuple, Optional, Any
import numpy as np
import tifffile

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatWriter
from image_pipeline.io.formats.tiff.options import TiffOptionsAdapter, TiffSaveOptions
from image_pipeline.types import ColorSpace, ImageMetadata, SaveOptions
from image_pipeline.constants import STANDARD_COLOR_PRIMARIES


class TiffFormatWriter(FormatWriter):
    """Writer for TIFF images using tifffile library"""

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.options_adapter = TiffOptionsAdapter()

    def write(self, img_data: ImageData, options: SaveOptions) -> None:
        """
        Write TIFF image with metadata

        Args:
            img_data: ImageData with pixels and metadata
            options: Save options

        Raises:
            ValueError: If validation fails
            IOError: If writing fails
        """
        # Validate image data
        self.validate(img_data)

        # Validate options
        validated_options = self.options_adapter.validate(options)

        # Prepare tifffile kwargs
        tiff_kwargs = self._prepare_tiff_kwargs(img_data, validated_options)

        # Write TIFF file
        self._write_pixels(img_data.pixels, tiff_kwargs)

    def validate(self, img_data: ImageData) -> None:
        """
        Validate that data is compatible with TIFF format

        TIFF supports:
        - Pixel dtypes: uint8, uint16, uint32, float32, float64
        - Channels: 1 (grayscale), 3 (RGB), 4 (RGBA)

        Args:
            img_data: ImageData to validate

        Raises:
            ValueError: If validation fails
        """
        pixels = img_data.pixels

        if not isinstance(pixels, np.ndarray):
            raise ValueError("Data must be a numpy array")

        if pixels.size == 0:
            raise ValueError("Empty pixel array")

        # Check dtype
        supported_dtypes = (np.uint8, np.uint16, np.uint32, np.float32, np.float64)
        if pixels.dtype not in supported_dtypes:
            raise ValueError(
                f"TIFF supports uint8, uint16, uint32, float32, float64 dtypes. "
                f"Got: {pixels.dtype}.\n"
                f"Solutions:\n"
                f"  1. For float64: convert to float32\n"
                f"  2. For int types: convert to appropriate uint type"
            )

        # Validate shape
        if pixels.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array, got shape {pixels.shape}")

        if pixels.ndim == 3 and pixels.shape[2] not in (1, 3, 4):
            raise ValueError(
                f"Expected 1, 3, or 4 channels, got {pixels.shape[2]}"
            )

    def _prepare_tiff_kwargs(
        self,
        img_data: ImageData,
        options: TiffSaveOptions
    ) -> Dict[str, Any]:
        """
        Prepare kwargs for tifffile.imwrite()

        Args:
            img_data: ImageData with pixels and metadata
            options: Validated TIFF save options

        Returns:
            Dictionary of kwargs for tifffile.imwrite()
        """
        kwargs: Dict[str, Any] = {}
        metadata = img_data.metadata

        # === Compression (Tag 259) ===
        compression = options.get('compression', 'none')
        if compression == 'none':
            kwargs['compression'] = None
        elif compression == 'lzw':
            kwargs['compression'] = 'lzw'
        elif compression == 'deflate':
            kwargs['compression'] = 'deflate'
            # Add compression level if specified
            if 'compression_level' in options:
                kwargs['compressionargs'] = {'level': options['compression_level']}
        elif compression == 'zstd':
            kwargs['compression'] = 'zstd'
            # Add compression level if specified
            if 'compression_level' in options:
                kwargs['compressionargs'] = {'level': options['compression_level']}
        elif compression == 'jpeg':
            kwargs['compression'] = 'jpeg'
            # JPEG quality (1-100) - tifffile passes this to imagecodecs.jpeg_encode
            # which expects 'level' parameter (0-100, higher = better quality)
            if 'quality' in options:
                kwargs['compressionargs'] = {'level': options['quality']}

        # === Photometric interpretation ===
        # Explicitly specify photometric to avoid deprecation warnings
        # 2D or (H, W, 1) -> MINISBLACK (grayscale)
        # (H, W, 3) -> RGB
        # (H, W, 4) -> RGB (with ExtraSamples tag for alpha)
        if img_data.pixels.ndim == 2 or (img_data.pixels.ndim == 3 and img_data.pixels.shape[2] == 1):
            kwargs['photometric'] = 'minisblack'
        else:
            kwargs['photometric'] = 'rgb'

        # === Color space metadata (Tags 318, 319) ===
        # WhitePoint (Tag 318) and PrimaryChromaticities (Tag 319)
        extratags = []

        # Get color primaries from metadata
        primaries = self._get_color_primaries(metadata)
        if primaries:
            # WhitePoint tag (318): (x_white, y_white) as RATIONAL pairs
            white = primaries['white']
            white_tag = self._make_chromaticity_tag(
                318,  # WhitePoint
                [white[0], white[1]]
            )
            extratags.append(white_tag)

            # PrimaryChromaticities tag (319): (x_R, y_R, x_G, y_G, x_B, y_B)
            primaries_tag = self._make_chromaticity_tag(
                319,  # PrimaryChromaticities
                [
                    primaries['red'][0], primaries['red'][1],
                    primaries['green'][0], primaries['green'][1],
                    primaries['blue'][0], primaries['blue'][1]
                ]
            )
            extratags.append(primaries_tag)

        if extratags:
            kwargs['extratags'] = extratags

        # === Metadata (ImageDescription tag 270) ===
        # Store HDR metadata and logical bit_depth in ImageDescription
        # This is a common convention for custom metadata
        description_parts = []

        # Logical bit depth (for 10-bit, 12-bit stored in uint16)
        if 'bit_depth' in metadata:
            bit_depth = metadata['bit_depth']
            # Only store if different from physical dtype bit depth
            dtype_bits = img_data.pixels.dtype.itemsize * 8
            if bit_depth != dtype_bits:
                description_parts.append(f"bit_depth={bit_depth}")

        if 'paper_white' in metadata:
            description_parts.append(f"paper_white={metadata['paper_white']}")

        if 'mastering_display_max_luminance' in metadata:
            description_parts.append(
                f"mastering_display_max_luminance={metadata['mastering_display_max_luminance']}"
            )

        if 'mastering_display_min_luminance' in metadata:
            description_parts.append(
                f"mastering_display_min_luminance={metadata['mastering_display_min_luminance']}"
            )

        if description_parts:
            kwargs['description'] = '; '.join(description_parts)

        return kwargs

    def _get_color_primaries(
        self,
        metadata: ImageMetadata
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Get color primaries from metadata

        Tries in order:
        1. Custom primaries from metadata['color_primaries']
        2. Standard primaries from metadata['color_space']

        Args:
            metadata: ImageMetadata dictionary

        Returns:
            Dictionary with 'red', 'green', 'blue', 'white' chromaticity coordinates
            or None if not available
        """
        # First check for custom primaries
        if 'color_primaries' in metadata:
            return metadata['color_primaries']

        # Then check for standard color space
        if 'color_space' in metadata:
            color_space = metadata['color_space']
            if isinstance(color_space, ColorSpace):
                return STANDARD_COLOR_PRIMARIES.get(color_space)

        return None

    def _make_chromaticity_tag(
        self,
        tag_code: int,
        values: list[float]
    ) -> tuple:
        """
        Create TIFF chromaticity tag (WhitePoint or PrimaryChromaticities)

        TIFF stores chromaticity as RATIONAL (numerator/denominator pairs).
        We use denominator = 1000000 for 6 decimal places precision.

        Args:
            tag_code: TIFF tag code (318 for WhitePoint, 319 for PrimaryChromaticities)
            values: List of float chromaticity values

        Returns:
            Tuple for tifffile extratags: (code, dtype, count, value, writeonce)
        """
        # Convert float to RATIONAL with 6 decimal places
        # RATIONAL = 5 in TIFF data types
        denominator = 1000000

        # Each float becomes a pair (numerator, denominator)
        rational_pairs = []
        for val in values:
            numerator = int(round(val * denominator))
            rational_pairs.extend([numerator, denominator])

        return (
            tag_code,           # Tag code
            5,                  # RATIONAL type (5)
            len(values),        # Count (number of RATIONAL values, not pairs)
            rational_pairs,     # Value as list of [num1, den1, num2, den2, ...]
            True                # writeonce
        )

    def _write_pixels(self, pixels: np.ndarray, kwargs: Dict[str, Any]) -> None:
        """
        Write TIFF pixel data using tifffile

        Args:
            pixels: Pixel array (uint8/16/32 or float32/64)
            kwargs: Keyword arguments for tifffile.imwrite()

        Raises:
            IOError: If writing fails
        """
        try:
            # tifffile automatically handles:
            # - SampleFormat (Tag 339): 1 for uint, 3 for float
            # - BitsPerSample (Tag 258): 8/16/32 based on dtype
            # - SamplesPerPixel (Tag 277): determined from shape
            # - Photometric (Tag 262): MINISBLACK (1) or RGB (2)
            tifffile.imwrite(
                str(self.filepath),
                pixels,
                **kwargs
            )

        except Exception as e:
            raise IOError(f"Error writing TIFF: {e}")