"""JPEG XR format writer using imagecodecs"""
from typing import Any
import numpy as np
from imagecodecs import jpegxr_encode

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatWriter
from image_pipeline.io.formats.jxr.options import JXRSaveOptionsAdapter, JXRSaveOptions
from image_pipeline.types import SaveOptions


class JXRFormatWriter(FormatWriter):
    """Writer for JPEG XR images using imagecodecs"""

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.options_adapter = JXRSaveOptionsAdapter()

    def write(self, img_data: ImageData, options: SaveOptions) -> None:
        """
        Write JPEG XR image

        Note: JPEG XR in imagecodecs doesn't support extended metadata
        (color primaries, transfer function, EXIF). For HDR metadata
        preservation, consider using PNG, AVIF, or OpenEXR formats.

        Args:
            img_data: ImageData with pixels
            options: Save options

        Raises:
            ValueError: If pixel data has unsupported dtype
            IOError: If writing fails
        """
        # Validate data first
        self.validate(img_data)

        # Validate options
        validated_options = self.options_adapter.validate(options)

        # Write pixels
        self._write_pixels(img_data.pixels, validated_options)

    def validate(self, img_data: ImageData) -> None:
        """
        Validate that data is compatible with JPEG XR format

        JPEG XR supports: uint8, uint16, float16, float32

        Args:
            img_data: ImageData to validate

        Raises:
            ValueError: If data is not compatible
        """
        pixels = img_data.pixels

        if not isinstance(pixels, np.ndarray):
            raise ValueError("Data must be a numpy array")

        if pixels.size == 0:
            raise ValueError("Empty pixel array")

        # Check dtype - JXR supports uint8, uint16, float16, float32
        supported_dtypes = (np.uint8, np.uint16, np.float16, np.float32)
        if pixels.dtype not in supported_dtypes:
            raise ValueError(
                f"JPEG XR format supports uint8, uint16, float16, float32. "
                f"Got {pixels.dtype}. "
                f"Use QuantizeFilter or appropriate conversion before saving."
            )

        # Validate shape
        if pixels.ndim not in (2, 3):
            raise ValueError(
                f"JPEG XR requires 2D or 3D array, got {pixels.ndim}D"
            )

        if pixels.ndim == 3:
            channels = pixels.shape[2]
            if channels not in (1, 3, 4):
                raise ValueError(
                    f"JPEG XR supports 1 (grayscale), 3 (RGB), or 4 (RGBA) channels. "
                    f"Got {channels} channels."
                )

    def _write_pixels(self, pixels: np.ndarray, options: JXRSaveOptions) -> None:
        """
        Encode and write JPEG XR pixel data using imagecodecs

        imagecodecs.jpegxr_encode parameters:
        - level: quality level (None or 1-100)
            None or 100 = lossless for uint8/uint16
            < 100 = lossy compression
            Note: float data always has minimal precision loss (~3e-5)
        - photometric: color model (None = auto-detect)
        - hasalpha: has alpha channel (None = auto-detect)
        - resolution: (horizontal_dpi, vertical_dpi) tuple

        Args:
            pixels: Pixel array (uint8, uint16, float16, or float32)
            options: Validated JXR save options

        Raises:
            IOError: If encoding or writing fails
        """
        try:
            # Extract options
            lossless = options.get('lossless', True)
            quality = options.get('quality', 90)
            photometric = options.get('photometric')
            resolution = options.get('resolution')

            # Determine compression level
            # For lossless: use level=100 (or None)
            # For lossy: use specified quality
            if lossless:
                level = 100
            else:
                level = quality

            # Build encode parameters
            encode_params: dict[str, Any] = {
                'level': level,
            }

            if photometric is not None:
                encode_params['photometric'] = photometric

            if resolution is not None:
                encode_params['resolution'] = resolution

            # Auto-detect alpha channel if not specified
            # (imagecodecs will auto-detect if hasalpha is not provided)

            # Encode to JPEG XR bytes
            jxr_bytes = jpegxr_encode(pixels, **encode_params)

            # Ensure directory exists
            self.ensure_directory()

            # Write to file
            with open(self.filepath, 'wb') as f:
                f.write(jxr_bytes)

        except Exception as e:
            raise IOError(f"Error writing JPEG XR: {e}")
