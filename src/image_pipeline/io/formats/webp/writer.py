"""WebP format writer using imagecodecs"""
import numpy as np
from imagecodecs import webp_encode

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatWriter
from image_pipeline.io.formats.webp.options import WebPOptionsAdapter, WebPSaveOptions
from image_pipeline.types import SaveOptions


class WebPFormatWriter(FormatWriter):
    """Writer for WebP images using imagecodecs"""

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.options_adapter = WebPOptionsAdapter()

    def write(self, img_data: ImageData, options: SaveOptions) -> None:
        """
        Write WebP image

        Note: WebP in imagecodecs doesn't support metadata (EXIF/XMP).
        For metadata preservation, consider using PNG or AVIF formats.

        Args:
            img_data: ImageData with pixels
            options: Save options

        Raises:
            ValueError: If pixel data is not uint8
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
        Validate that data is compatible with WebP format

        WebP only supports uint8 pixel data.

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

        # Strict validation: only uint8 allowed
        if pixels.dtype != np.uint8:
            raise ValueError(
                f"WebP format only supports uint8 data. Got {pixels.dtype}. "
                f"Use QuantizeFilter(bit_depth=8) before saving to WebP."
            )

        # Validate shape
        if pixels.ndim not in (2, 3):
            raise ValueError(
                f"WebP requires 2D or 3D array, got {pixels.ndim}D"
            )

        if pixels.ndim == 3:
            channels = pixels.shape[2]
            if channels not in (1, 3, 4):
                raise ValueError(
                    f"WebP supports 1 (grayscale), 3 (RGB), or 4 (RGBA) channels. "
                    f"Got {channels} channels."
                )

    def _write_pixels(self, pixels: np.ndarray, options: WebPSaveOptions) -> None:
        """
        Encode and write WebP pixel data using imagecodecs

        imagecodecs.webp_encode parameters:
        - level: quality/compression level (0-100)
        - lossless: bool (True for lossless, False for lossy)
        - method: compression effort (0-6)
        - numthreads: number of threads

        Args:
            pixels: Pixel array (uint8 only)
            options: Validated WebP save options

        Raises:
            IOError: If encoding or writing fails
        """
        try:
            # Extract options
            quality = options.get('quality', 80)
            lossless = options.get('lossless', False)
            method = options.get('method', 4)
            numthreads = options.get('numthreads')

            # Build encode parameters
            encode_params = {
                'level': quality,
                'lossless': lossless,
                'method': method,
            }

            if numthreads is not None:
                encode_params['numthreads'] = numthreads

            # Encode to WebP bytes
            webp_bytes = webp_encode(pixels, **encode_params)

            # Write to file
            with open(self.filepath, 'wb') as f:
                f.write(webp_bytes)

        except Exception as e:
            raise IOError(f"Error writing WebP: {e}")
