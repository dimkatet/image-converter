"""WebP format writer"""
import numpy as np
from PIL import Image

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatWriter
from image_pipeline.io.formats.webp.options import WebPOptionsAdapter, WebPSaveOptions
from image_pipeline.types import SaveOptions


class WebPFormatWriter(FormatWriter):
    """Writer for WebP images"""

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.options_adapter = WebPOptionsAdapter()

    def write(self, img_data: ImageData, options: SaveOptions) -> None:
        """
        Write WebP image with metadata

        Args:
            img_data: ImageData with pixels and metadata
            options: Save options

        Raises:
            ValueError: If pixel data is not uint8
            IOError: If writing fails
        """
        # Validate data first
        self.validate(img_data)

        # Validate options
        validated_options = self.options_adapter.validate(options)

        # Ensure output directory exists
        self.ensure_directory()

        # Write pixels with metadata
        self._write_image(img_data, validated_options)

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

    def _write_image(self, img_data: ImageData, options: WebPSaveOptions) -> None:
        """
        Write WebP image with pixels and metadata

        Args:
            img_data: ImageData with pixels and metadata
            options: Validated WebP save options
        """
        try:
            pixels = img_data.pixels

            # Convert numpy array to PIL Image
            # Handle grayscale (H, W) by converting to (H, W, 1)
            if pixels.ndim == 2:
                img = Image.fromarray(pixels, mode='L')
            elif pixels.shape[2] == 1:
                img = Image.fromarray(pixels.squeeze(-1), mode='L')
            elif pixels.shape[2] == 3:
                img = Image.fromarray(pixels, mode='RGB')
            elif pixels.shape[2] == 4:
                img = Image.fromarray(pixels, mode='RGBA')
            else:
                raise ValueError(f"Unsupported channel count: {pixels.shape[2]}")

            # Prepare save kwargs
            save_kwargs = {
                'quality': options.get('quality'),
                'method': options.get('method'),
                'lossless': options.get('lossless'),
                'exact': options.get('exact'),
            }

            # Add EXIF if present in metadata
            if 'exif' in img_data.metadata:
                save_kwargs['exif'] = img_data.metadata['exif']

            # Add ICC profile if present in metadata
            if 'icc_profile' in img_data.metadata:
                save_kwargs['icc_profile'] = img_data.metadata['icc_profile']

            # Save to file
            img.save(self.filepath, format='WebP', **save_kwargs)

        except Exception as e:
            raise IOError(f"Error writing WebP: {e}")
