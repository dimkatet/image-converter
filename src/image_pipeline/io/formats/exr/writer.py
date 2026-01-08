"""OpenEXR format writer"""

import numpy as np
import OpenEXR

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatWriter
from image_pipeline.types import SaveOptions
from .options import EXRSaveOptionsAdapter
from .metadata import EXRMetadataAdapter


class EXRFormatWriter(FormatWriter):
    """Writer for OpenEXR images"""

    def __init__(self, filepath):
        super().__init__(filepath)

    def write(self, img_data: ImageData, options: SaveOptions) -> None:
        """
        Write OpenEXR image with metadata

        Args:
            img_data: ImageData with pixels and metadata
            options: Save options (compression, pixel_type)
        """
        # Validate image data first
        self.validate(img_data)

        # Ensure directory exists
        self.ensure_directory()

        # Adapt options
        exr_options = EXRSaveOptionsAdapter.adapt(options)

        # Prepare pixels
        pixels = self.prepare_pixels(img_data)

        # Write pixels with metadata
        self._write_pixels_with_metadata(pixels, img_data, exr_options)

    def validate(self, img_data: ImageData) -> None:
        """
        Validate image data for EXR format

        Args:
            img_data: Image data to validate

        Raises:
            ValueError: If image data is invalid for EXR format
        """
        # EXR supports float data (will convert if needed)
        if img_data.pixels.ndim != 3:
            raise ValueError(
                f"EXR writer expects 3D array (H, W, C), got shape {img_data.pixels.shape}"
            )

        channels = img_data.pixels.shape[2]
        if channels not in [3, 4]:
            raise ValueError(
                f"EXR writer supports 3 (RGB) or 4 (RGBA) channels, got {channels}"
            )

    def prepare_pixels(self, img_data: ImageData) -> np.ndarray:
        """
        Prepare pixels for EXR encoding

        Args:
            img_data: Image data

        Returns:
            Prepared pixel array (unchanged, EXR handles float directly)
        """
        # EXR natively supports float32/float16
        # Keep as float32 for now, conversion happens in write_pixels
        return img_data.pixels.astype(np.float32)

    def _write_pixels_with_metadata(
        self,
        pixels: np.ndarray,
        img_data: ImageData,
        exr_options: dict
    ) -> None:
        """
        Write pixels to EXR file with metadata

        Args:
            pixels: Prepared pixel data (H, W, C) float32
            img_data: Image data with metadata
            exr_options: Adapted EXR options (compression, pixel_type)
        """
        try:
            height, width, channels = pixels.shape
            has_alpha = channels == 4

            # Get options
            compression = exr_options.get('compression', OpenEXR.ZIP_COMPRESSION)
            pixel_type_str = exr_options.get('pixel_type', 'float16')

            # Build header dict for OpenEXR.File API
            header = {
                'compression': compression
            }

            # Add metadata to header
            metadata_dict = EXRMetadataAdapter.to_exr_header(img_data.metadata, channels)
            for key, value in metadata_dict.items():
                header[key] = value

            # Prepare channel data dict and convert to target dtype
            if has_alpha:
                channel_dict = {
                    'R': pixels[:, :, 0].astype(pixel_type_str),
                    'G': pixels[:, :, 1].astype(pixel_type_str),
                    'B': pixels[:, :, 2].astype(pixel_type_str),
                    'A': pixels[:, :, 3].astype(pixel_type_str)
                }
            else:
                channel_dict = {
                    'R': pixels[:, :, 0].astype(pixel_type_str),
                    'G': pixels[:, :, 1].astype(pixel_type_str),
                    'B': pixels[:, :, 2].astype(pixel_type_str)
                }

            # Write using OpenEXR.File API
            with OpenEXR.File(header, channel_dict) as outfile:
                outfile.write(str(self.filepath))

        except Exception as e:
            raise IOError(f"Error writing OpenEXR file: {e}")
