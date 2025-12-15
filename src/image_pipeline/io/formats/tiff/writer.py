"""PNG format writer"""
import numpy as np
import imageio.v3 as iio
import png as pypng

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatWriter


class TiffFormatWriter(FormatWriter):
    """Writer for TIFF images"""

    def write(self, img_data: ImageData, options) -> None:
        """
        Write TIFF image

        TIFF doesn't have custom metadata implementation yet,
        so just write pixels.

        Args:
            img_data: ImageData with pixels and metadata
            options: Save options
        """
        # TODO: Implement TIFF metadata support
        pass

    def validate(self, img_data: ImageData) -> None:
        """
        Validate that data is compatible with TIFF format

        TIFF supports: uint8, uint16, uint32, float32, float64
        """
        pass