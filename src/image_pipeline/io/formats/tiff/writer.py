"""PNG format writer"""
import numpy as np
import imageio.v3 as iio
import png as pypng

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatWriter


class TiffFormatWriter(FormatWriter):
    """Writer for TIFF images"""
    
    def validate(self, img_data: ImageData) -> None:
        """
        Validate that data is compatible with TIFF format
        
        TIFF supports: uint8, uint16, uint32, float32, float64
        """
        pass
    
    def write_pixels(self, img_data: ImageData, compression_level: int = 6, **options) -> None:
        """
        Write PNG pixel data
        
        Args:
            img_data: ImageData with pixels
            compression_level: Compression level (0-9), default 6
            **options: Additional options (ignored)
        """
        pass