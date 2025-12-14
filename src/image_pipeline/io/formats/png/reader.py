"""PNG format reader"""
import imageio.v3 as iio
import os

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatReader
from image_pipeline.types import ImageMetadata


class PNGFormatReader(FormatReader):
    """Reader for PNG images"""
    
    def read(self) -> ImageData:
        """
        Read PNG image
        
        Returns:
            ImageData with pixels and metadata
        """
        try:
            # Read pixels
            pixels = iio.imread(self.filepath)
            
            # Minimal metadata - only typed fields
            metadata: ImageMetadata = {
                'format': 'PNG',
                'filename': self.filepath.name,
                'file_size': os.path.getsize(self.filepath),
            }
            
            # ImageData will automatically fill: shape, dtype, channels, bit_depth
            return ImageData(pixels, metadata)
            
        except Exception as e:
            raise IOError(f"Error reading PNG: {e}")