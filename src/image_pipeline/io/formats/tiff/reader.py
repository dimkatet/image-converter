"""TIFF format reader"""
import os

import tifffile

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatReader
from image_pipeline.types import ImageMetadata


class TiffFormatReader(FormatReader):
    """Reader for TIFF images"""
    
    def read(self) -> ImageData:
        """
        Read TIFF image
        
        Returns:
            ImageData with pixels and metadata
        """
        try:
            pixels = tifffile.imread(self.filepath)
            with tifffile.TiffFile(self.filepath) as tif:
                metadata: ImageMetadata = {
                    'format': 'TIFF',
                    'filename': self.filepath.name,
                    'file_size': os.path.getsize(self.filepath),
                }

                # TODO: Auto-detect paper_white from TIFF tags for scene-referred HDR
                # Scene-referred TIFF files encode linear values relative to a paper white
                # (typically 100 nits). Consider reading relevant EXIF/TIFF tags:
                # - WhitePoint tag (chromaticity)
                # - Photometric interpretation
                # - Color space metadata
                # - Custom tags for luminance reference
                # Then set metadata['paper_white'] = detected_value

            return ImageData(pixels, metadata)
            
        except Exception as e:
            raise IOError(f"Error reading TIFF file: {e}")