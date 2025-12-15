"""WebP format reader"""
import os
from pathlib import Path
from PIL import Image
import numpy as np

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatReader
from image_pipeline.types import ImageMetadata


class WebPFormatReader(FormatReader):
    """Reader for WebP images"""

    def read(self) -> ImageData:
        """
        Read WebP image

        Returns:
            ImageData with pixels and metadata

        Raises:
            IOError: If file cannot be read or is not a valid WebP
        """
        try:
            # Read image with Pillow
            with Image.open(self.filepath) as img:
                # Convert to numpy array
                pixels = np.array(img)

                # Extract metadata
                metadata: ImageMetadata = {
                    'format': 'WebP',
                    'filename': self.filepath.name,
                    'file_size': os.path.getsize(self.filepath),
                }

                # Extract EXIF if present
                if 'exif' in img.info:
                    # Store raw EXIF data for potential re-writing
                    metadata['exif'] = img.info['exif']  # type: ignore

                # Extract ICC profile if present
                if 'icc_profile' in img.info:
                    metadata['icc_profile'] = img.info['icc_profile']  # type: ignore

                # Extract XMP if present
                if 'xmp' in img.info:
                    metadata['xmp'] = img.info['xmp']  # type: ignore

                # ImageData will automatically fill: shape, dtype, channels, bit_depth
                return ImageData(pixels, metadata)

        except Exception as e:
            raise IOError(f"Error reading WebP: {e}")
