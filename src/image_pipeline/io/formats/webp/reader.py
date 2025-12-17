"""WebP format reader using imagecodecs"""
import os
from imagecodecs import webp_decode

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatReader
from image_pipeline.types import ImageMetadata


class WebPFormatReader(FormatReader):
    """Reader for WebP images using imagecodecs"""

    def read(self) -> ImageData:
        """
        Read WebP image

        Note: imagecodecs.webp_decode doesn't extract metadata (EXIF/XMP).
        Only pixel data and basic file info are returned.

        Returns:
            ImageData with pixels and minimal metadata

        Raises:
            IOError: If file cannot be read or is not a valid WebP
        """
        try:
            # Read WebP file bytes
            with open(self.filepath, 'rb') as f:
                webp_bytes = f.read()

            # Decode to numpy array
            pixels = webp_decode(webp_bytes)

            # Extract minimal metadata
            metadata: ImageMetadata = {
                'format': 'WebP',
                'filename': self.filepath.name,
                'file_size': os.path.getsize(self.filepath),
            }

            # ImageData will automatically fill: shape, dtype, channels, bit_depth
            return ImageData(pixels, metadata)

        except Exception as e:
            raise IOError(f"Error reading WebP: {e}")
