"""JPEG XR format reader using imagecodecs"""
import os
from imagecodecs import jpegxr_decode

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatReader
from image_pipeline.types import ImageMetadata


class JXRFormatReader(FormatReader):
    """Reader for JPEG XR images using imagecodecs"""

    def read(self) -> ImageData:
        """
        Read JPEG XR image

        Note: imagecodecs.jpegxr_decode doesn't extract extended metadata
        (color primaries, transfer function, EXIF). Only pixel data and
        basic file info are returned.

        Returns:
            ImageData with pixels and minimal metadata

        Raises:
            IOError: If file cannot be read or is not a valid JPEG XR
        """
        try:
            # Read JXR file bytes
            with open(self.filepath, 'rb') as f:
                jxr_bytes = f.read()

            # Decode to numpy array
            # fp2int=False: return fixed-point images as float32 (not int16/int32)
            pixels = jpegxr_decode(jxr_bytes, fp2int=False)

            # Extract minimal metadata
            metadata: ImageMetadata = {
                'format': 'JXR',
                'filename': self.filepath.name,
                'file_size': os.path.getsize(self.filepath),
            }

            # ImageData will automatically fill: shape, dtype, channels, bit_depth
            return ImageData(pixels, metadata)

        except Exception as e:
            raise IOError(f"Error reading JPEG XR: {e}")
