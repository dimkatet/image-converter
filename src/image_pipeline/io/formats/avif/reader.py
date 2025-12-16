"""AVIF format reader"""
import numpy as np
from imagecodecs import avif_decode

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatReader
from image_pipeline.types import ImageMetadata


class AVIFFormatReader(FormatReader):
    """Reader for AVIF images using imagecodecs"""

    def read(self) -> ImageData:
        """
        Read AVIF image from file

        Returns:
            ImageData object with pixels and minimal metadata

        Note:
            imagecodecs.avif_decode() does not return metadata.
            Only basic metadata (format, filename, shape, dtype) is extracted.
        """
        try:
            # Read AVIF file bytes
            with open(self.filepath, 'rb') as f:
                avif_bytes = f.read()

            # Decode to numpy array
            pixels = avif_decode(avif_bytes)

            # Read metadata (minimal - no CICP extraction)
            metadata = self._read_metadata(pixels)

            return ImageData(pixels, metadata)

        except Exception as e:
            raise IOError(f"Error reading AVIF: {e}")

    def _read_metadata(self, pixels: np.ndarray) -> ImageMetadata:
        """
        Extract minimal metadata from AVIF file

        Args:
            pixels: Decoded pixel array

        Returns:
            ImageMetadata dictionary with basic info

        Note:
            imagecodecs does not expose CICP metadata during decoding.
            Only dtype-based bit_depth inference is performed.
        """
        metadata: ImageMetadata = {
            'format': 'AVIF',
            'filename': self.filepath.name,
            'file_size': self.filepath.stat().st_size,
        }

        # Infer bit depth from dtype
        if pixels.dtype == np.uint8:
            metadata['bit_depth'] = 8
        elif pixels.dtype == np.uint16:
            # Could be 10-bit, 12-bit, or 16-bit - default to 10 for HDR
            metadata['bit_depth'] = 10

        return metadata
