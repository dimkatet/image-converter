"""AVIF format reader"""
import numpy as np
import pillow_heif

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatReader
from image_pipeline.types import ImageMetadata


class AVIFFormatReader(FormatReader):
    """Reader for AVIF images"""

    def read(self) -> ImageData:
        """
        Read AVIF image from file

        Returns:
            ImageData object with pixels and metadata
        """
        try:
            # Read AVIF file
            heif_file = pillow_heif.open_heif(str(self.filepath))

            # Convert to PIL Image then to numpy
            pil_image = heif_file.to_pillow()
            pixels = np.array(pil_image)

            # Read metadata
            metadata = self._read_metadata(heif_file, pixels)

            return ImageData(pixels, metadata)

        except Exception as e:
            raise IOError(f"Error reading AVIF: {e}")

    def _read_metadata(self, heif_file, pixels: np.ndarray) -> ImageMetadata:
        """
        Extract metadata from AVIF file

        Args:
            heif_file: pillow-heif file object
            pixels: Pixel array (for shape/dtype info)

        Returns:
            ImageMetadata dictionary
        """
        metadata: ImageMetadata = {
            'format': 'AVIF',
            'filename': self.filepath.name,
            'file_size': self.filepath.stat().st_size,
        }

        # Get bit depth from heif info
        if hasattr(heif_file, 'info') and 'bit_depth' in heif_file.info:
            metadata['bit_depth'] = heif_file.info['bit_depth']

        # Get CICP/NCLX color information
        if hasattr(heif_file, 'info') and 'nclx' in heif_file.info:
            nclx = heif_file.info['nclx']
            # nclx is tuple: (color_primaries, transfer_characteristics, matrix_coefficients, full_range_flag)
            if len(nclx) >= 2:
                # Map transfer characteristics back to transfer_function
                from image_pipeline.constants import TRANSFER_TO_CICP
                for tf_name, cicp_code in TRANSFER_TO_CICP.items():
                    if cicp_code == nclx[1]:
                        metadata['transfer_function'] = tf_name
                        break

                # Map color primaries back to color_space
                from image_pipeline.constants import COLORSPACE_TO_CICP
                for cs_name, cicp_code in COLORSPACE_TO_CICP.items():
                    if cicp_code == nclx[0]:
                        metadata['color_space'] = cs_name
                        break

        return metadata
