"""
JPEG Ultra HDR format reader using imagecodecs.libultrahdr
"""

import numpy as np
import imagecodecs

from image_pipeline.core.image_data import ImageData
from image_pipeline.types import TransferFunction, ColorSpace
from image_pipeline.io.formats.base import FormatReader


class UltraHDRReader(FormatReader):
    """
    Reader for JPEG Ultra HDR format

    Output:
        - dtype: float32 (converted from float16)
        - channels: 4 (RGBA - alpha channel from gainmap)
        - transfer_function: LINEAR or PQ (depending on decoding)
        - color_space: Detected from file metadata

    The reader will:
        1. Validate file is Ultra HDR using imagecodecs.ultrahdr_check()
        2. Decode to RGBAHalfFloat (float16)
        3. Convert to float32 for pipeline compatibility
        4. Extract and set metadata
    """

    def validate_file(self) -> None:
        """
        Validate that file exists and is a valid Ultra HDR JPEG

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid Ultra HDR JPEG
        """
        super().validate_file()

        # Read file and check if it's Ultra HDR
        with open(self.filepath, 'rb') as f:
            file_data = f.read()

        if not imagecodecs.ultrahdr_check(file_data):
            raise ValueError(
                f"File is not a valid JPEG Ultra HDR: {self.filepath}. "
                f"Use ultrahdr_check() to verify, or use standard JPEG reader for non-Ultra HDR files."
            )

    def read(self) -> ImageData:
        """
        Read JPEG Ultra HDR file

        Returns:
            ImageData with HDR pixels and metadata

        Raises:
            RuntimeError: If decoding fails
        """
        # Read file
        with open(self.filepath, 'rb') as f:
            file_data = f.read()

        # Decode to float16 (RGBAHalfFloat)
        # We use LINEAR transfer to get linear HDR data
        try:
            pixels_f16 = imagecodecs.ultrahdr_decode(
                file_data,
                dtype='float16',  # RGBAHalfFloat
                transfer=imagecodecs.ULTRAHDR.CT.LINEAR,
            )
        except Exception as e:
            raise RuntimeError(f"Ultra HDR decoding failed: {e}") from e

        # Convert to float32 for pipeline compatibility
        pixels = pixels_f16.astype(np.float32)

        # Create ImageData with basic metadata
        img_data = ImageData(pixels)

        # Set format metadata
        img_data.metadata['format'] = 'JPEG Ultra HDR'
        img_data.metadata['filename'] = self.filepath.name
        img_data.metadata['file_size'] = self.filepath.stat().st_size

        # Set transfer function (we decoded to LINEAR)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR

        # TODO: Extract color space from XMP metadata
        # For now, we don't know the color space from decoded data
        # imagecodecs.ultrahdr_decode() doesn't return metadata
        # This would require XMP parsing from the original JPEG
        # For MVP, we leave color_space unset

        return img_data
