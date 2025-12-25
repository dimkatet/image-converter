"""
JPEG format reader with automatic detection and delegation
"""

import imagecodecs
from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatReader


class JPEGReader(FormatReader):
    """
    Facade for JPEG reading with automatic format detection

    Detects and delegates to:
        - UltraHDRReader: if imagecodecs.ultrahdr_check() returns True
        - StandardJPEGReader: if file is standard JPEG (TODO: not implemented yet)

    This facade handles:
        - Automatic format detection
        - Delegation to appropriate reader
        - Consistent interface for both JPEG variants
    """

    def validate_file(self) -> None:
        """
        Validate that file exists

        Format-specific validation (Ultra HDR vs standard JPEG)
        happens in the delegated reader
        """
        # Only check file existence here
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        if not self.filepath.is_file():
            raise ValueError(f"Path is not a file: {self.filepath}")

    def read(self) -> ImageData:
        """
        Read JPEG file (standard or Ultra HDR)

        Returns:
            ImageData with pixels and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid JPEG
            RuntimeError: If decoding fails
            NotImplementedError: If standard JPEG is detected (not implemented yet)
        """
        # Read file to detect format
        with open(self.filepath, 'rb') as f:
            file_data = f.read()

        # Check if it's Ultra HDR
        is_ultrahdr = imagecodecs.ultrahdr_check(file_data)

        if is_ultrahdr:
            # Use Ultra HDR implementation
            from .ultrahdr import UltraHDRReader
            reader = UltraHDRReader(str(self.filepath))
            return reader.read()
        else:
            # Standard JPEG not implemented yet
            raise NotImplementedError(
                f"Standard JPEG reading is not implemented yet. "
                f"File {self.filepath} is not a JPEG Ultra HDR file. "
                f"Use imageio or Pillow for standard JPEG reading, "
                f"or convert to a supported HDR format (TIFF, AVIF, WebP)."
            )
