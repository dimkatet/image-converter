"""PNG format writer using imagecodecs"""
import numpy as np
from imagecodecs import png_encode

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatWriter
from image_pipeline.io.formats.png.options import PNGOptionsAdapter, PNGSaveOptions
from image_pipeline.types import SaveOptions


class PNGFormatWriter(FormatWriter):
    """Writer for PNG images using imagecodecs"""

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.options_adapter = PNGOptionsAdapter()

    def write(self, img_data: ImageData, options: SaveOptions) -> None:
        """
        Write PNG image with metadata

        PNG uses a two-step process:
        1. Encode pixels to PNG bytes using imagecodecs
        2. Add metadata chunks to the encoded PNG file

        Args:
            img_data: ImageData with pixels and metadata
            options: Save options
        """
        # Validate image data first
        self.validate(img_data)

        # Validate options
        validated_options = self.options_adapter.validate(options)

        # Load ICC profile from file if specified in options
        if 'icc_profile' in validated_options:
            icc_path = validated_options['icc_profile']
            try:
                with open(icc_path, 'rb') as f:
                    icc_data = f.read()
                # Add to metadata (overrides any existing profile)
                img_data.metadata['icc_profile'] = icc_data
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Failed to load ICC profile from {icc_path}: {e}",
                    UserWarning
                )

        # Step 1: Encode and write pixels
        self._write_pixels(img_data.pixels, validated_options)

        # Step 2: Add metadata chunks
        self._write_metadata(img_data.metadata)

    def _write_metadata(self, metadata) -> None:
        """Add metadata chunks to existing PNG file"""
        from image_pipeline.metadata.png import PNGMetadataWriter
        PNGMetadataWriter.write_metadata(str(self.filepath), metadata)
    
    def validate(self, img_data: ImageData) -> None:
        """
        Validate that data is compatible with PNG format

        PNG supports:
        - Pixel dtypes: uint8 (8-bit), uint16 (16-bit)
        - Note: 10-bit and 12-bit are NOT standard PNG bit depths
        - Use uint16 for >8-bit data (actual bit depth can be stored in metadata)

        Args:
            img_data: ImageData to validate

        Raises:
            ValueError: If validation fails
        """
        pixels = img_data.pixels

        if not isinstance(pixels, np.ndarray):
            raise ValueError("Data must be a numpy array")

        if pixels.size == 0:
            raise ValueError("Empty pixel array")

        if pixels.dtype not in (np.uint8, np.uint16):
            raise ValueError(
                f"PNG supports only uint8 and uint16 dtypes. "
                f"Got: {pixels.dtype}.\n"
                f"Solutions:\n"
                f"  1. For float: use quantize filter first\n"
                f"  2. For uint32: convert to uint16"
            )

        # Validate shape
        if pixels.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array, got shape {pixels.shape}")

        if pixels.ndim == 3 and pixels.shape[2] not in (1, 2, 3, 4):
            raise ValueError(
                f"Expected 1, 2, 3, or 4 channels, got {pixels.shape[2]}"
            )

        # Warn if metadata contains unsupported bit_depth
        bit_depth = img_data.metadata.get('bit_depth')
        if bit_depth is not None and bit_depth not in (8, 16):
            import warnings
            warnings.warn(
                f"PNG bit_depth in metadata is {bit_depth}, but PNG standard only supports 8 and 16 bits per sample. "
                f"The image will be stored as {pixels.dtype.itemsize * 8}-bit. "
                f"Logical bit_depth {bit_depth} will not be preserved on read.",
                UserWarning
            )

    def _write_pixels(self, pixels: np.ndarray, options: PNGSaveOptions) -> None:
        """
        Encode and write PNG pixel data using imagecodecs

        imagecodecs.png_encode handles both uint8 and uint16 automatically.

        Args:
            pixels: Pixel array (uint8 or uint16)
            options: Validated PNG save options

        Raises:
            IOError: If encoding or writing fails
        """
        try:
            # Extract options
            level = options.get('compression_level', 6)
            strategy = options.get('strategy', 0)

            # Encode to PNG bytes
            png_bytes = png_encode(pixels, level=level, strategy=strategy)

            # Write to file
            with open(self.filepath, 'wb') as f:
                f.write(png_bytes)

        except Exception as e:
            raise IOError(f"Error writing PNG: {e}")
