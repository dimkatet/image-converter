"""AVIF format writer"""
import numpy as np
from imagecodecs import avif_encode

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatWriter
from image_pipeline.io.formats.avif.options import AVIFOptionsAdapter, AVIFSaveOptions
from image_pipeline.io.formats.avif.adapter import AVIFEncodingMetadata, AVIFMetadataAdapter
from image_pipeline.types import SaveOptions

class AVIFFormatWriter(FormatWriter):
    """Writer for AVIF images with HDR metadata support using imagecodecs"""

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.options_adapter = AVIFOptionsAdapter()
        self.metadata_adapter = AVIFMetadataAdapter()

    def write(self, img_data: ImageData, options: SaveOptions) -> None:
        """
        Write AVIF image with embedded CICP metadata

        Uses imagecodecs.avif_encode with full control over:
        - Bit depth (8/10/12/16-bit)
        - CICP color parameters (primaries, transfer, matrix)
        - Quality and speed settings

        Args:
            img_data: ImageData with pixels and metadata
            options: Save options

        Raises:
            ValueError: If data validation fails
            IOError: If encoding or file write fails
        """
        pixels = img_data.pixels

        # Validate options
        validated_options = self.options_adapter.validate(options)

        # Prepare CICP metadata for encoding
        encoding_metadata = self.metadata_adapter.prepare_for_encoding(
            img_data.metadata
        )

        # Encode and write
        self._encode_and_write(pixels, validated_options, encoding_metadata)

    def validate(self, img_data: ImageData) -> None:
        """
        Validate that data is compatible with AVIF format

        AVIF supports: uint8, uint16 (for 8/10/12/16-bit)

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
                f"AVIF supports only uint8 and uint16. "
                f"Got: {pixels.dtype}.\n"
                f"Solutions:\n"
                f"  1. For float: use quantize filter first\n"
                f"  2. For uint32: convert to uint16"
            )

        # Validate shape
        if pixels.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array, got shape {pixels.shape}")

        if pixels.ndim == 3 and pixels.shape[2] not in (1, 3, 4):
            raise ValueError(
                f"Expected 1, 3, or 4 channels, got {pixels.shape[2]}"
            )

    def _encode_and_write(
        self,
        pixels: np.ndarray,
        options: AVIFSaveOptions,
        metadata: AVIFEncodingMetadata
    ) -> None:
        """
        Encode pixels to AVIF bytes and write to file

        Args:
            pixels: Pixel array (uint8 or uint16)
            options: Validated AVIF save options
            metadata: Encoding metadata (CICP parameters, bit depth)

        Raises:
            IOError: If encoding or file write fails
        """
        try:
            # Prepare encoding parameters
            encode_params = self._build_encode_params(pixels, options, metadata)

            # Encode to AVIF bytes
            avif_bytes = avif_encode(pixels, **encode_params)

            # Write to file
            with open(self.filepath, 'wb') as f:
                f.write(avif_bytes)

        except Exception as e:
            raise IOError(f"Error writing AVIF: {e}")

    def _build_encode_params(
        self,
        pixels: np.ndarray,
        options: AVIFSaveOptions,
        metadata: AVIFEncodingMetadata
    ) -> dict:
        """
        Build parameter dictionary for imagecodecs.avif_encode

        Combines save options and metadata into a single dict.
        Auto-detects bitspersample from dtype if not specified.

        Args:
            pixels: Pixel array
            options: Save options (quality, speed, numthreads)
            metadata: Encoding metadata (primaries, transfer, matrix, bitspersample)

        Returns:
            Dictionary of parameters for avif_encode()
        """
        params = {}

        # Quality (required parameter)
        params['level'] = options.get('quality', 90)

        # Speed (optional)
        if 'speed' in options:
            params['speed'] = options['speed']

        # Number of threads (optional)
        if 'numthreads' in options:
            params['numthreads'] = options['numthreads']

        # Bit depth - prefer metadata, fallback to options, then auto-detect
        if 'bitspersample' in metadata:
            params['bitspersample'] = metadata['bitspersample']
        elif 'bitspersample' in options:
            params['bitspersample'] = options['bitspersample']
        else:
            # Auto-detect from dtype
            if pixels.dtype == np.uint8:
                params['bitspersample'] = 8
            elif pixels.dtype == np.uint16:
                # Default to 10-bit for uint16 (common HDR use case)
                params['bitspersample'] = 10

        # params['bitspersample'] = 12
        # CICP parameters (primaries, transfer, matrix)
        if 'primaries' in metadata:
            params['primaries'] = metadata['primaries']
        if 'transfer' in metadata:
            params['transfer'] = metadata['transfer']
        if 'matrix' in metadata:
            params['matrix'] = metadata['matrix']

        return params
