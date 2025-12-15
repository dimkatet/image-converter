"""AVIF format writer"""
import numpy as np
import pillow_heif

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatWriter
from image_pipeline.io.formats.avif.options import AVIFOptionsAdapter, AVIFSaveOptions
from image_pipeline.io.formats.avif.adapter import AVIFEncodingMetadata, AVIFMetadataAdapter
from image_pipeline.types import SaveOptions


class AVIFFormatWriter(FormatWriter):
    """Writer for AVIF images with HDR metadata support"""

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.options_adapter = AVIFOptionsAdapter()
        self.metadata_adapter = AVIFMetadataAdapter()

    def write(self, img_data: ImageData, options: SaveOptions) -> None:
        """
        Write AVIF image with embedded metadata

        AVIF uses a single-pass write where metadata is encoded
        together with pixels during compression.

        Args:
            img_data: ImageData with pixels and metadata
            options: Save options
        """
        pixels = img_data.pixels

        # Validate options
        validated_options = self.options_adapter.validate(options)

        # Prepare metadata for encoding
        encoding_metadata = self.metadata_adapter.prepare_for_encoding(
            img_data.metadata
        )

        # Write everything at once
        self._write_with_metadata(pixels, validated_options, encoding_metadata)

    def validate(self, img_data: ImageData) -> None:
        """
        Validate that data is compatible with AVIF format

        AVIF supports: uint8, uint16 (for 8/10/12-bit)
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

    def _write_with_metadata(
        self,
        pixels: np.ndarray,
        options: AVIFSaveOptions,
        metadata: AVIFEncodingMetadata
    ) -> None:
        """
        Write AVIF with embedded metadata using pillow-heif

        Uses add_frombytes to preserve true 10/12-bit data without
        downsampling to 8-bit.

        Args:
            pixels: Pixel array (uint8 or uint16)
            options: Validated AVIF save options
            metadata: Encoding metadata (bit_depth, CICP, etc.)
        """
        try:
            # Create HeifFile
            hf = pillow_heif.HeifFile()

            # Determine mode and dimensions
            height, width = pixels.shape[:2]

            if len(pixels.shape) == 2:
                # Grayscale
                mode = "L" if pixels.dtype == np.uint8 else "I;16"

            elif len(pixels.shape) == 3:
                channels = pixels.shape[2]

                if channels == 1:
                    # Grayscale with channel dimension
                    pixels = pixels.squeeze(-1)
                    mode = "L" if pixels.dtype == np.uint8 else "I;16"

                elif channels == 3:
                    # RGB
                    mode = "RGB" if pixels.dtype == np.uint8 else "RGB;16"

                elif channels == 4:
                    # RGBA
                    mode = "RGBA" if pixels.dtype == np.uint8 else "RGBA;16"

                else:
                    raise ValueError(f"Unsupported number of channels: {channels}")
            else:
                raise ValueError(f"Unsupported array shape: {pixels.shape}")

            # Add image from bytes (preserves uint16 data)
            img = hf.add_frombytes(mode, (width, height), pixels.tobytes())

            # Set bit_depth from metadata (critical for 10/12-bit)
            if 'bit_depth' in metadata:
                img.info['bit_depth'] = metadata['bit_depth']

            # Prepare save parameters
            save_params = {
                'quality': options.get('quality', 90),
                'chroma': options.get('chroma_subsampling', '444'),
            }

            # Add CICP/NCLX parameters if available (as separate keys, not dict)
            if all(k in metadata for k in ['color_primaries', 'transfer_characteristics']):
                save_params['color_primaries'] = metadata.get('color_primaries')
                save_params['transfer_characteristic'] = metadata.get('transfer_characteristics')
                save_params['matrix_coefficients'] = metadata.get('matrix_coefficients')
                save_params['full_range_flag'] = 1 if metadata.get('full_range_flag', True) else 0

            # Save
            hf.save(str(self.filepath), **save_params)

        except Exception as e:
            raise IOError(f"Error writing AVIF: {e}")
