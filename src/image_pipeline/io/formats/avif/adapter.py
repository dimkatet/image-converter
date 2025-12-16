"""
AVIF Metadata Adapter - converts ImageMetadata to AVIF encoding parameters
"""
from typing import TypedDict, Optional

from image_pipeline.types import ImageMetadata
from image_pipeline.constants import (
    TRANSFER_TO_CICP,
    COLORSPACE_TO_CICP,
    COLORSPACE_TO_MATRIX
)


class AVIFEncodingMetadata(TypedDict, total=False):
    """Metadata prepared for AVIF encoding via imagecodecs.avif_encode"""
    primaries: int           # CICP color primaries code
    transfer: int            # CICP transfer characteristics code
    matrix: int              # CICP matrix coefficients code
    bitspersample: int       # Bit depth: 8, 10, 12, or 16


class AVIFMetadataAdapter:
    """Converts generic ImageMetadata to imagecodecs AVIF encoding parameters"""

    @staticmethod
    def prepare_for_encoding(metadata: ImageMetadata) -> AVIFEncodingMetadata:
        """
        Convert ImageMetadata to imagecodecs.avif_encode parameters

        Args:
            metadata: Generic image metadata

        Returns:
            AVIF-specific encoding metadata (primaries, transfer, matrix, bitspersample)

        Note:
            imagecodecs uses parameter names:
            - primaries (not color_primaries)
            - transfer (not transfer_characteristics)
            - matrix (not matrix_coefficients)
            - bitspersample (not bit_depth)
        """
        result: AVIFEncodingMetadata = {}

        # Bit depth mapping
        bit_depth = metadata.get('bit_depth')
        if bit_depth:
            result['bitspersample'] = AVIFMetadataAdapter._map_bit_depth(bit_depth)

        # CICP parameters (primaries, transfer, matrix)
        cicp_params = AVIFMetadataAdapter._extract_cicp_params(metadata)
        if cicp_params:
            result.update(cicp_params)

        return result

    @staticmethod
    def _map_bit_depth(bit_depth: int) -> int:
        """
        Map bit_depth to AVIF-compatible values

        Args:
            bit_depth: Original bit depth (8, 10, 12, 16, 32)

        Returns:
            AVIF bit depth (8, 10, 12, or 16)
        """
        if bit_depth <= 8:
            return 8
        elif bit_depth <= 10:
            return 10
        elif bit_depth <= 12:
            return 12
        else:  # 16, 32
            return 16

    @staticmethod
    def _extract_cicp_params(metadata: ImageMetadata) -> Optional[dict]:
        """
        Extract CICP parameters from metadata

        Returns dict with: primaries, transfer, matrix codes

        Note:
            Uses COLORSPACE_TO_CICP, TRANSFER_TO_CICP, COLORSPACE_TO_MATRIX
            mappings from constants.py
        """
        transfer_function = metadata.get('transfer_function')
        color_space = metadata.get('color_space')

        # If neither is specified, return None (use encoder defaults)
        if not transfer_function and not color_space:
            return None

        result = {}

        # Transfer characteristics
        if transfer_function:
            transfer_code = TRANSFER_TO_CICP.get(transfer_function)
            if transfer_code is not None:
                result['transfer'] = transfer_code

        # Color primaries
        if color_space:
            primaries_code = COLORSPACE_TO_CICP.get(color_space)
            if primaries_code is not None:
                result['primaries'] = primaries_code

            # Matrix coefficients (derived from color space)
            matrix_code = COLORSPACE_TO_MATRIX.get(color_space)
            if matrix_code is not None:
                result['matrix'] = matrix_code

        return result if result else None
