"""
AVIF Metadata Adapter - converts ImageMetadata to AVIF encoding parameters
"""
from typing import TypedDict, Optional

from image_pipeline.types import ImageMetadata
from image_pipeline.constants import TRANSFER_TO_CICP, COLORSPACE_TO_CICP


class AVIFEncodingMetadata(TypedDict, total=False):
    """Metadata prepared for AVIF encoding via pillow-heif"""
    bit_depth: int
    color_primaries: int
    transfer_characteristics: int
    matrix_coefficients: int
    full_range_flag: bool


class AVIFMetadataAdapter:
    """Converts generic ImageMetadata to AVIF encoding parameters"""

    @staticmethod
    def prepare_for_encoding(metadata: ImageMetadata) -> AVIFEncodingMetadata:
        """
        Convert ImageMetadata to pillow-heif encoding parameters

        Args:
            metadata: Generic image metadata

        Returns:
            AVIF-specific encoding metadata
        """
        result: AVIFEncodingMetadata = {}

        # Bit depth - pass through directly for 8/10/12, map others
        bit_depth = metadata.get('bit_depth')
        if bit_depth:
            # AVIF supports 8, 10, 12-bit
            if bit_depth in (8, 10, 12):
                result['bit_depth'] = bit_depth
            elif bit_depth <= 8:
                result['bit_depth'] = 8
            elif bit_depth <= 10:
                result['bit_depth'] = 10
            else:  # 16, 32
                result['bit_depth'] = 12

        # Color information (CICP parameters)
        cicp_params = AVIFMetadataAdapter._extract_cicp_params(metadata)
        if cicp_params:
            result.update(cicp_params)

        return result

    @staticmethod
    def _extract_cicp_params(metadata: ImageMetadata) -> Optional[dict]:
        """
        Extract CICP parameters from metadata

        Returns dict with: color_primaries, transfer_characteristics,
        matrix_coefficients, full_range_flag
        """
        transfer_function = metadata.get('transfer_function')
        color_space = metadata.get('color_space')

        if not transfer_function and not color_space:
            return None

        # Transfer characteristics
        transfer_code = TRANSFER_TO_CICP.get(transfer_function, 2) if transfer_function else 2

        # Color primaries
        color_primaries_code = 2  # Default: unspecified
        if color_space:
            color_primaries_code = COLORSPACE_TO_CICP.get(color_space, 2)

        # Matrix coefficients
        # For BT.2020 HDR: use 9 (BT.2020 non-constant luminance YCbCr)
        # AVIF/AV1 encodes in YCbCr, not RGB
        if color_space and 'BT2020' in str(color_space):
            matrix_coefficients = 1  # BT.2020 NCL
        else:
            matrix_coefficients = 2  # Unspecified

        # Full range flag (True for full range)
        full_range_flag = False

        return {
            'color_primaries': color_primaries_code,
            'transfer_characteristics': transfer_code,
            'matrix_coefficients': matrix_coefficients,
            'full_range_flag': full_range_flag
        }
