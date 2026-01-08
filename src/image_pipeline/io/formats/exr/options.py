"""OpenEXR save options adapter"""

from typing import Any
import OpenEXR

from image_pipeline.io.formats.base import SaveOptionsAdapter
from image_pipeline.types import SaveOptions


# OpenEXR.File API uses OpenEXR compression constants
COMPRESSION_MAP = {
    'none': OpenEXR.NO_COMPRESSION,
    'rle': OpenEXR.RLE_COMPRESSION,
    'zip': OpenEXR.ZIP_COMPRESSION,
    'zips': OpenEXR.ZIPS_COMPRESSION,
    'piz': OpenEXR.PIZ_COMPRESSION,
    'pxr24': OpenEXR.PXR24_COMPRESSION,
    'b44': OpenEXR.B44_COMPRESSION,
    'b44a': OpenEXR.B44A_COMPRESSION,
    'dwaa': OpenEXR.DWAA_COMPRESSION,
    'dwab': OpenEXR.DWAB_COMPRESSION,
}

# OpenEXR.File API uses dtype strings for pixel types
PIXEL_TYPE_MAP = {
    'half': 'float16',
    'float': 'float32',
    'uint': 'uint32',
}


class EXRSaveOptionsAdapter(SaveOptionsAdapter):
    """Adapter for EXR-specific save options"""

    @staticmethod
    def adapt(options: SaveOptions) -> dict[str, Any]:
        """
        Convert SaveOptions to EXR-specific parameters

        Args:
            options: Generic save options

        Returns:
            Dictionary with EXR-specific options

        Supported options:
            - compression: 'none', 'rle', 'zip' (default), 'zips', 'piz',
                          'pxr24', 'b44', 'b44a', 'dwaa', 'dwab'
            - pixel_type: 'half' (default), 'float', 'uint'
        """
        result: dict[str, Any] = {}

        # Compression
        compression = options.get('compression', 'zip')
        if isinstance(compression, str):
            compression = compression.lower()
            if compression in COMPRESSION_MAP:
                result['compression'] = COMPRESSION_MAP[compression]
            else:
                # Default to ZIP if unknown
                result['compression'] = OpenEXR.ZIP_COMPRESSION
        else:
            # Default
            result['compression'] = OpenEXR.ZIP_COMPRESSION

        # Pixel type (maps to numpy dtype)
        pixel_type = options.get('pixel_type', 'half')
        if isinstance(pixel_type, str):
            pixel_type = pixel_type.lower()
            if pixel_type in PIXEL_TYPE_MAP:
                result['pixel_type'] = PIXEL_TYPE_MAP[pixel_type]
            else:
                # Default to float16 (half)
                result['pixel_type'] = 'float16'
        else:
            # Default to float16 (half)
            result['pixel_type'] = 'float16'

        return result
