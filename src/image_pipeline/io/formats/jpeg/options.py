"""
Save options adapter for JPEG format (standard and Ultra HDR)
"""

from typing import Any
from image_pipeline.types import SaveOptions


class JPEGSaveOptionsAdapter:
    """
    Adapter for JPEG save options (both standard and Ultra HDR)

    Supported options:
        - quality: JPEG quality (1-100, default: 95)
        - ultra_hdr: Enable Ultra HDR encoding (bool, default: False)
        - gainmap_scale: Gainmap downscale factor for Ultra HDR (default: 4)

    Note: For Ultra HDR, hdr_nits/hdr_gamut/hdr_transfer are derived from ImageMetadata,
    not from SaveOptions. These must be set via filters (ColorConvertFilter, PQEncodeFilter, etc.)
    """

    SUPPORTED_OPTIONS = {
        'quality',
        'ultra_hdr',
        'gainmap_scale',
    }

    DEFAULTS = {
        'quality': 95,
        'ultra_hdr': False,
        'gainmap_scale': 4,
    }

    @classmethod
    def adapt(cls, options: SaveOptions) -> dict[str, Any]:
        """
        Validate and adapt save options for JPEG encoding

        Args:
            options: Raw save options dictionary

        Returns:
            Dictionary with validated JPEG options

        Raises:
            ValueError: If option values are invalid
        """
        adapted: dict[str, Any] = {}

        # Quality
        quality = options.get('quality', cls.DEFAULTS['quality'])
        if not isinstance(quality, int):
            raise TypeError(f"quality must be int, got {type(quality).__name__}")
        if not 1 <= quality <= 100:
            raise ValueError(f"quality must be in range [1, 100], got {quality}")
        adapted['quality'] = quality

        # Ultra HDR flag
        ultra_hdr = options.get('ultra_hdr', cls.DEFAULTS['ultra_hdr'])
        if not isinstance(ultra_hdr, bool):
            raise TypeError(f"ultra_hdr must be bool, got {type(ultra_hdr).__name__}")
        adapted['ultra_hdr'] = ultra_hdr

        # Gainmap scale (only for Ultra HDR)
        gainmap_scale = options.get('gainmap_scale', cls.DEFAULTS['gainmap_scale'])
        if not isinstance(gainmap_scale, int):
            raise TypeError(f"gainmap_scale must be int, got {type(gainmap_scale).__name__}")
        if gainmap_scale < 1:
            raise ValueError(f"gainmap_scale must be >= 1, got {gainmap_scale}")
        adapted['gainmap_scale'] = gainmap_scale

        return adapted
