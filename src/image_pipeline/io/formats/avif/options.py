"""
AVIF save options adapter
"""
from typing import TypedDict, Mapping, Any
import warnings

from image_pipeline.io.formats.base import SaveOptionsAdapter
from image_pipeline.types import SaveOptions


class AVIFSaveOptions(TypedDict, total=False):
    """AVIF-specific save options"""
    quality: int  # 0-100, default 90
    speed: int  # 0-10 (0=slowest/best, 10=fastest/worst), default 4
    chroma_subsampling: str  # "444", "422", "420", default "444"


class AVIFOptionsAdapter(SaveOptionsAdapter):
    """Adapter for AVIF save options"""

    def validate(self, options: SaveOptions) -> AVIFSaveOptions:
        """
        Validate and normalize AVIF save options

        Args:
            options: Generic save options

        Returns:
            Validated AVIF-specific options

        Raises:
            ValueError: If option values are invalid
        """
        self._warn_unsupported(options, "AVIF")

        result: AVIFSaveOptions = {}

        # Quality (0-100)
        if 'quality' in options:
            quality = options['quality']
            if not isinstance(quality, int) or not 0 <= quality <= 100:
                raise ValueError(f"quality must be 0-100, got {quality}")
            result['quality'] = quality
        else:
            result['quality'] = 90  # Default

        # Speed (0-10)
        if 'speed' in options:
            speed = options['speed']
            if not isinstance(speed, int) or not 0 <= speed <= 10:
                raise ValueError(f"speed must be 0-10, got {speed}")
            result['speed'] = speed
        else:
            result['speed'] = 4  # Default

        # Chroma subsampling
        if 'chroma_subsampling' in options:
            chroma = options['chroma_subsampling']
            valid_chroma = {'444', '422', '420'}
            if chroma not in valid_chroma:
                raise ValueError(
                    f"chroma_subsampling must be one of {valid_chroma}, got {chroma}"
                )
            result['chroma_subsampling'] = chroma
        else:
            result['chroma_subsampling'] = '444'  # Default: 444 for HDR quality

        return result

    def get_supported_options(self) -> set[str]:
        """Return set of AVIF-supported option names"""
        return {'quality', 'speed', 'chroma_subsampling'}
