"""
AVIF save options adapter
"""
from typing import TypedDict

from image_pipeline.io.formats.base import SaveOptionsAdapter
from image_pipeline.types import SaveOptions


class AVIFSaveOptions(TypedDict, total=False):
    """AVIF-specific save options for imagecodecs"""
    quality: int           # 0-100 (0=lowest, 100=lossless), default 90
    speed: int             # 0-10 (0=slowest/best, 10=fastest/worst), default 6
    bitspersample: int     # 8, 10, 12, or 16, default auto-detect
    numthreads: int        # Number of encoding threads, default 1


class AVIFOptionsAdapter(SaveOptionsAdapter):
    """Adapter for AVIF save options (imagecodecs backend)"""

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
            result['speed'] = 6  # Default (balanced)

        # Bit depth
        if 'bit_depth' in options:
            bit_depth = options['bit_depth']
            valid_depths = {8, 10, 12, 16}
            if bit_depth not in valid_depths:
                raise ValueError(
                    f"bit_depth must be one of {valid_depths}, got {bit_depth}"
                )
            result['bitspersample'] = bit_depth

        # Number of threads
        if 'numthreads' in options:
            threads = options['numthreads']
            if not isinstance(threads, int) or threads < 1:
                raise ValueError(f"numthreads must be >= 1, got {threads}")
            result['numthreads'] = threads

        return result

    def get_supported_options(self) -> set[str]:
        """Return set of AVIF-supported option names"""
        return {'quality', 'speed', 'bit_depth', 'numthreads'}
