"""
WebP format save options adapter
"""
from typing import TypedDict
from image_pipeline.io.formats.base import SaveOptionsAdapter
from image_pipeline.types import SaveOptions


class WebPSaveOptions(TypedDict, total=False):
    """
    WebP-specific save options

    Options:
        quality: Quality level (0-100, default: 80)
            For lossy: lower = smaller file, worse quality
            For lossless: ignored

        lossless: Lossless compression (bool, default: False)
            True = lossless compression
            False = lossy compression

        method: Compression effort (0-6, default: 4)
            0 = fastest, largest file
            6 = slowest, smallest file

        numthreads: Number of threads (default: None = auto)
            Use multi-threading for encoding
    """
    quality: int        # 0-100 for lossy compression
    lossless: bool      # True = lossless, False = lossy
    method: int         # 0-6, compression effort/speed
    numthreads: int     # Number of threads for encoding


class WebPOptionsAdapter(SaveOptionsAdapter):
    """Adapter for WebP save options"""

    def get_supported_options(self) -> set[str]:
        """WebP supports quality, lossless, method, and numthreads"""
        return {'quality', 'lossless', 'method', 'numthreads'}

    def validate(self, options: SaveOptions) -> WebPSaveOptions:
        """
        Validate and normalize WebP save options

        Args:
            options: User-provided save options

        Returns:
            Dictionary with validated WebP-specific options

        Raises:
            ValueError: If option values are invalid
            TypeError: If option types are incorrect
        """
        validated: WebPSaveOptions = {}

        # Warn about unsupported options
        self._warn_unsupported(options, 'WebP')

        # lossless (bool) - check first as it affects quality handling
        if 'lossless' in options:
            lossless = options['lossless']

            if not isinstance(lossless, bool):
                raise TypeError(
                    f"lossless must be bool, got {type(lossless).__name__}"
                )

            validated['lossless'] = lossless
        else:
            validated['lossless'] = False  # Default: lossy

        # quality (0-100) - only meaningful for lossy
        if 'quality' in options:
            quality = options['quality']

            if not isinstance(quality, int):
                raise TypeError(
                    f"quality must be int, got {type(quality).__name__}"
                )

            if not 0 <= quality <= 100:
                raise ValueError(
                    f"quality must be in range [0, 100], got {quality}"
                )

            validated['quality'] = quality
        else:
            validated['quality'] = 80  # Default

        # method (0-6) - compression effort
        if 'method' in options:
            method = options['method']

            if not isinstance(method, int):
                raise TypeError(
                    f"method must be int, got {type(method).__name__}"
                )

            if not 0 <= method <= 6:
                raise ValueError(
                    f"method must be in range [0, 6], got {method}"
                )

            validated['method'] = method
        else:
            validated['method'] = 4  # Default

        # numthreads (int) - number of threads
        if 'numthreads' in options:
            numthreads = options['numthreads']

            if not isinstance(numthreads, int):
                raise TypeError(
                    f"numthreads must be int, got {type(numthreads).__name__}"
                )

            if numthreads < 1:
                raise ValueError(
                    f"numthreads must be >= 1, got {numthreads}"
                )

            validated['numthreads'] = numthreads
        # else: Don't set default - let imagecodecs decide

        return validated
