"""
WebP format save options adapter
"""
from typing import TypedDict
from image_pipeline.io.formats.base import SaveOptionsAdapter
from image_pipeline.types import SaveOptions


class WebPSaveOptions(TypedDict, total=False):
    """
    WebP-specific save options
    """
    quality: int        # 0-100 for lossy compression (ignored for lossless)
    lossless: bool      # True = lossless, False = lossy (default)
    method: int         # 0-6, compression effort/speed trade-off
    exact: bool         # Preserve RGB values in transparent areas


class WebPOptionsAdapter(SaveOptionsAdapter):
    """Adapter for WebP save options"""

    def get_supported_options(self) -> set[str]:
        """WebP supports quality, lossless, method, and exact"""
        return {'quality', 'lossless', 'method', 'exact'}

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

        # exact (bool) - preserve RGB in transparent areas
        if 'exact' in options:
            exact = options['exact']

            if not isinstance(exact, bool):
                raise TypeError(
                    f"exact must be bool, got {type(exact).__name__}"
                )

            validated['exact'] = exact
        else:
            validated['exact'] = False  # Default

        return validated
