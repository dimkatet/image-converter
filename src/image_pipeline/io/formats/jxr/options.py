"""
JPEG XR format save options adapter
"""
from typing import TypedDict, Optional
from image_pipeline.io.formats.base import SaveOptionsAdapter
from image_pipeline.types import SaveOptions


class JXRSaveOptions(TypedDict, total=False):
    """
    JPEG XR-specific save options

    Options:
        lossless: Lossless compression (bool, default: True)
            True = lossless compression (level=100)
            False = lossy compression using specified quality level
            Note: float data always has minimal precision loss (~3e-5)

        quality: Quality level (1-100, default: 90)
            Only used when lossless=False
            Higher = better quality, larger file size
            For uint8/uint16: affects compression ratio
            For float16/float32: minimal effect (format limitation)

        photometric: Color model (int or None, default: None)
            None = auto-detect from image data
            Rarely needs manual setting

        resolution: Resolution in DPI (tuple of 2 floats or None, default: None)
            (horizontal_dpi, vertical_dpi)
            None = no resolution metadata
    """
    lossless: bool              # True = lossless (level=100)
    quality: int                # 1-100, quality for lossy mode
    photometric: Optional[int]  # Color model (None = auto)
    resolution: Optional[tuple[float, float]]  # DPI (None = no resolution)


class JXRSaveOptionsAdapter(SaveOptionsAdapter):
    """Adapter for JPEG XR save options"""

    def get_supported_options(self) -> set[str]:
        """JXR supports lossless, quality, photometric, and resolution"""
        return {'lossless', 'quality', 'photometric', 'resolution'}

    def validate(self, options: SaveOptions) -> JXRSaveOptions:
        """
        Validate and normalize JPEG XR save options

        Args:
            options: User-provided save options

        Returns:
            Dictionary with validated JXR-specific options

        Raises:
            ValueError: If option values are invalid
            TypeError: If option types are incorrect
        """
        validated: JXRSaveOptions = {}

        # Warn about unsupported options
        self._warn_unsupported(options, 'JXR')

        # lossless (bool) - check first as it affects quality handling
        if 'lossless' in options:
            lossless = options['lossless']

            if not isinstance(lossless, bool):
                raise TypeError(
                    f"lossless must be bool, got {type(lossless).__name__}"
                )

            validated['lossless'] = lossless
        else:
            validated['lossless'] = True  # Default: lossless

        # quality (1-100) - only used for lossy mode
        if 'quality' in options:
            quality = options['quality']

            if not isinstance(quality, int):
                raise TypeError(
                    f"quality must be int, got {type(quality).__name__}"
                )

            if not 1 <= quality <= 100:
                raise ValueError(
                    f"quality must be in range [1, 100], got {quality}"
                )

            validated['quality'] = quality
        else:
            validated['quality'] = 90  # Default

        # photometric (int or None) - color model
        if 'photometric' in options:
            photometric = options['photometric']

            if photometric is not None and not isinstance(photometric, int):
                raise TypeError(
                    f"photometric must be int or None, got {type(photometric).__name__}"
                )

            validated['photometric'] = photometric
        # else: Don't set default - let imagecodecs auto-detect

        # resolution (tuple of 2 floats or None)
        if 'resolution' in options:
            resolution = options['resolution']

            if resolution is not None:
                if not isinstance(resolution, (tuple, list)):
                    raise TypeError(
                        f"resolution must be tuple/list or None, got {type(resolution).__name__}"
                    )

                if len(resolution) != 2:
                    raise ValueError(
                        f"resolution must have 2 values (horizontal, vertical), got {len(resolution)}"
                    )

                try:
                    h_dpi = float(resolution[0])
                    v_dpi = float(resolution[1])

                    if h_dpi <= 0 or v_dpi <= 0:
                        raise ValueError(
                            f"resolution DPI values must be positive, got {resolution}"
                        )

                    validated['resolution'] = (h_dpi, v_dpi)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid resolution values: {e}")
        # else: Don't set default - no resolution metadata

        return validated
