"""
TIFF format save options adapter
"""
from typing import TypedDict, Literal
from image_pipeline.io.formats.base import SaveOptionsAdapter
from image_pipeline.types import SaveOptions


# TIFF compression schemes (from TIFF 6.0 spec)
TiffCompression = Literal['none', 'lzw', 'deflate', 'zstd', 'jpeg']


class TiffSaveOptions(TypedDict, total=False):
    """
    TIFF-specific save options

    Options:
        compression: Compression scheme
            'none' - No compression (Tag 259 = 1)
            'lzw' - LZW compression (Tag 259 = 5)
            'deflate' - Deflate/ZIP compression (Tag 259 = 8)
            'zstd' - ZSTD compression (Tag 259 = 50000)
            'jpeg' - JPEG compression (Tag 259 = 7, lossy)

        quality: JPEG quality (1-100, only for compression='jpeg')
            Higher = better quality, larger file size

        compression_level: Compression level (0-9, for deflate/zstd)
            0 = fastest, worst compression
            9 = slowest, best compression
    """
    compression: TiffCompression
    quality: int              # 1-100 (for JPEG compression)
    compression_level: int    # 0-9 (for deflate/zstd)


class TiffOptionsAdapter(SaveOptionsAdapter):
    """Adapter for TIFF save options"""

    def get_supported_options(self) -> set[str]:
        """TIFF supports compression, quality, and compression_level"""
        return {'compression', 'quality', 'compression_level'}

    def validate(self, options: SaveOptions) -> TiffSaveOptions:
        """
        Validate and normalize TIFF save options

        Args:
            options: User-provided save options

        Returns:
            Dictionary with validated TIFF-specific options

        Raises:
            ValueError: If option values are invalid
            TypeError: If option types are incorrect
        """
        validated: TiffSaveOptions = {}

        # Warn about unsupported options
        self._warn_unsupported(options, 'TIFF')

        # compression (default: none for lossless)
        if 'compression' in options:
            compression = options['compression']

            # Allow None as equivalent to 'none'
            if compression is None:
                compression = 'none'

            if not isinstance(compression, str):
                raise TypeError(
                    f"compression must be str or None, got {type(compression).__name__}"
                )

            valid_compressions = {'none', 'lzw', 'deflate', 'zstd', 'jpeg'}
            if compression not in valid_compressions:
                raise ValueError(
                    f"compression must be one of {valid_compressions}, got '{compression}'"
                )

            validated['compression'] = compression  # type: ignore
        else:
            validated['compression'] = 'none'  # Default: uncompressed

        # quality (only for JPEG compression)
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

            # Warn if quality is specified but compression is not JPEG
            if validated.get('compression') != 'jpeg':
                import warnings
                warnings.warn(
                    f"quality option is only used with compression='jpeg', "
                    f"but compression='{validated.get('compression')}'. "
                    f"quality will be ignored.",
                    UserWarning
                )

            validated['quality'] = quality

        # compression_level (for deflate/zstd)
        if 'compression_level' in options:
            level = options['compression_level']

            if not isinstance(level, int):
                raise TypeError(
                    f"compression_level must be int, got {type(level).__name__}"
                )

            if not 0 <= level <= 9:
                raise ValueError(
                    f"compression_level must be in range [0, 9], got {level}"
                )

            # Warn if compression_level is specified for wrong compression type
            comp = validated.get('compression')
            if comp not in ('deflate', 'zstd'):
                import warnings
                warnings.warn(
                    f"compression_level is only used with compression='deflate' or 'zstd', "
                    f"but compression='{comp}'. compression_level will be ignored.",
                    UserWarning
                )

            validated['compression_level'] = level

        return validated
