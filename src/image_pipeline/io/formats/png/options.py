"""
PNG format save options adapter
"""
from typing import TypedDict
from pathlib import Path
from image_pipeline.io.formats.base import SaveOptionsAdapter
from image_pipeline.types import SaveOptions

class PNGSaveOptions(TypedDict, total=False):
    """
    PNG-specific save options

    Options:
        compression_level: Compression level (0-9, default: 6)
            0 = no compression (fast)
            9 = maximum compression (slow)

        strategy: Compression strategy (0-4, default: 0)
            0 = Z_DEFAULT_STRATEGY (best for most images)
            1 = Z_FILTERED (for filtered data)
            2 = Z_HUFFMAN_ONLY (fast, poor compression)
            3 = Z_RLE (good for images with large uniform areas)
            4 = Z_FIXED (fastest, worst compression)

        icc_profile: Path to ICC color profile file
    """
    compression_level: int  # 0 (no compression) to 9 (max compression)
    strategy: int           # zlib compression strategy (0-4)
    icc_profile: str        # Path to ICC profile file


class PNGOptionsAdapter(SaveOptionsAdapter):
    """Adapter for PNG save options"""

    def get_supported_options(self) -> set[str]:
        """PNG supports compression_level, strategy, and icc_profile"""
        return {'compression_level', 'strategy', 'icc_profile'}

    def validate(self, options: SaveOptions) -> PNGSaveOptions:
        """
        Validate and normalize PNG save options

        Args:
            options: User-provided save options

        Returns:
            Dictionary with validated PNG-specific options

        Raises:
            ValueError: If option values are invalid
            TypeError: If option types are incorrect
        """
        validated: PNGSaveOptions = {}

        # Warn about unsupported options
        self._warn_unsupported(options, 'PNG')

        # compression_level (0-9)
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

            validated['compression_level'] = level
        else:
            validated['compression_level'] = 6  # Default

        # strategy (0-4)
        if 'strategy' in options:
            strategy = options['strategy']

            if not isinstance(strategy, int):
                raise TypeError(
                    f"strategy must be int, got {type(strategy).__name__}"
                )

            if not 0 <= strategy <= 4:
                raise ValueError(
                    f"strategy must be in range [0, 4], got {strategy}\n"
                    f"Valid strategies:\n"
                    f"  0 = DEFAULT (best for most images)\n"
                    f"  1 = FILTERED\n"
                    f"  2 = HUFFMAN_ONLY (not recommended)\n"
                    f"  3 = RLE (fast for uniform areas)\n"
                    f"  4 = FIXED (not recommended)"
                )

            validated['strategy'] = strategy
        else:
            validated['strategy'] = 0  # Default to Z_DEFAULT_STRATEGY

        # icc_profile (path to ICC profile file)
        if 'icc_profile' in options:
            icc_path = options['icc_profile']

            if not isinstance(icc_path, str):
                raise TypeError(
                    f"icc_profile must be str (path), got {type(icc_path).__name__}"
                )

            # Check if file exists
            icc_file = Path(icc_path)
            if not icc_file.exists():
                raise FileNotFoundError(
                    f"ICC profile file not found: {icc_path}"
                )

            if not icc_file.is_file():
                raise ValueError(
                    f"ICC profile path is not a file: {icc_path}"
                )

            validated['icc_profile'] = icc_path

        return validated