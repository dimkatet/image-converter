"""
JPEG format support with automatic delegation

This module provides JPEG format support with automatic delegation to:
    - Ultra HDR: For JPEG Ultra HDR (ISO/TS 21496-1) files
    - Standard JPEG: For regular JPEG files (TODO: not implemented)

The JPEGReader and JPEGWriter facades automatically detect the format
and delegate to the appropriate implementation.

Usage:
    # Writing Ultra HDR (scene-referred LINEAR data)
    python main.py input.tiff output.jpg \\
        --ultra-hdr \\
        --filter remove_alpha \\
        --filter color_convert:source=bt709,target=bt2020 \\
        --option quality=95

    IMPORTANT: Do NOT use AbsoluteLuminanceFilter or PQEncodeFilter
               libultrahdr handles encoding internally

    # Reading (automatic detection)
    python main.py input_ultrahdr.jpg output.png
"""

from .reader import JPEGReader
from .writer import JPEGWriter
from .options import JPEGSaveOptionsAdapter

__all__ = [
    'JPEGReader',
    'JPEGWriter',
    'JPEGSaveOptionsAdapter',
]
