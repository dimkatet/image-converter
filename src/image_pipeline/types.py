"""
Type definitions for image pipeline
"""
from enum import Enum
from typing import TypedDict, Optional, Any


class TransferFunction(Enum):
    """Transfer function (EOTF/OETF) types"""
    LINEAR = 'linear'
    SRGB = 'sRGB'
    PQ = 'PQ'        # ST.2084 / BT.2100 PQ
    HLG = 'HLG'      # BT.2100 HLG


class ColorSpace(Enum):
    """Color space / color primaries"""
    BT709 = 'BT.709'
    BT2020 = 'BT.2020'
    DISPLAY_P3 = 'Display-P3'


class ImageMetadata(TypedDict, total=False):
    """
    Typed dictionary for ImageData.metadata
    All fields are optional except those set automatically by ImageData
    """
    # === Base fields (set automatically by ImageData) ===
    shape: tuple[int, ...]
    dtype: str
    channels: int
    bit_depth: int

    # === File information ===
    format: str              # 'PNG', 'TIFF', 'EXR', etc.
    filename: str            # Original filename
    file_size: int           # File size in bytes

    # === Fields from filters ===
    transfer_function: TransferFunction
    color_space: Optional[ColorSpace]
    color_primaries: Optional[dict[str, tuple[float, float]]]  # {'red': (x, y), 'green': ..., 'blue': ..., 'white': ...}

    # === HDR parameters (optional) ===
    # Scene reference:
    paper_white: float                          # Reference white level, nits (describes the scene's white point)

    # Mastering display (from mDCv chunk or manual):
    mastering_display_max_luminance: float      # Reference display peak brightness, nits
    mastering_display_min_luminance: float      # Reference display black level, nits

    # Content light levels (computed or from cLLi chunk, for HDR10 signaling):
    max_cll: int                                # Maximum Content Light Level, nits (display-referred only)
    max_fall: int                               # Maximum Frame Average Light Level, nits (display-referred only)

    # === Text metadata ===
    text: dict[str, str]      # Arbitrary key-value pairs for tEXt chunks


class SaveOptions(TypedDict, total=False):
    """
    Common save options for all image formats
    
    Not all formats support all options - format-specific adapters
    will validate and filter options for each format.
    
    Supported formats and their options:
    
    PNG:
        - compression_level (0-9, default: 6)
        - optimize (bool, default: False)
    
    JPEG:
        - quality (1-100, default: 90)
        - optimize (bool, default: False)
        - progressive (bool, default: False)
        - subsampling ('4:4:4', '4:2:2', '4:2:0')
    
    TIFF:
        - compression ('none', 'lzw', 'jpeg', 'deflate')
        - quality (1-100, for JPEG compression)
    
    WebP:
        - quality (1-100, default: 90)
        - lossless (bool, default: False)
        - method (0-6, speed vs size, default: 4)
    
    AVIF:
        - quality (1-100, default: 90)
        - lossless (bool, default: False)
        - speed (0-10, faster vs better, default: 6)
        - bit_depth (8, 10, 12)

    JPEG Ultra HDR:
        - ultra_hdr (bool, default: False) - Enable Ultra HDR encoding
        - quality (1-100, default: 95)
        - gainmap_scale (int, default: 4) - Gainmap downscale factor
    """
    # Compression/Quality
    quality: int              # 1-100, for lossy formats (JPEG, WebP, AVIF)
    compression_level: int    # 0-9, for PNG
    compression: str          # Compression type for TIFF ('none', 'lzw', 'jpeg', etc.)

    # Format-specific flags
    optimize: bool            # Optimize encoding (JPEG, PNG)
    progressive: bool         # Progressive encoding (JPEG)
    lossless: bool           # Lossless mode (WebP, AVIF)
    ultra_hdr: bool          # Enable JPEG Ultra HDR encoding (JPEG only)

    # Advanced
    method: int              # Encoding method/speed (WebP: 0-6, AVIF: 0-10)
    speed: int               # Encoding speed for AVIF (0-10)
    bit_depth: int           # Bit depth for output (AVIF: 8, 10, 12)
    subsampling: str         # Chroma subsampling (JPEG: '4:4:4', '4:2:2', '4:2:0')
    gainmap_scale: int       # Gainmap downscale factor for JPEG Ultra HDR (default: 4)
