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
    # === Базовые поля (устанавливаются автоматически ImageData) ===
    shape: tuple[int, ...]
    dtype: str
    channels: int
    bit_depth: int
    
    # === Информация о файле ===
    format: str              # 'PNG', 'TIFF', 'EXR', etc.
    filename: str            # Original filename
    file_size: int           # File size in bytes
    
    # === Поля от фильтров ===
    transfer_function: TransferFunction
    color_space: Optional[ColorSpace]
    color_primaries: Optional[dict[str, tuple[float, float]]]  # {'red': (x, y), 'green': ..., 'blue': ..., 'white': ...}
    
    # === HDR параметры (опциональные) ===
    peak_luminance: float      # nits
    min_luminance: float       # nits
    max_cll: int              # Maximum Content Light Level, nits
    max_fall: int             # Maximum Frame Average Light Level, nits
    
    # === Текстовые метаданные ===
    text: dict[str, str]      # Произвольные key-value пары для tEXt chunks
