"""
Filter registry for CLI
Maps CLI filter names to filter classes
"""
from typing import Dict, Type
from image_pipeline.filters import (
    RemoveAlphaFilter,
    NormalizeFilter,
    PQEncodeFilter,
    PQDecodeFilter,
    GrayscaleFilter,
    QuantizeFilter,
    DequantizeFilter,
    SharpenFilter,
    BlurFilter,
    ColorConvertFilter,
)
from image_pipeline.filters.base import ImageFilter


FILTER_REGISTRY: Dict[str, Type[ImageFilter]] = {
    'remove_alpha': RemoveAlphaFilter,
    'normalize': NormalizeFilter,
    'pq_encode': PQEncodeFilter,
    'pq_decode': PQDecodeFilter,
    'grayscale': GrayscaleFilter,
    'quantize': QuantizeFilter,
    'dequantize': DequantizeFilter,
    'sharpen': SharpenFilter,
    'blur': BlurFilter,
    'color_convert': ColorConvertFilter,
}


def get_available_filters() -> str:
    """
    Get formatted string of available filters
    
    Returns:
        String with filter names and their classes
    """
    lines = ["Available filters:"]
    for name in sorted(FILTER_REGISTRY.keys()):
        filter_class = FILTER_REGISTRY[name]
        lines.append(f"  • {name:20} -> {filter_class.__name__}")
    return "\n".join(lines)