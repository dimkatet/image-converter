"""
Module for reading images of various formats
"""
import numpy as np
from copy import deepcopy
from typing import Optional

from image_pipeline.types import ImageMetadata


class ImageData:
    """Контейнер для пикселей + метаданные"""
    
    def __init__(self, pixels: np.ndarray, metadata: Optional[ImageMetadata] = None):
        self.pixels = pixels
        self.metadata: ImageMetadata = metadata or {}

        # Set bit_depth ONCE if not already specified
        # This preserves explicit bit_depth values (e.g., 10-bit, 12-bit in uint16)
        if 'bit_depth' not in self.metadata:
            self.metadata['bit_depth'] = self.pixels.dtype.itemsize * 8

        self._sync_metadata()

    def _sync_metadata(self):
        """
        Sync basic metadata with pixel array state

        Note: bit_depth is NOT synced here to preserve explicit values
        (e.g., 10-bit or 12-bit data stored in uint16 arrays)
        """
        self.metadata['shape'] = self.pixels.shape
        self.metadata['dtype'] = str(self.pixels.dtype)
        self.metadata['channels'] = self.channels
        # bit_depth is NOT auto-synced - it's set once in __init__ if missing
    
    @property
    def shape(self) -> tuple:
        return self.pixels.shape
    
    @property
    def dtype(self) -> np.dtype:
        return self.pixels.dtype
    
    @property
    def width(self) -> int:
        return self.pixels.shape[1]
    
    @property
    def height(self) -> int:
        return self.pixels.shape[0]
    
    @property
    def channels(self) -> int:
        return self.pixels.shape[2] if len(self.pixels.shape) > 2 else 1
    
    @property
    def format(self) -> str:
        return self.metadata.get('format', 'unknown')
    
    def copy(self) -> 'ImageData':
        """Глубокая копия"""
        return ImageData(
            pixels=self.pixels.copy(),
            metadata=deepcopy(self.metadata)
        )
    
    def __repr__(self) -> str:
        return (f"ImageData(shape={self.shape}, dtype={self.dtype}, "
                f"channels={self.channels})")
