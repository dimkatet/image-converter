"""
Module for reading images of various formats
"""
import numpy as np
from copy import deepcopy
from typing import Dict, Any, Optional


class ImageData:
    """Контейнер для пикселей + метаданные"""
    
    def __init__(self, pixels: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        self.pixels = pixels
        self.metadata = metadata or {}
        self._sync_metadata()
    
    def _sync_metadata(self):
        """Синхронизация базовых метаданных с реальным состоянием пикселей"""
        self.metadata['shape'] = self.pixels.shape
        self.metadata['dtype'] = str(self.pixels.dtype)
        self.metadata['channels'] = self.channels
        self.metadata['bit_depth'] = self.pixels.dtype.itemsize * 8
    
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
