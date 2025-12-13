"""
Module for reading images of various formats
"""
import numpy as np
from typing import Dict, Any


class ImageData:
    """Class for storing image data and metadata"""
    
    def __init__(self, pixels: np.ndarray, metadata: Dict[str, Any]):
        """
        Initialize image data
        
        Args:
            pixels: Numpy array with image pixels
            metadata: Dictionary with metadata
        """
        self.pixels = pixels
        self.metadata = metadata
    
    @property
    def shape(self) -> tuple:
        """Shape of the pixel array (height, width, channels)"""
        return self.pixels.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Pixel data type"""
        return self.pixels.dtype
    
    @property
    def width(self) -> int:
        """Image width"""
        return self.pixels.shape[1]
    
    @property
    def height(self) -> int:
        """Image height"""
        return self.pixels.shape[0]
    
    @property
    def channels(self) -> int:
        """Number of channels"""
        return self.pixels.shape[2] if len(self.pixels.shape) > 2 else 1
    
    @property
    def format(self) -> str:
        """Source file format"""
        return self.metadata.get('format', 'unknown')
    
    def __repr__(self) -> str:
        return (f"ImageData(shape={self.shape}, dtype={self.dtype}, "
                f"format={self.format})")
