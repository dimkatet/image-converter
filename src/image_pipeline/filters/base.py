from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class ImageFilter(ABC):
    """Base abstract class for all filters"""
    
    def __init__(self, name: Optional[str] = None):
        """
        Filter initialization
        
        Args:
            name: Filter name (optional)
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        """
        Apply the filter to a pixel array
        
        Args:
            pixels: Input pixel array
            
        Returns:
            Processed pixel array
        """
        pass
    
    def validate(self, pixels: np.ndarray) -> None:
        """
        Validate input data
        
        Args:
            pixels: Pixel array to check
            
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(pixels, np.ndarray):
            raise ValueError(f"{self.name}: input data must be a numpy array")
        
        if pixels.size == 0:
            raise ValueError(f"{self.name}: empty pixel array")
    
    def __repr__(self) -> str:
        return f"{self.name}()"
