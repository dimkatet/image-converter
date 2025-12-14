from abc import ABC, abstractmethod
from typing import Optional
import warnings

import numpy as np

from image_pipeline.core.image_data import ImageData


class ImageFilter(ABC):
    """Base abstract class for all filters"""
    
    def __init__(self, name: Optional[str] = None):
        """
        Filter initialization
        
        Args:
            name: Filter name (optional)
        """
        self.name = name or self.__class__.__name__
    
    def __call__(self, img_data: ImageData) -> ImageData:
        """Apply the filter"""
        result = img_data.copy()
        result.pixels = self.apply(result.pixels)
        self.update_metadata(result)
        return result
    
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
    
    def update_metadata(self, img_data: ImageData) -> None:
        """
        Update metadata after processing
        
        Args:
            img_data: ImageData object to update
        
        Returns:     
            None
        """
        img_data._sync_metadata()
    
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
    
    def _check_dtype(self, pixels: np.ndarray, allowed_dtypes: list) -> None:
        """
        Check if pixel dtype is in allowed list
        
        Args:
            pixels: Pixel array
            allowed_dtypes: List of allowed numpy dtypes
            
        Raises:
            ValueError: If dtype is not allowed
        """
        if pixels.dtype not in allowed_dtypes:
            allowed_str = ", ".join(str(dt) for dt in allowed_dtypes)
            raise ValueError(
                f"{self.name}: requires dtype in [{allowed_str}], "
                f"got {pixels.dtype}"
            )
    
    def _check_range(self, pixels: np.ndarray, min_val: float, max_val: float) -> None:
        """
        Check if pixel values are within specified range
        
        Args:
            pixels: Pixel array
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Raises:
            ValueError: If values are outside the range
        """
        actual_min = pixels.min()
        actual_max = pixels.max()
        
        if actual_min < min_val or actual_max > max_val:
            raise ValueError(
                f"{self.name}: expects values in [{min_val}, {max_val}], "
                f"got [{actual_min:.6f}, {actual_max:.6f}]"
            )
    
    def _check_positive(self, pixels: np.ndarray) -> None:
        """
        Check if all pixel values are non-negative
        
        Args:
            pixels: Pixel array
            
        Raises:
            ValueError: If any values are negative
        """
        if pixels.min() < 0:
            raise ValueError(
                f"{self.name}: input contains negative values "
                f"(min={pixels.min():.6f}), expected non-negative values"
            )
    
    def __repr__(self) -> str:
        return f"{self.name}()"
