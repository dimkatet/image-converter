from typing import Optional, Union
import numpy as np

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.writer import ImageWriter

class ImageSaver:
    """Helper class for batch save operations"""
    
    @staticmethod
    def save_with_format_conversion(data: Union['ImageData', np.ndarray],
                                    output_path: str,
                                    target_dtype: Optional[np.dtype] = None,
                                    **save_options) -> None:
        """
        Save with data type conversion
        
        Args:
            data: ImageData or numpy array
            output_path: Path to save
            target_dtype: Target data type (uint8, uint16, float32, etc.)
            **save_options: Additional save options
        """
        # Extract pixels
        if hasattr(data, 'pixels'):
            pixels = data.pixels
            metadata = data.metadata
        else:
            pixels = data
            metadata = {}
        
        # Convert data type if needed
        if target_dtype and pixels.dtype != target_dtype:
            pixels = ImageSaver._convert_dtype(pixels, target_dtype)
        
        # Save
        writer = ImageWriter(output_path)
        writer.write(pixels, metadata=metadata, **save_options)
    
    @staticmethod
    def _convert_dtype(pixels: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
        """Convert data type with proper scaling"""
        source_dtype = pixels.dtype
        
        # If types are the same, return as is
        if source_dtype == target_dtype:
            return pixels
        
        # float -> uint
        if np.issubdtype(source_dtype, np.floating) and \
           np.issubdtype(target_dtype, np.unsignedinteger):
            pix_min = pixels.min()
            pix_max = pixels.max()
            
            if pix_max > pix_min:
                normalized = (pixels - pix_min) / (pix_max - pix_min)
            else:
                normalized = np.zeros_like(pixels)
            
            max_val = np.iinfo(target_dtype).max
            return (normalized * max_val).astype(target_dtype)
        
        # uint -> float
        elif np.issubdtype(source_dtype, np.unsignedinteger) and \
             np.issubdtype(target_dtype, np.floating):
            max_val = np.iinfo(source_dtype).max
            return (pixels.astype(target_dtype) / max_val)
        
        # uint -> uint of different bit depth
        elif np.issubdtype(source_dtype, np.unsignedinteger) and \
             np.issubdtype(target_dtype, np.unsignedinteger):
            source_max = np.iinfo(source_dtype).max
            target_max = np.iinfo(target_dtype).max
            return (pixels.astype(np.float64) / source_max * target_max).astype(target_dtype)
        
        # For all other cases just cast the type
        return pixels.astype(target_dtype)
