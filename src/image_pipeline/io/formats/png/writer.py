"""PNG format writer"""
from typing import Optional
import numpy as np
import imageio.v3 as iio
import png as pypng

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatWriter
from image_pipeline.io.formats.png.options import PNGOptionsAdapter, PNGSaveOptions
from image_pipeline.types import SaveOptions


class PNGFormatWriter(FormatWriter):
    """Writer for PNG images"""
    
    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.options_adapter = PNGOptionsAdapter()
    
    def validate(self, img_data: ImageData) -> None:
        """
        Validate that data is compatible with PNG format
        
        PNG supports: uint8, uint16
        """
        pixels = img_data.pixels
        
        if not isinstance(pixels, np.ndarray):
            raise ValueError("Data must be a numpy array")
        
        if pixels.size == 0:
            raise ValueError("Empty pixel array")
        
        if pixels.dtype not in (np.uint8, np.uint16):
            raise ValueError(
                f"PNG supports only uint8 and uint16. "
                f"Got: {pixels.dtype}.\n"
                f"Solutions:\n"
                f"  1. For float: use TIFF, EXR or HDR\n"
                f"  2. For uint32: convert to uint16 or use TIFF"
            )
    
    def write_pixels(self, img_data: ImageData, options: SaveOptions) -> None:
        """
        Write PNG pixel data
        
        Args:
            img_data: ImageData with pixels
            **options: Save options (will be validated by adapter)
        """
        pixels = img_data.pixels
        
        # Validate options
        validated_options = self.options_adapter.validate(options)
        
        try:
            if pixels.dtype == np.uint8:
                self._write_uint8(pixels, validated_options)
            elif pixels.dtype == np.uint16:
                self._write_uint16(pixels, validated_options)
            else:
                raise ValueError(f"PNG does not support type {pixels.dtype}")
                
        except Exception as e:
            raise IOError(f"Error writing PNG: {e}")
    
    def _write_uint8(self, pixels: np.ndarray, options: PNGSaveOptions) -> None:
        """Write uint8 PNG using imageio"""
        iio.imwrite(
            self.filepath,
            pixels,
            compress_level=options.get('compression_level'),
            optimize=options.get('optimize')
        )
    
    def _write_uint16(self, pixels: np.ndarray, options: PNGSaveOptions) -> None:
        """Write uint16 PNG using pypng"""
        height, width = pixels.shape[:2]
        
        # Determine image type
        if len(pixels.shape) == 2:
            # Grayscale
            greyscale = True
            alpha = False
            img_data = pixels
        
        elif len(pixels.shape) == 3:
            channels = pixels.shape[2]
            
            if channels == 1:
                greyscale = True
                alpha = False
                img_data = pixels.squeeze(-1)
            
            elif channels == 2:
                # Grayscale + Alpha
                greyscale = True
                alpha = True
                img_data = pixels.reshape(height, width * 2)
            
            elif channels == 3:
                # RGB
                greyscale = False
                alpha = False
                img_data = pixels.reshape(height, width * 3)
            
            elif channels == 4:
                # RGBA
                greyscale = False
                alpha = True
                img_data = pixels.reshape(height, width * 4)
            
            else:
                raise ValueError(f"Unsupported number of channels: {channels}")
        
        else:
            raise ValueError(f"Unsupported array shape: {pixels.shape}")
        
        # Create PNG writer
        writer = pypng.Writer(
            width=width,
            height=height,
            greyscale=greyscale, # type: ignore
            alpha=alpha,
            bitdepth=16,
            compression=options.get('compression_level')
        )
        
        # Write to file
        with open(self.filepath, 'wb') as f:
            writer.write(f, img_data)
