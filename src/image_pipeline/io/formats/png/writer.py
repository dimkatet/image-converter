"""PNG format writer"""
import numpy as np
import imageio.v3 as iio
import png as pypng

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatWriter


class PNGFormatWriter(FormatWriter):
    """Writer for PNG images"""
    
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
    
    def write_pixels(self, img_data: ImageData, compression_level: int = 6, **options) -> None:
        """
        Write PNG pixel data
        
        Args:
            img_data: ImageData with pixels
            compression_level: Compression level (0-9), default 6
            **options: Additional options (ignored)
        """
        pixels = img_data.pixels
        
        try:
            if pixels.dtype == np.uint8:
                self._write_uint8(pixels, compression_level)
            elif pixels.dtype == np.uint16:
                self._write_uint16(pixels)
            else:
                raise ValueError(f"PNG does not support type {pixels.dtype}")
                
        except Exception as e:
            raise IOError(f"Error writing PNG: {e}")
    
    def _write_uint8(self, pixels: np.ndarray, compression_level: int) -> None:
        """Write uint8 PNG using imageio"""
        iio.imwrite(
            self.filepath,
            pixels,
            compress_level=compression_level,
            optimize=True
        )
    
    def _write_uint16(self, pixels: np.ndarray) -> None:
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
            compression=9
        )
        
        # Write to file
        with open(self.filepath, 'wb') as f:
            writer.write(f, img_data)
