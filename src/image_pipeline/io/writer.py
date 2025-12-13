"""
Module for saving images in various formats
"""
import numpy as np
import imageio.v3 as iio
import tifffile
from pathlib import Path
from typing import Optional, Union, Dict, Any
import warnings

try:
    import png as pypng
    HAS_PYPNG = True
except ImportError:
    HAS_PYPNG = False

from image_pipeline.core.image_data import ImageData


class ImageWriter:
    """Class for saving images to a file"""
    
    TIFF_FORMATS = {'.tiff', '.tif'}
    HDR_FORMATS = {'.exr', '.hdr', '.pfm'}
    PNG_FORMAT = {'.png'}
    UINT8_FORMATS = {'.jpg', '.jpeg', '.bmp', '.webp'}
    SUPPORTED_FORMATS = TIFF_FORMATS | HDR_FORMATS | PNG_FORMAT | UINT8_FORMATS
    
    # Available compression methods for TIFF
    TIFF_COMPRESSIONS = {
        'none': 0,
        'lzw': 5,
        'jpeg': 7,
        'deflate': 8,
        'zstd': 50000,
    }
    
    def __init__(self, filepath: str):
        """
        Initialize the writer
        
        Args:
            filepath: Path to save the file
        """
        self.filepath = Path(filepath)
        self._validate_format()
    
    def _validate_format(self) -> None:
        """Check if the format is supported"""
        ext = self.filepath.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {ext}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
    
    def write(self, 
              data: Union['ImageData', np.ndarray],
              quality: int = 95,
              compression: str = 'lzw',
              compression_level: int = 6,
              metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save image to file
        
        Args:
            data: ImageData object or numpy array with pixels
            quality: Quality for JPEG/WebP (1-100), default 95
            compression: Compression type for TIFF ('none', 'lzw', 'jpeg', 'deflate', 'zstd')
            compression_level: Compression level for PNG (0-9), default 6
            metadata: Additional metadata to save
        """
        # Extract pixels and metadata
        if hasattr(data, 'pixels'):
            pixels = data.pixels
            saved_metadata = metadata or data.metadata
        else:
            pixels = data
            saved_metadata = metadata or {}
        
        # Validate data
        self._validate_data(pixels)
        
        # Create directory if it does not exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        ext = self.filepath.suffix.lower()
        
        if ext in self.TIFF_FORMATS:
            self._write_tiff(pixels, compression, saved_metadata)
        elif ext in self.PNG_FORMAT:
            self._write_png(pixels, compression_level, saved_metadata)
        elif ext in self.HDR_FORMATS:
            self._write_hdr(pixels, saved_metadata)
        else:
            self._write_uint8(pixels, quality, saved_metadata)
    
    def _validate_data(self, pixels: np.ndarray) -> None:
        """
        Validate data before saving
        
        Args:
            pixels: Pixel array
            
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(pixels, np.ndarray):
            raise ValueError("Data must be a numpy array")
        
        if pixels.size == 0:
            raise ValueError("Empty pixel array")
        
        ext = self.filepath.suffix.lower()
        
        # PNG supports uint8 and uint16
        if ext == '.png':
            if pixels.dtype not in (np.uint8, np.uint16):
                raise ValueError(
                    f"PNG supports only uint8 and uint16. "
                    f"Got: {pixels.dtype}.\n"
                    f"Solutions:\n"
                    f"  1. For float: use TIFF, EXR or HDR\n"
                    f"  2. For uint32: convert to uint16 or use TIFF"
                )
        
        # Other uint8 formats
        elif ext in self.UINT8_FORMATS:
            if pixels.dtype != np.uint8:
                raise ValueError(
                    f"{ext.upper()} supports only uint8. "
                    f"Got: {pixels.dtype}.\n"
                    f"Solutions:\n"
                    f"  1. Use QuantizeFilter(bit_depth=8) for conversion\n"
                    f"  2. For uint16: save as PNG or TIFF\n"
                    f"  3. For float32: save as TIFF, EXR or HDR"
                )
            
            # JPEG does not support transparency
            if ext in {'.jpg', '.jpeg'}:
                if len(pixels.shape) == 3 and pixels.shape[2] == 4:
                    raise ValueError(
                        "JPEG does not support transparency (RGBA). "
                        "Use PNG or convert to RGB."
                    )
        
        # HDR formats require float
        elif ext in self.HDR_FORMATS:
            if not np.issubdtype(pixels.dtype, np.floating):
                raise ValueError(
                    f"{ext.upper()} requires float data. "
                    f"Got: {pixels.dtype}. "
                    f"Convert to float32 before saving."
                )
        
        # Check range for integer types
        if np.issubdtype(pixels.dtype, np.integer):
            pix_min = pixels.min()
            pix_max = pixels.max()
            
            if pix_min < 0:
                raise ValueError(
                    f"Negative pixel values ({pix_min}) are not allowed"
                )
            
            dtype_max = np.iinfo(pixels.dtype).max
            if pix_max > dtype_max:
                raise ValueError(
                    f"Pixel values ({pix_max}) exceed maximum for {pixels.dtype} ({dtype_max})"
                )
    
    def _write_tiff(self, 
                    pixels: np.ndarray, 
                    compression: str,
                    metadata: Dict[str, Any]) -> None:
        """
        Save as TIFF using tifffile
        Supports: uint8, uint16, uint32, float32, float64
        """
        try:
            if compression not in self.TIFF_COMPRESSIONS:
                warnings.warn(
                    f"Compression '{compression}' is not supported, using 'lzw'",
                    UserWarning
                )
                compression = 'lzw'
            
            tiff_metadata = {}
            if 'description' in metadata:
                tiff_metadata['description'] = str(metadata['description'])
            
            tifffile.imwrite(
                self.filepath,
                pixels,
                compression=compression,
                metadata=tiff_metadata
            )
            
        except Exception as e:
            raise IOError(f"Error saving TIFF: {e}")
    
    def _write_png(self,
                   pixels: np.ndarray,
                   compression_level: int,
                   metadata: Dict[str, Any]) -> None:
        """
        Save PNG with uint16 support
        Uses pypng for uint16, imageio for uint8
        """
        try:
            # uint8 - use imageio (faster)
            if pixels.dtype == np.uint8:
                self._write_png_imageio(pixels, compression_level)
            
            # uint16 - use pypng (the only library with proper support)
            elif pixels.dtype == np.uint16:
                if not HAS_PYPNG:
                    raise ImportError(
                        "Saving uint16 PNG requires the pypng library.\n"
                        "Install: pip install pypng"
                    )
                self._write_png_pypng(pixels)
            
            else:
                raise ValueError(f"PNG does not support type {pixels.dtype}")
                
        except Exception as e:
            raise IOError(f"Error saving PNG: {e}")
    
    def _write_png_imageio(self, pixels: np.ndarray, compression_level: int) -> None:
        """Save uint8 PNG using imageio"""
        iio.imwrite(
            self.filepath,
            pixels,
            compress_level=compression_level,
            optimize=True
        )
    
    def _write_png_pypng(self, pixels: np.ndarray) -> None:
        """Save uint16 PNG using pypng"""
        height, width = pixels.shape[:2]
        
        # Determine image type
        if len(pixels.shape) == 2:
            # Grayscale
            greyscale = True
            alpha = False
            planes = 1
            # pypng requires 2D array for grayscale
            img_data = pixels
        
        elif len(pixels.shape) == 3:
            channels = pixels.shape[2]
            
            if channels == 1:
                # Grayscale with one channel
                greyscale = True
                alpha = False
                planes = 1
                img_data = pixels.squeeze(-1)
            
            elif channels == 2:
                # Grayscale + Alpha
                greyscale = True
                alpha = True
                planes = 2
                # Convert to 2D array of rows
                img_data = pixels.reshape(height, width * 2)
            
            elif channels == 3:
                # RGB
                greyscale = False
                alpha = False
                planes = 3
                # Convert to 2D array of rows
                img_data = pixels.reshape(height, width * 3)
            
            elif channels == 4:
                # RGBA
                greyscale = False
                alpha = True
                planes = 4
                # Convert to 2D array of rows
                img_data = pixels.reshape(height, width * 4)
            
            else:
                raise ValueError(f"Unsupported number of channels: {channels}")
        
        else:
            raise ValueError(f"Unsupported array shape: {pixels.shape}")
        
        # Create PNG writer
        writer = pypng.Writer(
            width=width,
            height=height,
            greyscale=greyscale,
            alpha=alpha,
            bitdepth=16,  # uint16 = 16 bit
            compression=9  # maximum compression
        )
        
        # Save
        with open(self.filepath, 'wb') as f:
            # pypng requires a list of rows (each row is a 1D array)
            if len(img_data.shape) == 2 and planes > 1:
                # Already in (height, width*channels) format
                writer.write(f, img_data)
            else:
                # Grayscale 2D
                writer.write(f, img_data)
    
    def _write_hdr(self,
                   pixels: np.ndarray,
                   metadata: Dict[str, Any]) -> None:
        """
        Save HDR formats using imageio
        Supports: float32, float64
        """
        try:
            kwargs = {}
            
            # For EXR you can specify compression
            if self.filepath.suffix.lower() == '.exr':
                kwargs['compression'] = 'ZIP_COMPRESSION'
            
            iio.imwrite(self.filepath, pixels, **kwargs)
            
        except Exception as e:
            raise IOError(f"Error saving HDR format: {e}")
    
    def _write_uint8(self, 
                     pixels: np.ndarray,
                     quality: int,
                     metadata: Dict[str, Any]) -> None:
        """
        Save uint8 formats using imageio
        Supports: only uint8
        """
        try:
            ext = self.filepath.suffix.lower()
            kwargs = {}
            
            if ext in {'.jpg', '.jpeg'}:
                kwargs['quality'] = quality
                kwargs['optimize'] = True
            
            elif ext == '.webp':
                kwargs['quality'] = quality
                kwargs['method'] = 6
            
            iio.imwrite(self.filepath, pixels, **kwargs)
            
        except Exception as e:
            raise IOError(f"Error saving {ext.upper()}: {e}")