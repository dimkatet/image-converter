"""
Module for reading images of various formats
"""
import numpy as np
import imageio.v3 as iio
import os
from pathlib import Path
import tifffile

from image_pipeline.core.image_data import ImageData


class ImageReader:
    """Class for reading images of various formats"""
    
    TIFF_FORMATS = {'.tiff', '.tif'}
    IMAGEIO_FORMATS = {
        '.jpg', '.jpeg', '.png', '.bmp', '.gif', 
        '.webp', '.ico', '.exr', '.hdr', '.pfm'
    }
    SUPPORTED_FORMATS = TIFF_FORMATS | IMAGEIO_FORMATS
    
    def __init__(self, filepath: str):
        """
        Initialize image reader
        
        Args:
            filepath: Path to the image file
        """
        self.filepath = Path(filepath)
        self._validate_file()
    
    def _validate_file(self) -> None:
        """Check file existence and format"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        if not self.filepath.is_file():
            raise ValueError(f"Path is not a file: {self.filepath}")
        
        ext = self.filepath.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {ext}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
    
    def read(self) -> ImageData:
        """
        Read image from file
        
        Returns:
            ImageData object with pixels and metadata
        """
        ext = self.filepath.suffix.lower()
        
        if ext in self.TIFF_FORMATS:
            return self._read_tiff()
        else:
            return self._read_imageio()
    
    def _read_tiff(self) -> ImageData:
        """Read TIFF files using tifffile"""
        try:
            pixels = tifffile.imread(self.filepath)
            
            with tifffile.TiffFile(self.filepath) as tif:
                metadata = {
                    'format': 'TIFF',
                    'filename': self.filepath.name,
                    'file_size': os.path.getsize(self.filepath),
                    'bit_depth': pixels.dtype.itemsize * 8,
                    'is_float': np.issubdtype(pixels.dtype, np.floating),
                    'pages': len(tif.pages),
                    'shape': pixels.shape,
                    'dtype': str(pixels.dtype),
                }
                
                if tif.pages:
                    page = tif.pages[0]
                    metadata['compression'] = page.compression.name if hasattr(page, 'compression') else 'unknown'
                    metadata['photometric'] = page.photometric.name if hasattr(page, 'photometric') else 'unknown'
                    
                    if hasattr(page, 'description') and page.description:
                        metadata['description'] = page.description
            
            return ImageData(pixels, metadata)
            
        except Exception as e:
            raise IOError(f"Error reading TIFF file: {e}")
    
    def _read_imageio(self) -> ImageData:
        """Read images using imageio"""
        try:
            # Read image
            pixels = iio.imread(self.filepath)
            
            # Get metadata
            props = iio.improps(self.filepath)
            
            metadata = {
                'format': self.filepath.suffix.upper().lstrip('.'),
                'filename': self.filepath.name,
                'file_size': os.path.getsize(self.filepath),
                'shape': pixels.shape,
                'dtype': str(pixels.dtype),
                'bit_depth': pixels.dtype.itemsize * 8,
                'is_float': np.issubdtype(pixels.dtype, np.floating),
            }
            
            # Add properties from imageio
            if props.shape:
                metadata['original_shape'] = props.shape
            if props.n_images:
                metadata['n_images'] = props.n_images
            if props.is_batch:
                metadata['is_batch'] = props.is_batch
            
            # Check transparency
            if len(pixels.shape) == 3:
                channels = pixels.shape[2]
                metadata['channels'] = channels
                metadata['has_transparency'] = channels in (2, 4)  # LA or RGBA
            elif len(pixels.shape) == 2:
                metadata['channels'] = 1
                metadata['has_transparency'] = False
            
            # Try to get EXIF from imageio metadata
            try:
                meta = iio.immeta(self.filepath)
                if meta:
                    metadata['imageio_meta'] = meta
            except:
                pass
            
            return ImageData(pixels, metadata)
            
        except Exception as e:
            raise IOError(f"Error reading image via imageio: {e}")