"""
Image reader facade with automatic format detection
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Type

from image_pipeline.core.image_data import ImageData


@dataclass
class FormatReaderConfig:
    """Configuration for a file format reader"""
    reader: Type  # FormatReader class


class ImageReader:
    """
    Facade for reading images with automatic format detection
    
    Automatically selects the appropriate format-specific reader
    based on file extension.
    """
    
    # Mapping of extensions to format reader configs
    _READERS = {}
    
    def __init__(self, filepath: str):
        """
        Args:
            filepath: Path to the image file
        """
        self.filepath = Path(filepath)
        self._reader = self._create_reader()
    
    def _create_reader(self):
        """Create appropriate format-specific reader"""
        ext = self.filepath.suffix.lower()
        
        config = self._READERS.get(ext)
        if not config:
            supported = ', '.join(sorted(self._READERS.keys()))
            raise ValueError(
                f"Unsupported format: {ext}. "
                f"Supported formats: {supported}"
            )
        
        return config.reader(self.filepath)
    
    def read(self) -> ImageData:
        """
        Read image from file
        
        Returns:
            ImageData object with pixels and metadata
        """
        return self._reader.read()
    
    @classmethod
    def register_format(cls, extensions: list, reader_class):
        """
        Register a format reader
        
        Args:
            extensions: List of file extensions (e.g., ['.png'])
            reader_class: FormatReader subclass
        """
        config = FormatReaderConfig(reader=reader_class)
        
        for ext in extensions:
            cls._READERS[ext.lower()] = config


# Import and register formats
from image_pipeline.io.formats.png import PNGFormatReader
from image_pipeline.io.formats.tiff.reader import TiffFormatReader
from image_pipeline.io.formats.avif import AVIFFormatReader
from image_pipeline.io.formats.webp import WebPFormatReader
from image_pipeline.io.formats.exr import EXRFormatReader
from image_pipeline.io.formats.jxr import JXRFormatReader

ImageReader.register_format(
    extensions=['.png'],
    reader_class=PNGFormatReader
)

ImageReader.register_format(
    extensions=['.tiff', '.tif'],
    reader_class=TiffFormatReader
)

ImageReader.register_format(
    extensions=['.avif'],
    reader_class=AVIFFormatReader
)

ImageReader.register_format(
    extensions=['.webp'],
    reader_class=WebPFormatReader
)

ImageReader.register_format(
    extensions=['.exr'],
    reader_class=EXRFormatReader
)

ImageReader.register_format(
    extensions=['.jxr', '.wdp', '.hdp'],
    reader_class=JXRFormatReader
)

# JPEG - automatically detects standard JPEG vs Ultra HDR
from image_pipeline.io.formats.jpeg import JPEGReader

ImageReader.register_format(
    extensions=['.jpg', '.jpeg'],
    reader_class=JPEGReader
)