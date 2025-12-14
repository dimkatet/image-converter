"""
Image writer facade with automatic format detection
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Type, Optional

from image_pipeline.core import ImageData
from image_pipeline.metadata import MetadataWriter
from image_pipeline.types import SaveOptions

from .formats import FormatWriter

@dataclass
class FormatWriterConfig:
    """Configuration for a file format writer"""
    writer: Type[FormatWriter]  # FormatWriter class
    metadata_writer: Optional[Type[MetadataWriter]] = None  # MetadataWriter class (if format supports it)


class ImageWriter:
    """
    Facade for writing images with automatic format detection
    
    Automatically selects the appropriate format-specific writer
    based on file extension. Handles both pixel data and metadata.
    """
    
    # Mapping of extensions to format writer configs
    _WRITERS = {}
    
    def __init__(self, filepath: str):
        """
        Args:
            filepath: Path to save the image
        """
        self.filepath = Path(filepath)
    
    def write(self, img_data: ImageData, options: SaveOptions) -> None:
        """
        Write image to file (pixels + metadata)
        
        Args:
            img_data: ImageData object with pixels and metadata
            options: Format-specific options (quality, compression, etc.)
        """
        # Get format configuration
        config = self._get_format_config()
        
        # Create format writer
        format_writer = config.writer(str(self.filepath))
        
        # 1. Validate data
        format_writer.validate(img_data)
        
        # 2. Ensure directory exists
        format_writer.ensure_directory()
        
        # 3. Write pixels
        format_writer.write_pixels(img_data, options)
        
        # 4. Write metadata (if format supports it)
        if config.metadata_writer:
            config.metadata_writer.write_metadata(str(self.filepath), img_data.metadata)
    
    def _get_format_config(self) -> FormatWriterConfig:
        """Get format configuration based on file extension"""
        ext = self.filepath.suffix.lower()
        
        config = self._WRITERS.get(ext)
        if not config:
            supported = ', '.join(sorted(self._WRITERS.keys()))
            raise ValueError(
                f"Unsupported format: {ext}. "
                f"Supported formats: {supported}"
            )
        
        return config
    
    @classmethod
    def register_format(cls, extensions: list, writer_class, metadata_writer_class=None):
        """
        Register a format writer
        
        Args:
            extensions: List of file extensions (e.g., ['.png'])
            writer_class: FormatWriter subclass
            metadata_writer_class: MetadataWriter class (optional)
        """
        config = FormatWriterConfig(
            writer=writer_class,
            metadata_writer=metadata_writer_class
        )
        
        for ext in extensions:
            cls._WRITERS[ext.lower()] = config


# Import and register formats
from .formats.png import PNGFormatWriter
from image_pipeline.metadata.png import PNGMetadataWriter

ImageWriter.register_format(
    extensions=['.png'],
    writer_class=PNGFormatWriter,
    metadata_writer_class=PNGMetadataWriter
)