"""
Image writer facade with automatic format detection
"""
from pathlib import Path
from typing import Type

from image_pipeline.core import ImageData
from image_pipeline.types import SaveOptions

from .formats import FormatWriter


class ImageWriter:
    """
    Facade for writing images with automatic format detection

    Automatically selects the appropriate format-specific writer
    based on file extension. Each writer handles both pixels and metadata.
    """

    # Mapping of extensions to FormatWriter classes
    _WRITERS: dict[str, Type[FormatWriter]] = {}

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
        # Get format writer class
        writer_class = self._get_writer_class()

        # Create format writer
        format_writer = writer_class(str(self.filepath))

        # 1. Validate data
        format_writer.validate(img_data)

        # 2. Ensure directory exists
        format_writer.ensure_directory()

        # 3. Write image (pixels + metadata)
        format_writer.write(img_data, options)

    def _get_writer_class(self) -> Type[FormatWriter]:
        """Get writer class based on file extension"""
        ext = self.filepath.suffix.lower()

        writer_class = self._WRITERS.get(ext)
        if not writer_class:
            supported = ', '.join(sorted(self._WRITERS.keys()))
            raise ValueError(
                f"Unsupported format: {ext}. "
                f"Supported formats: {supported}"
            )

        return writer_class

    @classmethod
    def register_format(cls, extensions: list[str], writer_class: Type[FormatWriter]):
        """
        Register a format writer

        Args:
            extensions: List of file extensions (e.g., ['.png'])
            writer_class: FormatWriter subclass
        """
        for ext in extensions:
            cls._WRITERS[ext.lower()] = writer_class


# Import and register formats
from .formats.png import PNGFormatWriter
from .formats.avif import AVIFFormatWriter
from .formats.tiff import TiffFormatWriter
from .formats.webp import WebPFormatWriter
from .formats.jpeg import JPEGWriter
from .formats.exr import EXRFormatWriter
from .formats.jxr import JXRFormatWriter

ImageWriter.register_format(
    extensions=['.png'],
    writer_class=PNGFormatWriter
)

ImageWriter.register_format(
    extensions=['.tiff', '.tif'],
    writer_class=TiffFormatWriter
)

ImageWriter.register_format(
    extensions=['.avif'],
    writer_class=AVIFFormatWriter
)

ImageWriter.register_format(
    extensions=['.webp'],
    writer_class=WebPFormatWriter
)

ImageWriter.register_format(
    extensions=['.jpg', '.jpeg'],
    writer_class=JPEGWriter
)

ImageWriter.register_format(
    extensions=['.exr'],
    writer_class=EXRFormatWriter
)

ImageWriter.register_format(
    extensions=['.jxr', '.wdp', '.hdp'],
    writer_class=JXRFormatWriter
)