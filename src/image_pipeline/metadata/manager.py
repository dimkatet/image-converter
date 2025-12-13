"""
Metadata Manager - determines format and delegates to specific writers
"""
from pathlib import Path
from typing import Dict, Any

from image_pipeline.types import ImageMetadata


class MetadataManager:
    """
    Central dispatcher for metadata writing across different image formats
    """
    
    @staticmethod
    def write(filepath: str, metadata: ImageMetadata) -> None:
        """
        Apply metadata to saved image file
        
        Determines format by file extension and calls appropriate writer
        
        Args:
            filepath: Path to the image file
            metadata: ImageMetadata dictionary
        """
        if not metadata:
            return
        
        ext = Path(filepath).suffix.lower()
        
        if ext == '.png':
            from image_pipeline.metadata.png_metadata_writer import PNGMetadataWriter
            PNGMetadataWriter.write(filepath, metadata)
        
        # Future formats:
        # elif ext == '.avif':
        #     from image_pipeline.metadata.avif_metadata_writer import AVIFMetadataWriter
        #     AVIFMetadataWriter.write(filepath, metadata)
        # elif ext in ['.jpg', '.jpeg']:
        #     from image_pipeline.metadata.jpeg_metadata_writer import JPEGMetadataWriter
        #     JPEGMetadataWriter.write(filepath, metadata)
        
        # If format doesn't support metadata, silently skip