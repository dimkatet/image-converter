import warnings

from image_pipeline.metadata.base import MetadataWriter
from .codec import PNGMetadataCodec
from .adapter import PNGMetadataAdapter
from image_pipeline.types import ImageMetadata


class PNGMetadataWriter(MetadataWriter):
    """
    Handles PNG-specific metadata writing
    Converts ImageMetadata → PNG chunks and writes to file
    """
    
    @staticmethod
    def write_metadata(filepath: str, metadata: ImageMetadata) -> None:
        """
        Apply metadata to PNG file
        
        Process:
        1. Convert metadata to PNG chunks using adapter
        2. Read existing PNG chunks
        3. Add/update metadata chunks
        4. Write back to file
        
        Args:
            filepath: Path to PNG file (must already exist)
            metadata: ImageMetadata dictionary
        """
        try:
            # Step 1: Convert metadata to PNG-specific chunks
            chunks = PNGMetadataAdapter.convert(metadata)
            
            # If no chunks to add, skip
            if not chunks:
                return
            
            # Step 2: Read existing PNG
            codec = PNGMetadataCodec(filepath)
            codec.read_chunks()

            # Step 3: Add metadata chunks
            codec.set_metadata_chunks(
                cicp=chunks.get('cicp'),
                mdcv=chunks.get('mdcv'),
                clli=chunks.get('clli'),
                chrm=chunks.get('chrm'),
                gama=chunks.get('gama'),
                srgb=chunks.get('srgb'),
                gmap=chunks.get('gmap'),
                gdat=chunks.get('gdat')
            )
            
            # Add text metadata chunks
            if 'text' in chunks:
                codec.set_metadata(chunks['text'])
            
            # Step 4: Write back to file
            codec.write_chunks(filepath)
            
        except Exception as e:
            # Don't crash the whole pipeline if metadata fails
            warnings.warn(
                f"Failed to apply PNG metadata to {filepath}: {e}",
                UserWarning
            )
