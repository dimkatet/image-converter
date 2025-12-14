"""
Base class for metadata and usage examples
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

from image_pipeline.types import ImageMetadata


class MetadataCodec(ABC):
    """Base class for metadata codecs"""
    
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = Path(filepath) if filepath else None
    
    @abstractmethod
    def read_metadata(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Read metadata from file
        
        Args:
            filepath: Path to file
            
        Returns:
            Dictionary with metadata
        """
        pass
    
    @abstractmethod
    def write_metadata(self, metadata: Dict[str, Any], 
                      source: str, destination: str) -> None:
        """
        Write metadata to file
        
        Args:
            metadata: Metadata to write
            source: Source file
            destination: Destination file
        """
        pass
    
    @abstractmethod
    def update_metadata(self, filepath: str, 
                       metadata: Dict[str, Any]) -> None:
        """
        Update metadata in an existing file
        
        Args:
            filepath: Path to file
            metadata: Metadata to update
        """
        pass
    
class MetadataWriter(ABC):
    """Base class for metadata writers"""
    
    def __init__(self, filepath: str):
        """
        Args:
            filepath: Path to save the image
        """
        self.filepath = Path(filepath)
    
    @staticmethod
    @abstractmethod
    def write_metadata(filepath: str, metadata: ImageMetadata) -> None:
        """
        Apply metadata to PNG file
        
        Args:
            filepath: Path to PNG file (must already exist)
            metadata: ImageMetadata dictionary
            
        Raises:
            ValueError: If data is not compatible with format
        """
        pass
    
   
