"""
Base classes for format-specific readers and writers
"""
from abc import ABC, abstractmethod
from pathlib import Path

from image_pipeline.core.image_data import ImageData


class FormatReader(ABC):
    """Base class for format-specific readers"""
    
    def __init__(self, filepath: str):
        """
        Args:
            filepath: Path to the image file
        """
        self.filepath = Path(filepath)
        self.validate_file()
    
    @abstractmethod
    def read(self) -> ImageData:
        """
        Read image from file
        
        Returns:
            ImageData object with pixels and metadata
        """
        pass
    
    def validate_file(self) -> None:
        """Validate that file exists and is a file"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        if not self.filepath.is_file():
            raise ValueError(f"Path is not a file: {self.filepath}")


class FormatWriter(ABC):
    """Base class for format-specific writers"""
    
    def __init__(self, filepath: str):
        """
        Args:
            filepath: Path to save the image
        """
        self.filepath = Path(filepath)
    
    @abstractmethod
    def validate(self, img_data: ImageData) -> None:
        """
        Validate that image data is compatible with this format
        
        Args:
            img_data: ImageData to validate
            
        Raises:
            ValueError: If data is not compatible with format
        """
        pass
    
    @abstractmethod
    def write_pixels(self, img_data: ImageData, **options) -> None:
        """
        Write pixel data to file
        
        Args:
            img_data: ImageData with pixels and metadata
            **options: Format-specific options (quality, compression, etc.)
        """
        pass
    
    def ensure_directory(self) -> None:
        """Create parent directory if it doesn't exist"""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
