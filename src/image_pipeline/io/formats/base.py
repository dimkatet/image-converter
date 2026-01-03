"""
Base classes for format-specific readers and writers
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping
import warnings

from image_pipeline.core.image_data import ImageData
from image_pipeline.types import SaveOptions


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

class SaveOptionsAdapter(ABC):
    """
    Base class for format-specific save options validation
    
    Each format implements its own adapter to:
    - Validate option values
    - Filter unsupported options
    - Provide defaults
    - Give helpful warnings
    """
    
    @abstractmethod
    def validate(self, options: SaveOptions) -> Mapping[str, Any]:
        """
        Validate and normalize options for specific format
        
        Args:
            options: SaveOptions dict (may contain unsupported keys)
            
        Returns:
            Dictionary with validated options for this format
            
        Raises:
            ValueError: If option values are invalid
            TypeError: If option types are incorrect
        """
        pass
    
    @abstractmethod
    def get_supported_options(self) -> set[str]:
        """
        Return set of option names supported by this format
        
        Returns:
            Set of supported option names
        """
        pass
    
    def _warn_unsupported(self, options: SaveOptions, format_name: str) -> None:
        """
        Warn about unsupported options
        
        Args:
            options: User-provided options
            format_name: Name of the format (e.g., 'PNG', 'JPEG')
        """
        supported = self.get_supported_options()
        unsupported = set(options.keys()) - supported
        
        if unsupported:
            warnings.warn(
                f"{format_name} format does not support options: {', '.join(sorted(unsupported))}",
                UserWarning
            )


class FormatWriter(ABC):
    """Base class for format-specific writers"""
    
    def __init__(self, filepath: str):
        """
        Args:
            filepath: Path to save the image
        """
        self.filepath = Path(filepath)
    
    @abstractmethod
    def write(self, img_data: ImageData, options: SaveOptions) -> None:
        """
        Write image with metadata to file

        This is the main entry point for writing images. Each format
        decides how to handle pixels and metadata (separate writes,
        single write, etc.)

        Args:
            img_data: ImageData with pixels and metadata
            options: Format-specific options (quality, compression, etc.)
        """
        pass

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

    def ensure_directory(self) -> None:
        """Create parent directory if it doesn't exist"""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
