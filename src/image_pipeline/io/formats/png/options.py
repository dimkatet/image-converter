"""
PNG format save options adapter
"""
from typing import TypedDict
from image_pipeline.io.formats.base import SaveOptionsAdapter
from image_pipeline.types import SaveOptions

class PNGSaveOptions(TypedDict, total=False):
    """
    PNG-specific save options
    """
    compression_level: int  # 0 (no compression) to 9 (max compression)
    optimize: bool          # Whether to optimize the PNG file size


class PNGOptionsAdapter(SaveOptionsAdapter):
    """Adapter for PNG save options"""
    
    def get_supported_options(self) -> set[str]:
        """PNG supports compression_level and optimize"""
        return {'compression_level', 'optimize'}
    
    def validate(self, options: SaveOptions) -> PNGSaveOptions:
        """
        Validate and normalize PNG save options
        
        Args:
            options: User-provided save options
            
        Returns:
            Dictionary with validated PNG-specific options
            
        Raises:
            ValueError: If option values are invalid
            TypeError: If option types are incorrect
        """
        validated: PNGSaveOptions = {}
        
        # Warn about unsupported options
        self._warn_unsupported(options, 'PNG')
        
        # compression_level (0-9)
        if 'compression_level' in options:
            level = options['compression_level']
            
            if not isinstance(level, int):
                raise TypeError(
                    f"compression_level must be int, got {type(level).__name__}"
                )
            
            if not 0 <= level <= 9:
                raise ValueError(
                    f"compression_level must be in range [0, 9], got {level}"
                )
            
            validated['compression_level'] = level
        else:
            validated['compression_level'] = 6  # Default
        
        # optimize (bool)
        if 'optimize' in options:
            optimize = options['optimize']
            
            if not isinstance(optimize, bool):
                raise TypeError(
                    f"optimize must be bool, got {type(optimize).__name__}"
                )
            
            validated['optimize'] = optimize
        else:
            validated['optimize'] = False  # Default
        
        return validated