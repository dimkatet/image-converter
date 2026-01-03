from typing import List
from typing import Optional

from image_pipeline.core.image_data import ImageData
from image_pipeline.filters.base import ImageFilter


class FilterPipeline:
    """Class for sequentially applying multiple filters"""
    
    def __init__(self, filters: Optional[List[ImageFilter]] = None):
        """
        Args:
            filters: List of filters to apply
        """
        self.filters: List[ImageFilter] = filters or []
    
    def add(self, filter: ImageFilter) -> 'FilterPipeline':
        """
        Add a filter to the pipeline
        
        Args:
            filter: Filter to add
            
        Returns:
            self for chaining
        """
        self.filters.append(filter)
        return self
    
    def remove(self, index: int) -> 'FilterPipeline':
        """
        Remove a filter by index
        
        Args:
            index: Index of the filter
            
        Returns:
            self for chaining
        """
        if 0 <= index < len(self.filters):
            self.filters.pop(index)
        return self
    
    def clear(self) -> 'FilterPipeline':
        """Remove all filters"""
        self.filters.clear()
        return self
    
    def apply(self, img_data: ImageData, verbose: bool = False) -> ImageData:
        """
        Apply all filters sequentially
        
        Args:
            img_data: Input ImageData object
            verbose: Print information about each step
            
        Returns:
            Processed pixel array
        """
        result = img_data.copy()
        
        for i, filter in enumerate(self.filters):
            if verbose:
                print(f"Step {i+1}/{len(self.filters)}: {filter}")
            result = filter(result)
        
        return result
    
    def __len__(self) -> int:
        return len(self.filters)
    
    def __repr__(self) -> str:
        filters_str = ", ".join(str(f) for f in self.filters)
        return f"FilterPipeline([{filters_str}])"
