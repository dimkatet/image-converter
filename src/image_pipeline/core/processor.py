"""
Image processing pipeline
Core functionality without CLI dependencies
"""
from pathlib import Path
from typing import List, Optional, Union

from image_pipeline.core.filter_pipeline import FilterPipeline
from image_pipeline.core.image_data import ImageData
from image_pipeline.filters.base import ImageFilter
from image_pipeline.io.reader import ImageReader
from image_pipeline.io.writer import ImageWriter
from image_pipeline.types import SaveOptions


def process_image(input_path: str,
                 output_path: str,
                 filters: Optional[Union[List[ImageFilter], FilterPipeline]] = None,
                 save_options: Optional[SaveOptions] = None,
                 verbose: bool = False) -> ImageData:
    """
    Image processing pipeline: read -> apply filters -> save
    
    Args:
        input_path: Path to the input file
        output_path: Path to save the result
        filters: List of filters or FilterPipeline to apply (optional)
        save_options: Save parameters (quality, compression, etc.)
        verbose: Print detailed process information
        
    Returns:
        ImageData with the processed image
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If processing fails
        
    Examples:
        >>> from image_pipeline.filters import GrayscaleFilter, BlurFilter
        >>> filters = [GrayscaleFilter(), BlurFilter(sigma=2.0)]
        >>> result = process_image('input.tiff', 'output.png', filters)
        
        >>> # Without filters (format conversion only)
        >>> result = process_image('input.tiff', 'output.png')
    """
    save_options = save_options or {}
    filters = filters or []
    
    try:
        # 1. Read image
        if verbose:
            print(f"Reading file: {input_path}")
        
        reader = ImageReader(input_path)
        img_data = reader.read()
        
        if verbose:
            print(f"  Loaded: {img_data.shape}, {img_data.dtype}")
            print(f"  Format: {img_data.format}")
        
        # 2. Apply filters
        if filters:
            if verbose:
                num_filters = len(filters) if isinstance(filters, list) else len(filters.filters)
                print(f"\nApplying {num_filters} filter(s)...")
            
            # If a list of filters is passed, create a pipeline
            if isinstance(filters, list):
                pipeline = FilterPipeline(filters)
            else:
                pipeline = filters
            
            # Apply filters
            processed_data = pipeline.apply(img_data, verbose=verbose)
        else:
            if verbose:
                print("\nNo filters to apply, converting format only...")
            processed_data = img_data
                
        # 3. Save result
        if verbose:
            print(f"\nSaving to: {output_path} with options: {save_options}")
        
        ImageWriter(output_path).write(processed_data, options=save_options)
        
        if verbose:
            print(f"  Successfully saved!")
            output_size = Path(output_path).stat().st_size / 1024  # KB
            print(f"  File size: {output_size:.2f} KB")
        
        return processed_data
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Input file not found: {input_path}") from e
    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}") from e