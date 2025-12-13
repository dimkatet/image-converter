import argparse
from pathlib import Path
from typing import List, Optional, Union

from image_pipeline.core.filter_pipeline import FilterPipeline
from image_pipeline import RemoveAlphaFilter, PQEncodeFilter, QuantizeFilter
from image_pipeline.core.image_data import ImageData
from image_pipeline.filters.base import ImageFilter
from image_pipeline.io.reader import ImageReader
from image_pipeline.io.saver import ImageSaver

def process_image(input_path: str,
                 output_path: str,
                 filters: Union[List['ImageFilter'], 'FilterPipeline'],
                 save_options: Optional[dict] = None,
                 verbose: bool = False) -> 'ImageData':
    """
    Image processing: read -> apply filters -> save
    
    Args:
        input_path: Path to the input file
        output_path: Path to save the result
        filters: List of filters or FilterPipeline to apply
        save_options: Save parameters (quality, compression, etc.)
        verbose: Print detailed process information
        
    Returns:
        ImageData with the processed image
        
    Example:
        >>> from image_filters import GrayscaleFilter, BlurFilter
        >>> filters = [GrayscaleFilter(), BlurFilter(sigma=2.0)]
        >>> result = process_image('input.tiff', 'output.png', filters)
    """
    save_options = save_options or {}
    
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
        if verbose:
            print(f"\nApplying filters...")
        
        # If a list of filters is passed, create a pipeline
        if isinstance(filters, list):
            pipeline = FilterPipeline(filters)
        else:
            pipeline = filters
        
        # Apply filters
        processed_data = pipeline.apply(img_data, verbose=verbose)
                
        # 3. Save result
        if verbose:
            print(f"\nSaving to: {output_path}")
        
        ImageSaver.save_with_format_conversion(processed_data, output_path, **save_options)
        
        if verbose:
            print(f"  Successfully saved!")
            output_size = Path(output_path).stat().st_size / 1024  # KB
            print(f"  File size: {output_size:.2f} KB")
        
        return processed_data
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Input file not found: {input_path}") from e
    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}") from e



def main():
    parser = argparse.ArgumentParser(description="Image Converter")
    parser.add_argument("input_file", help="Path to the input image file")
    parser.add_argument("output_file", help="Path to the output image file")
    parser.add_argument("--bit-depth", type=int, choices=[8, 12, 16, 32], default=8,
                        help="Bit depth for output image (default: 8)")
    parser.add_argument("--quality", type=int, choices=range(1, 101), metavar="[1-100]",
                        help="Quality for output image (for lossy formats)")
    parser.add_argument("--format", type=str, help="Output image format (e.g., png, jpg, tiff)")
    # Add more optional arguments as needed

    args = parser.parse_args()
    

    process_image(
        input_path=args.input_file,
        output_path=args.output_file,
        filters=[RemoveAlphaFilter(), PQEncodeFilter(peak_luminance=10000), QuantizeFilter(bit_depth=16)],
        save_options={
            # "bit_depth": args.bit_depth,
            "quality": args.quality,
            # "format": args.format,
        },
        verbose=True
    )

if __name__ == "__main__":
    main()