"""
CLI interface for image pipeline
"""
import argparse
import sys

from image_pipeline.core.processor import process_image
from image_pipeline.cli.filter_parser import parse_filters
from image_pipeline.cli.filter_registry import FILTER_REGISTRY


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Image processing pipeline with filters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple format conversion
  python main.py input.tif output.png

  # Apply single filter
  python main.py input.tif --filter remove_alpha output.png

  # Apply multiple filters
  python main.py input.exr \\
    --filter pq_encode:peak_luminance=10000 \\
    --filter blur:sigma=2.5 \\
    --filter quantize:bit_depth=16 \\
    --quality 90 \\
    output.png

  # Complex pipeline
  python main.py input.tif \\
    --filter normalize:min_val=0.0,max_val=1.0 \\
    --filter grayscale:method=luminosity \\
    --filter sharpen:strength=1.5 \\
    --verbose \\
    output.jpg

Available filters:
  remove_alpha, normalize, pq_encode, pq_decode, grayscale,
  quantize, dequantize, sharpen, blur

For filter details:
  python main.py --list-filters
        """
    )
    
    parser.add_argument(
        "input",
        nargs='?',
        help="Input image file path"
    )
    
    parser.add_argument(
        "output",
        nargs='?',
        help="Output image file path"
    )
    
    parser.add_argument(
        "--filter", "-f",
        action='append',
        dest='filters',
        metavar='SPEC',
        help="Filter to apply (format: name:param1=value1,param2=value2). Can be specified multiple times."
    )
    
    parser.add_argument(
        "--quality",
        type=int,
        metavar='N',
        help="Quality for lossy formats (1-100)"
    )
    
    parser.add_argument(
        "--compression",
        type=int,
        metavar='N',
        help="Compression level (format-dependent)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action='store_true',
        help="Print detailed processing information"
    )
    
    parser.add_argument(
        "--list-filters",
        action='store_true',
        help="List all available filters and exit"
    )
    
    return parser


def list_available_filters():
    """Print list of available filters"""
    print("Available filters:")
    print()
    for name in sorted(FILTER_REGISTRY.keys()):
        filter_class = FILTER_REGISTRY[name]
        print(f"  {name:20} -> {filter_class.__name__}")
    print()
    print("Use --filter <name>:<params> to apply filters")
    print("Example: --filter blur:sigma=2.5")


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle --list-filters
    if args.list_filters:
        list_available_filters()
        sys.exit(0)
    
    # Validate required arguments
    if not args.input or not args.output:
        parser.error("input and output arguments are required")
    
    # Parse filters
    try:
        filters = parse_filters(args.filters) if args.filters else []
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Prepare save options
    save_options = {}
    if args.quality is not None:
        save_options['quality'] = args.quality
    if args.compression is not None:
        save_options['compression_level'] = args.compression
    
    # Process image
    try:
        process_image(
            input_path=args.input,
            output_path=args.output,
            filters=filters,
            save_options=save_options,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()