# Image Pipeline

A flexible library for image processing and conversion with support for HDR, 16-bit PNG, and various color spaces.

## Features

- **Wide format support**: TIFF, PNG (8/16-bit), JPEG, WebP, EXR, HDR, PFM and more
- **HDR processing**: work with high dynamic range images
- **PQ encoding**: Perceptual Quantizer (ST.2084) support for HDR content
- **Flexible filter system**: modular architecture for creating processing pipelines
- **Precise bit depth handling**: conversion between 8/10/12/16/32-bit formats
- **Convenient API**: simple interface for quick image processing

## Installation

```bash
pip install image_pipeline
```

Or for development:

```bash
git clone <repository-url>
cd image_pipeline
pip install -e .
```

## Quick Start

### Basic Usage

```python
from image_pipeline import process_image, GrayscaleFilter, BlurFilter

# Simple processing with filters
result = process_image(
    input_path='input.tiff',
    output_path='output.png',
    filters=[GrayscaleFilter(), BlurFilter(sigma=2.0)]
)
```

### HDR Processing with PQ Encoding

```python
from image_pipeline import (
    RemoveAlphaFilter, 
    PQEncodeFilter, 
    QuantizeFilter,
    process_image
)

# HDR → PQ-encoded 16-bit PNG
process_image(
    input_path='hdr_image.exr',
    output_path='output_pq.png',
    filters=[
        RemoveAlphaFilter(),
        PQEncodeFilter(peak_luminance=10000),
        QuantizeFilter(bit_depth=16)
    ],
    verbose=True
)
```

### Using FilterPipeline

```python
from image_pipeline import FilterPipeline, ImageReader, ImageSaver
from image_pipeline import NormalizeFilter, SharpenFilter

# Create pipeline
pipeline = FilterPipeline()
pipeline.add(NormalizeFilter(min_val=0.0, max_val=1.0))
pipeline.add(SharpenFilter(strength=1.5))

# Read → process → save
reader = ImageReader('input.jpg')
img_data = reader.read()

processed_pixels = pipeline.apply(img_data.pixels, verbose=True)
ImageSaver.save_with_format_conversion(
    processed_pixels,
    'output.png',
    quality=95
)
```

## Available Filters

### Color Transformations
- **`RemoveAlphaFilter()`** - remove alpha channel (RGBA → RGB)
- **`GrayscaleFilter(method='luminosity')`** - convert to grayscale
- **`NormalizeFilter(min_val=0.0, max_val=1.0)`** - normalize value range

### HDR and Quantization
- **`PQEncodeFilter(peak_luminance=10000)`** - apply PQ gamma curve (ST.2084)
- **`PQDecodeFilter(peak_luminance=10000)`** - decode PQ back to linear values
- **`QuantizeFilter(bit_depth=8)`** - quantize float → integer (8/10/12/16/32-bit)
- **`DequantizeFilter(bit_depth=None)`** - dequantize integer → float

### Image Enhancement
- **`BlurFilter(sigma=1.0)`** - Gaussian blur
- **`SharpenFilter(strength=1.0)`** - sharpen image

## Supported Formats

### Reading
- **TIFF/TIF**: uint8, uint16, uint32, float32, float64
- **PNG**: uint8, uint16
- **HDR formats**: EXR, HDR, PFM (float32, float64)
- **Standard**: JPEG, BMP, WebP, GIF

### Writing
- **TIFF**: all data types, compression (LZW, Deflate, ZSTD, JPEG)
- **PNG**: uint8, uint16 (via pypng for 16-bit)
- **HDR**: EXR, HDR, PFM (float32, float64)
- **Standard**: JPEG, WebP (uint8 only)

## CLI Usage

```bash
# Basic conversion
python main.py input.tiff output.png

# With quality parameters
python main.py input.exr output.jpg --quality 90

# Specify bit depth
python main.py input.png output.tiff --bit-depth 16
```

## Example Workflows

### HDR Photo → SDR Display

```python
from image_pipeline import FilterPipeline
from image_pipeline import PQEncodeFilter, QuantizeFilter, NormalizeFilter

pipeline = FilterPipeline([
    NormalizeFilter(0, 1),                      # Normalize
    PQEncodeFilter(peak_luminance=10000),       # PQ encoding
    QuantizeFilter(bit_depth=8)                 # Convert to uint8
])

result = pipeline.apply(hdr_pixels)
```

### Batch Processing with Enhancements

```python
from pathlib import Path
from image_pipeline import process_image, SharpenFilter, NormalizeFilter

filters = [
    NormalizeFilter(),
    SharpenFilter(strength=0.5)
]

for img_path in Path('input/').glob('*.tiff'):
    output_path = f'output/{img_path.stem}.png'
    process_image(str(img_path), output_path, filters)
```

## Architecture

```
image_pipeline/
├── core/
│   ├── filter_pipeline.py   # Filter chain
│   └── image_data.py         # Image container
├── filters/
│   ├── base.py               # Base filter class
│   ├── pq_encode.py          # PQ encoding
│   ├── pq_decode.py          # PQ decoding
│   ├── quantize.py           # Quantization
│   └── ...                   # Other filters
└── io/
    ├── reader.py             # Image reading
    ├── writer.py             # Image writing
    └── saver.py              # High-level saving
```

## Requirements

- Python ≥ 3.12
- NumPy
- Pillow
- imageio
- tifffile
- scipy
- pypng (optional, for uint16 PNG)

## Development

### Creating Custom Filters

```python
from image_pipeline.filters.base import ImageFilter
import numpy as np

class MyCustomFilter(ImageFilter):
    def __init__(self, param=1.0):
        super().__init__()
        self.param = param
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        # Your processing logic
        result = pixels * self.param
        return result.astype(pixels.dtype)
```

## License

MIT

## Author

Dima Teterin (tet.dima.211@gmail.com)

## Roadmap

- [ ] AVIF format support
- [ ] Additional color spaces (Lab, XYZ)
- [ ] Tone mapping operators
- [ ] Batch processing via CLI
- [ ] Extended metadata (EXIF, ICC profiles)