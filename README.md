# Image Pipeline

A flexible library for professional HDR image processing with support for PQ encoding, scene/display-referred workflows, and comprehensive HDR metadata handling.

## Features

- **Wide format support**: TIFF, PNG (8/16-bit), JPEG, WebP, AVIF, EXR, HDR, PFM
- **Professional HDR workflows**: scene-referred ↔ display-referred conversion with paper white tracking
- **PQ encoding**: Perceptual Quantizer (ST.2084) support for HDR10 content
- **Color space conversion**: RGB color space transforms (BT.709, BT.2020, Display P3) via CIE XYZ
- **HDR metadata**: MaxCLL/MaxFALL computation, mastering display metadata, PNG cICP/mDCv/cLLi chunks
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

### HDR Processing with Scene-Referred Workflow

```python
from image_pipeline import (
    RemoveAlphaFilter,
    ColorConvertFilter,
    AbsoluteLuminanceFilter,
    PQEncodeFilter,
    QuantizeFilter,
    process_image
)
from image_pipeline.types import ColorSpace

# Scene-referred EXR → Display-referred PQ AVIF with HDR10 metadata
process_image(
    input_path='scene_linear.exr',
    output_path='hdr10_output.avif',
    filters=[
        RemoveAlphaFilter(),
        ColorConvertFilter(source=ColorSpace.BT709, target=ColorSpace.BT2020),
        AbsoluteLuminanceFilter(paper_white=100),  # Scene → Display (computes MaxCLL/MaxFALL)
        PQEncodeFilter(reference_peak=10000),       # Apply PQ curve
        QuantizeFilter(bit_depth=12)                # 12-bit for HDR10
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

### Scene/Display Luminance Conversion
- **`AbsoluteLuminanceFilter(paper_white=100)`** - Convert scene-referred → display-referred (computes MaxCLL/MaxFALL)
- **`RelativeLuminanceFilter(paper_white=100)`** - Convert display-referred → scene-referred (inverse operation)

### Color Space Conversion
- **`ColorConvertFilter(source, target)`** - RGB color space conversion via CIE XYZ
  - Supported: `ColorSpace.BT709`, `ColorSpace.BT2020`, `ColorSpace.DISPLAY_P3`
  - Requires linear (non-gamma) input

### HDR Encoding/Decoding
- **`PQEncodeFilter(reference_peak=10000)`** - Apply PQ gamma curve (ST.2084) for HDR10
- **`PQDecodeFilter(peak_luminance=10000)`** - Decode PQ back to linear luminance
- **`QuantizeFilter(bit_depth=8)`** - Quantize float → integer (8/10/12/16/32-bit)
- **`DequantizeFilter(bit_depth=None)`** - Dequantize integer → float

### Color Transformations
- **`RemoveAlphaFilter()`** - Remove alpha channel (RGBA → RGB)
- **`GrayscaleFilter(method='luminosity')`** - Convert to grayscale
- **`NormalizeFilter(min_val=0.0, max_val=1.0)`** - Normalize value range

### Image Enhancement
- **`BlurFilter(sigma=1.0)`** - Gaussian blur
- **`SharpenFilter(strength=1.0)`** - Sharpen image

## Supported Formats

### Reading
- **TIFF/TIF**: uint8, uint16, uint32, float32, float64 (with LZW, Deflate, ZSTD, JPEG compression)
- **PNG**: uint8, uint16 (with HDR metadata support planned)
- **AVIF**: uint8, uint10, uint12 with HDR metadata
- **HDR formats**: EXR, HDR, PFM (float32, float64)
- **Standard**: JPEG, BMP, WebP, GIF

### Writing
- **TIFF**: all data types, compression (LZW, Deflate, ZSTD, JPEG)
- **PNG**: uint8, uint16 (via pypng for 16-bit) with HDR metadata (cICP, mDCv, cLLi chunks)
- **AVIF**: uint8, uint10, uint12 with HDR10 metadata
- **WebP**: lossy/lossless modes
- **HDR**: EXR, HDR, PFM (float32, float64)
- **Standard**: JPEG (uint8 only)

## CLI Usage

```bash
# Basic conversion
python main.py input.tiff output.png

# HDR workflow: Scene-referred TIFF → HDR10 AVIF
python main.py input.tiff output.avif \
  --filter remove_alpha \
  --filter color_convert:source=bt709,target=bt2020 \
  --filter absolute_luminance:paper_white=100 \
  --filter pq_encode:reference_peak=10000 \
  --filter quantize:bit_depth=12 \
  --verbose

# List available filters
python main.py --list-filters

# With quality and compression
python main.py input.exr output.avif --quality 90 --verbose
```

## Example Workflows

### Scene-Referred HDR → Display-Referred PQ

```python
from image_pipeline import FilterPipeline, ImageReader, ImageWriter
from image_pipeline.filters import (
    AbsoluteLuminanceFilter,
    PQEncodeFilter,
    QuantizeFilter
)

# Scene-referred linear EXR → Display-referred PQ PNG
reader = ImageReader('scene_linear.exr')
img_data = reader.read()

pipeline = FilterPipeline([
    AbsoluteLuminanceFilter(paper_white=100),   # Scene → Display (sets MaxCLL/MaxFALL)
    PQEncodeFilter(reference_peak=10000),       # Apply PQ curve
    QuantizeFilter(bit_depth=16)                # Convert to uint16
])

processed = pipeline.apply_to_image_data(img_data)

writer = ImageWriter('output_pq.png', processed)
writer.write()
```

### Round-Trip: Scene → Display → Scene

```python
from image_pipeline.filters import AbsoluteLuminanceFilter, RelativeLuminanceFilter

# Scene-referred data
scene_data = reader.read()  # pixel values relative to paper_white

# Convert to display-referred (absolute nits)
abs_filter = AbsoluteLuminanceFilter(paper_white=100)
display_data = abs_filter.apply(scene_data.pixels)
# metadata now has: paper_white=100, max_cll, max_fall

# Process in display-referred space...
# (e.g., tone mapping, color grading)

# Convert back to scene-referred
rel_filter = RelativeLuminanceFilter(paper_white=100)
back_to_scene = rel_filter.apply(display_data)
# metadata preserves: paper_white=100 (no max_cll/max_fall)
```

### Color Space Conversion for HDR

```python
from image_pipeline.filters import ColorConvertFilter
from image_pipeline.types import ColorSpace

# BT.709 → BT.2020 (for HDR delivery)
pipeline = FilterPipeline([
    ColorConvertFilter(
        source=ColorSpace.BT709,
        target=ColorSpace.BT2020
    )
])

# Note: Requires linear (non-gamma) input
result = pipeline.apply(linear_rgb_pixels)
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
- tifffile
- scipy
- pypng
- imagecodecs

## Development

### Setup

```bash
# Clone repository
git clone <repository-url>
cd image_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install package with dev dependencies (includes pytest, pyright)
pip install -e ".[dev]"
```

### Testing

The project uses pytest for testing:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_filters/test_pq_encode_decode.py

# Run with verbose output
pytest -v

# Run with coverage (if pytest-cov installed)
pytest --cov=src/image_pipeline --cov-report=html
```

**Test structure:**
- `tests/unit/test_filters/` - Unit tests for filters (PQ, quantize, color conversion, etc.)
- `tests/integration/` - End-to-end workflow tests (planned)
- `tests/conftest.py` - Shared fixtures

See [tests/README.md](tests/README.md) for detailed testing documentation.

### Type Checking

The project uses Pyright for static type checking:

```bash
# Run type checker
pyright

# Pyright checks only src/ directory (configured in pyproject.toml)
# Must pass with 0 errors before committing
```

### CI/CD

GitHub Actions runs automated checks on every push and pull request:

1. **Pyright type checking** - Must pass with 0 errors
2. **Pytest test suite** - All tests must pass

See [.github/workflows/ci.yml](.github/workflows/ci.yml) for CI configuration.

**Run CI checks locally before pushing:**

```bash
# Type checking
pyright

# Tests
pytest tests/ -v --tb=short
```

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

## Building Executables

You can build standalone executables using PyInstaller:

### Installation

```bash
# Install PyInstaller (included in dev dependencies)
pip install -e ".[dev]"
```

### Building

**Important**: Executables are platform-specific. Build on the same OS you want to run on:
- Build on Linux → Linux executable
- Build on Windows → Windows .exe
- Build on macOS → macOS app

```bash
# Single-file executable
pyinstaller --onefile --name image-pipeline main.py

# Output will be in dist/image-pipeline (or dist/image-pipeline.exe on Windows)
```

### Running the Executable

```bash
# Linux/macOS
./dist/image-pipeline input.tiff output.png --filter blur:sigma=2.0

# Windows
dist\image-pipeline.exe input.tiff output.png --filter blur:sigma=2.0
```

**Note**: The executable will be 100-300 MB due to NumPy, Pillow, and scipy bundled inside.

## License

MIT

## Author

Dima Teterin (tet.dima.211@gmail.com)

## Roadmap

- [x] AVIF format support (completed)
- [x] Color space conversion (BT.709, BT.2020, Display P3) (completed)
- [x] Scene/display-referred workflows (completed)
- [ ] Metadata reading from PNG/AVIF (cICP, mDCv, cLLi chunks)
- [ ] Tone mapping operators (Reinhard, ACES, etc.)
- [ ] Brightness/contrast filters with MaxCLL/MaxFALL updates
- [ ] Batch processing via CLI
- [ ] Extended metadata (EXIF, ICC profiles)
- [ ] Additional color spaces (Lab, XYZ direct support)