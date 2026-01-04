# I/O Tests

Tests for image format readers and writers (round-trip testing).

## Test Strategy

Round-trip testing: Write → Read → Verify
1. Create ImageData with pixels + metadata
2. Write to file using format Writer
3. Read back using format Reader
4. Verify pixels and metadata are preserved

## PNG I/O Tests (`test_png_io.py`)

**23 tests covering:**

### Basic I/O (4 tests)
- 8-bit RGB, 16-bit RGB
- Grayscale (8-bit 2D)
- RGBA (with alpha channel)

### HDR Metadata (7 tests)
- **cICP chunk**: transfer_function (PQ, sRGB), color_space (BT.709, BT.2020)
- **cLLi chunk**: max_cll, max_fall (content light levels)
- **mDCv chunk**: mastering_display_max_luminance, mastering_display_min_luminance
- Complete HDR metadata set (all chunks together)
- **Additional chunks**: cHRM (chromaticity), sRGB (rendering intent)

### Bit Depth (3 tests)
- Standard 8-bit and 16-bit PNG
- Warning for non-standard bit_depth (10/12-bit)
- Note: PNG standard supports only 8 and 16 bits per sample

### Save Options (2 tests)
- Compression level variations
- Default options

### Validation (4 tests)
- Rejects float32 dtype (must use quantize filter first)
- Rejects empty arrays
- Accepts uint8 and uint16

### Edge Cases (3 tests)
- Large images (100x100)
- Single pixel (1x1)
- Basic metadata without HDR fields

## Key Features Tested

✅ **Lossless pixel preservation** (PNG is lossless)
✅ **HDR metadata round-trip** (cICP, cLLi, mDCv chunks)
✅ **Format validation** (dtype, shape, channels)
✅ **Multiple chunk writing** (cICP + cHRM + sRGB for BT.709)

## Known Limitations

⚠️ **bit_depth preservation**: PNG standard doesn't support storing logical bit_depth for sub-16bit data in uint16 arrays. 10-bit and 12-bit data stored in uint16 will read back as 16-bit.

## TODO

Future tests to add:
- TIFF I/O (multiple compression formats)
- AVIF I/O (HDR format with metadata)
- JPEG I/O (lossy, tolerance testing)
- WebP I/O (lossy/lossless modes)
- ICC profile round-trip testing
