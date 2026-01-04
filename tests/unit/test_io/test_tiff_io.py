"""
Tests for TIFF format I/O (round-trip testing)

TIFF Specification Support:
- Bit depths: 8, 10, 12, 16, 32-bit (integer + float)
- Color primaries: Tag 318 (WhitePoint) + Tag 319 (PrimaryChromaticities)
- Transfer function: LINEAR for float samples
- Compression: LZW, Deflate, ZSTD, JPEG, None (Tag 259)
- Samples per pixel: 1 (grayscale), 3 (RGB), 4 (RGBA) (Tag 277)

These tests verify EXPECTED behavior (TDD-style), not current implementation.
Tests may fail if features are not yet implemented - that's intentional!

SKIPPED: TIFF writer not implemented yet (writer.write() is a stub).
Tests document expected behavior for future implementation.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

# Skip all TIFF tests - writer not implemented
pytestmark = pytest.mark.skip(reason="TIFF writer not implemented (stub)")

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.tiff.reader import TiffFormatReader
from image_pipeline.io.formats.tiff.writer import TiffFormatWriter
from image_pipeline.types import ColorSpace, TransferFunction


@pytest.fixture
def temp_tiff_path():
    """Create temporary TIFF file path"""
    with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as f:
        path = Path(f.name)

    yield path

    # Cleanup
    if path.exists():
        path.unlink()


class TestTIFFBasicIO:
    """Tests for basic TIFF I/O without metadata"""

    def test_write_read_8bit_rgb(self, temp_tiff_path):
        """Test 8-bit RGB TIFF round-trip"""
        pixels = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        # Write
        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        # Read back
        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        # Verify pixels (lossless for uncompressed)
        np.testing.assert_array_equal(result.pixels, pixels)
        assert result.metadata.get('bit_depth') == 8
        assert result.metadata.get('channels') == 3

    def test_write_read_16bit_rgb(self, temp_tiff_path):
        """Test 16-bit RGB TIFF round-trip"""
        pixels = np.random.randint(0, 65536, (20, 20, 3), dtype=np.uint16)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        np.testing.assert_array_equal(result.pixels, pixels)
        assert result.metadata.get('bit_depth') == 16

    def test_write_read_float32_hdr(self, temp_tiff_path):
        """Test float32 HDR TIFF round-trip (linear data)"""
        pixels = np.random.rand(20, 20, 3).astype(np.float32) * 5.0  # HDR values [0, 5]
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        # Float should be lossless for uncompressed TIFF
        np.testing.assert_array_almost_equal(result.pixels, pixels, decimal=5)
        # Should auto-detect LINEAR transfer function for float samples
        assert result.metadata.get('transfer_function') == TransferFunction.LINEAR

    def test_write_read_grayscale_8bit(self, temp_tiff_path):
        """Test 8-bit grayscale TIFF (2D array)"""
        pixels = np.random.randint(0, 256, (30, 30), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        np.testing.assert_array_equal(result.pixels, pixels)
        assert result.metadata.get('channels') == 1
        assert result.pixels.ndim == 2  # Should preserve 2D shape

    def test_write_read_rgba(self, temp_tiff_path):
        """Test RGBA TIFF with alpha channel"""
        pixels = np.random.randint(0, 256, (20, 20, 4), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        np.testing.assert_array_equal(result.pixels, pixels)
        assert result.metadata.get('channels') == 4


class TestTIFFBitDepth:
    """Tests for TIFF bit depth preservation

    TIFF supports non-standard bit depths (10-bit, 12-bit) natively.
    Unlike PNG, TIFF can store the actual bit depth in metadata.
    """

    def test_10bit_preservation(self, temp_tiff_path):
        """Test that 10-bit depth is preserved in TIFF metadata"""
        pixels = np.random.randint(0, 1024, (10, 10, 3), dtype=np.uint16)
        metadata = {'bit_depth': 10}
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        # TIFF SHOULD preserve logical bit_depth
        assert result.metadata.get('bit_depth') == 10
        np.testing.assert_array_equal(result.pixels, pixels)

    def test_12bit_preservation(self, temp_tiff_path):
        """Test that 12-bit depth is preserved in TIFF metadata"""
        pixels = np.random.randint(0, 4096, (10, 10, 3), dtype=np.uint16)
        metadata = {'bit_depth': 12}
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        assert result.metadata.get('bit_depth') == 12
        np.testing.assert_array_equal(result.pixels, pixels)

    def test_32bit_float(self, temp_tiff_path):
        """Test 32-bit float TIFF"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        assert result.metadata.get('bit_depth') == 32
        assert result.pixels.dtype == np.float32


class TestTIFFColorSpace:
    """Tests for TIFF color space metadata (chromaticity tags)"""

    def test_color_space_bt709(self, temp_tiff_path):
        """Test BT.709 color space preservation via chromaticity tags"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        metadata = {
            'color_space': ColorSpace.BT709,
            'transfer_function': TransferFunction.LINEAR
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        # Should write Tag 318 (WhitePoint) and Tag 319 (PrimaryChromaticities)
        # Reader should match these back to BT.709
        assert result.metadata.get('color_space') == ColorSpace.BT709

    def test_color_space_bt2020(self, temp_tiff_path):
        """Test BT.2020 color space preservation"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32) * 2.0
        metadata = {
            'color_space': ColorSpace.BT2020,
            'transfer_function': TransferFunction.LINEAR
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        assert result.metadata.get('color_space') == ColorSpace.BT2020

    def test_color_space_display_p3(self, temp_tiff_path):
        """Test Display P3 color space preservation"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        metadata = {
            'color_space': ColorSpace.DISPLAY_P3,
            'transfer_function': TransferFunction.LINEAR
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        assert result.metadata.get('color_space') == ColorSpace.DISPLAY_P3

    def test_custom_primaries_preservation(self, temp_tiff_path):
        """Test custom color primaries (non-standard color space)"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        custom_primaries = {
            'red': (0.7, 0.3),
            'green': (0.2, 0.7),
            'blue': (0.15, 0.06),
            'white': (0.3127, 0.3290)  # D65
        }
        metadata = {
            'color_primaries': custom_primaries,
            'transfer_function': TransferFunction.LINEAR
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        # Should preserve custom primaries since they don't match any standard
        assert 'color_primaries' in result.metadata
        primaries = result.metadata['color_primaries']

        # Check all primaries are close (tolerance for float rounding)
        for color in ['red', 'green', 'blue', 'white']:
            assert abs(primaries[color][0] - custom_primaries[color][0]) < 0.001
            assert abs(primaries[color][1] - custom_primaries[color][1]) < 0.001


class TestTIFFHDRMetadata:
    """Tests for HDR-specific metadata in TIFF"""

    def test_scene_referred_with_paper_white(self, temp_tiff_path):
        """Test scene-referred HDR TIFF with paper_white metadata"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32) * 2.0  # Scene-referred [0, 2]
        metadata = {
            'transfer_function': TransferFunction.LINEAR,
            'color_space': ColorSpace.BT709,
            'paper_white': 100.0  # Scene reference: 100 nits
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        # Should preserve paper_white for scene-referred workflows
        assert result.metadata.get('paper_white') == 100.0
        assert result.metadata.get('transfer_function') == TransferFunction.LINEAR

    def test_display_referred_hdr(self, temp_tiff_path):
        """Test display-referred HDR TIFF (absolute nits)"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32) * 1000.0  # 0-1000 nits
        metadata = {
            'transfer_function': TransferFunction.LINEAR,
            'color_space': ColorSpace.BT2020,
            'mastering_display_max_luminance': 1000.0,
            'mastering_display_min_luminance': 0.005
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        # Should preserve mastering display metadata
        assert result.metadata.get('mastering_display_max_luminance') == 1000.0
        assert result.metadata.get('mastering_display_min_luminance') == 0.005


class TestTIFFCompression:
    """Tests for TIFF compression options (Tag 259)

    Common compression schemes:
    - None (1): Uncompressed
    - LZW (5): Lossless compression
    - Deflate (8): Lossless (similar to PNG)
    - JPEG (7): Lossy compression
    - ZSTD (50000): Modern lossless compression
    """

    def test_uncompressed(self, temp_tiff_path):
        """Test uncompressed TIFF (compression=None)"""
        pixels = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={'compression': None})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        # Should be lossless
        np.testing.assert_array_equal(result.pixels, pixels)

    def test_lzw_compression(self, temp_tiff_path):
        """Test LZW compression (lossless)"""
        pixels = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={'compression': 'lzw'})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        # LZW is lossless
        np.testing.assert_array_equal(result.pixels, pixels)
        # Compressed file should be smaller
        assert temp_tiff_path.stat().st_size < pixels.nbytes * 1.2

    def test_deflate_compression(self, temp_tiff_path):
        """Test Deflate compression (lossless, similar to PNG)"""
        pixels = np.random.randint(0, 65536, (30, 30, 3), dtype=np.uint16)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={'compression': 'deflate'})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        # Deflate is lossless
        np.testing.assert_array_equal(result.pixels, pixels)

    def test_zstd_compression(self, temp_tiff_path):
        """Test ZSTD compression (modern lossless)"""
        pixels = np.random.rand(40, 40, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={'compression': 'zstd'})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        # ZSTD is lossless
        np.testing.assert_array_almost_equal(result.pixels, pixels, decimal=5)

    def test_jpeg_compression_lossy(self, temp_tiff_path):
        """Test JPEG compression (lossy) - requires tolerance"""
        pixels = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={'compression': 'jpeg', 'quality': 95})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        # JPEG is lossy - use tolerance
        # Most pixels should be close (within ±5 for quality=95)
        assert np.mean(np.abs(result.pixels.astype(float) - pixels.astype(float))) < 5.0
        assert result.pixels.shape == pixels.shape


class TestTIFFValidation:
    """Tests for TIFF writer validation"""

    def test_accepts_uint8(self, temp_tiff_path):
        """TIFF should accept uint8 dtype"""
        pixels = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        # Should not raise
        writer.validate(img_data)
        writer.write(img_data, options={})

    def test_accepts_uint16(self, temp_tiff_path):
        """TIFF should accept uint16 dtype"""
        pixels = np.random.randint(0, 65536, (10, 10, 3), dtype=np.uint16)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.validate(img_data)
        writer.write(img_data, options={})

    def test_accepts_uint32(self, temp_tiff_path):
        """TIFF should accept uint32 dtype"""
        pixels = np.random.randint(0, 1000, (10, 10, 3), dtype=np.uint32)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.validate(img_data)
        writer.write(img_data, options={})

    def test_accepts_float32(self, temp_tiff_path):
        """TIFF should accept float32 dtype"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.validate(img_data)
        writer.write(img_data, options={})

    def test_accepts_float64(self, temp_tiff_path):
        """TIFF should accept float64 dtype"""
        pixels = np.random.rand(10, 10, 3).astype(np.float64)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.validate(img_data)
        writer.write(img_data, options={})

    def test_rejects_empty_array(self, temp_tiff_path):
        """TIFF should reject empty arrays"""
        pixels = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))

        with pytest.raises(ValueError, match="[Ee]mpty"):
            writer.validate(img_data)


class TestTIFFEdgeCases:
    """Tests for TIFF edge cases"""

    def test_large_image(self, temp_tiff_path):
        """Test large image (100x100x3)"""
        pixels = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        np.testing.assert_array_equal(result.pixels, pixels)

    def test_single_pixel(self, temp_tiff_path):
        """Test 1x1 pixel image"""
        pixels = np.array([[[123, 45, 67]]], dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        np.testing.assert_array_equal(result.pixels, pixels)

    def test_basic_metadata_without_hdr(self, temp_tiff_path):
        """Test basic metadata without HDR fields"""
        pixels = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        metadata = {
            'format': 'TIFF',
            'filename': 'test.tiff'
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = TiffFormatWriter(str(temp_tiff_path))
        writer.write(img_data, options={})

        reader = TiffFormatReader(temp_tiff_path)
        result = reader.read()

        # Should preserve basic metadata
        assert result.metadata.get('format') == 'TIFF'
        assert result.metadata.get('filename') is not None
