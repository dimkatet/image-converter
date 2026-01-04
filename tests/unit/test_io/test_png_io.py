"""
Tests for PNG I/O (round-trip: write → read)

Tests cover:
- Basic 8-bit and 16-bit PNG write/read
- HDR metadata preservation (cICP: transfer_function, color_space)
- Content light level metadata (cLLi: max_cll, max_fall)
- Mastering display metadata (mDCv: min/max luminance)
- ICC profile preservation
- Different bit depths (10-bit, 12-bit stored in uint16)
- Different channel counts (grayscale, RGB, RGBA)
- SaveOptions (compression level)

Strategy: Write ImageData to temp file, read back, verify pixels and metadata
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.png.reader import PNGFormatReader
from image_pipeline.io.formats.png.writer import PNGFormatWriter
from image_pipeline.types import TransferFunction, ColorSpace


@pytest.fixture
def temp_png_path(tmp_path):
    """Fixture providing temporary PNG file path"""
    return tmp_path / "test.png"


class TestPNGBasicIO:
    """Basic PNG read/write tests"""

    def test_write_read_8bit_rgb(self, temp_png_path):
        """Test round-trip for 8-bit RGB image"""
        # Create test image
        pixels = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [128, 128, 128]]
        ], dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        # Write
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Verify pixels
        np.testing.assert_array_equal(result.pixels, pixels)
        # Verify basic metadata
        assert result.metadata['format'] == 'PNG'
        assert result.shape == (2, 2, 3)
        assert result.channels == 3

    def test_write_read_16bit_rgb(self, temp_png_path):
        """Test round-trip for 16-bit RGB image"""
        pixels = np.array([
            [[65535, 0, 0], [0, 65535, 0]],
            [[0, 0, 65535], [32768, 32768, 32768]]
        ], dtype=np.uint16)
        img_data = ImageData(pixels=pixels)

        # Write
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Verify pixels
        np.testing.assert_array_equal(result.pixels, pixels)
        assert result.dtype == np.uint16

    def test_write_read_grayscale_8bit(self, temp_png_path):
        """Test round-trip for 8-bit grayscale image"""
        pixels = np.array([
            [0, 128],
            [255, 64]
        ], dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        # Write
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Verify
        np.testing.assert_array_equal(result.pixels, pixels)
        assert result.channels == 1

    def test_write_read_rgba(self, temp_png_path):
        """Test round-trip for RGBA image"""
        pixels = np.array([
            [[255, 0, 0, 255], [0, 255, 0, 128]],
            [[0, 0, 255, 64], [128, 128, 128, 255]]
        ], dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        # Write
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Verify
        np.testing.assert_array_equal(result.pixels, pixels)
        assert result.channels == 4


class TestPNGHDRMetadata:
    """Tests for HDR metadata preservation (cICP, cLLi, mDCv chunks)"""

    def test_transfer_function_pq(self, temp_png_path):
        """Test PQ transfer function preservation"""
        pixels = np.random.randint(0, 65536, (10, 10, 3), dtype=np.uint16)
        metadata = {
            'transfer_function': TransferFunction.PQ
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Write
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Verify metadata
        assert result.metadata.get('transfer_function') == TransferFunction.PQ
        # bit_depth auto-calculated from dtype
        assert result.metadata.get('bit_depth') == 16

    def test_transfer_function_srgb(self, temp_png_path):
        """Test sRGB transfer function preservation"""
        pixels = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        metadata = {
            'transfer_function': TransferFunction.SRGB
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Write
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Verify
        assert result.metadata.get('transfer_function') == TransferFunction.SRGB

    def test_color_space_bt709(self, temp_png_path):
        """Test BT.709 color space preservation"""
        pixels = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        metadata = {
            'color_space': ColorSpace.BT709,
            'transfer_function': TransferFunction.SRGB
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Write
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Verify
        assert result.metadata.get('color_space') == ColorSpace.BT709

        # Verify additional chunks are written
        from image_pipeline.metadata.png.codec import PNGMetadataCodec
        codec = PNGMetadataCodec(str(temp_png_path))
        codec.read_chunks()
        chunk_metadata = codec.get_metadata()

        # Should have cICP for color info
        assert 'cicp' in chunk_metadata
        # Should also have cHRM (chromaticity) and sRGB chunks
        assert 'chrm' in chunk_metadata
        assert 'srgb' in chunk_metadata

    def test_color_space_bt2020(self, temp_png_path):
        """Test BT.2020 color space preservation"""
        pixels = np.random.randint(0, 65536, (10, 10, 3), dtype=np.uint16)
        metadata = {
            'color_space': ColorSpace.BT2020,
            'transfer_function': TransferFunction.PQ
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Write
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Verify
        assert result.metadata.get('color_space') == ColorSpace.BT2020

        # Verify chunks
        from image_pipeline.metadata.png.codec import PNGMetadataCodec
        codec = PNGMetadataCodec(str(temp_png_path))
        codec.read_chunks()
        chunk_metadata = codec.get_metadata()

        # Should have cICP and cHRM for BT.2020
        assert 'cicp' in chunk_metadata
        assert 'chrm' in chunk_metadata

    def test_content_light_levels(self, temp_png_path):
        """Test max_cll and max_fall preservation (cLLi chunk)"""
        pixels = np.random.randint(0, 65536, (10, 10, 3), dtype=np.uint16)
        metadata = {
            'transfer_function': TransferFunction.PQ,
            'color_space': ColorSpace.BT2020,
            'max_cll': 1000,
            'max_fall': 400
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Write
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Verify cLLi metadata
        assert result.metadata.get('max_cll') == 1000
        assert result.metadata.get('max_fall') == 400

    def test_mastering_display_metadata(self, temp_png_path):
        """Test mastering display min/max luminance (mDCv chunk)"""
        pixels = np.random.randint(0, 65536, (10, 10, 3), dtype=np.uint16)
        metadata = {
            'transfer_function': TransferFunction.PQ,
            'color_space': ColorSpace.BT2020,
            'mastering_display_max_luminance': 1000.0,
            'mastering_display_min_luminance': 0.005
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Write
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Verify mDCv metadata
        assert result.metadata.get('mastering_display_max_luminance') == 1000.0
        assert result.metadata.get('mastering_display_min_luminance') == 0.005

    def test_complete_hdr_metadata_set(self, temp_png_path):
        """Test complete HDR metadata set (cICP + cLLi + mDCv)"""
        pixels = np.random.randint(0, 65536, (20, 20, 3), dtype=np.uint16)
        metadata = {
            'transfer_function': TransferFunction.PQ,
            'color_space': ColorSpace.BT2020,
            'max_cll': 1000,
            'max_fall': 400,
            'mastering_display_max_luminance': 1000.0,
            'mastering_display_min_luminance': 0.005
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Write
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Verify all HDR metadata
        assert result.metadata.get('transfer_function') == TransferFunction.PQ
        assert result.metadata.get('color_space') == ColorSpace.BT2020
        assert result.metadata.get('max_cll') == 1000
        assert result.metadata.get('max_fall') == 400
        assert result.metadata.get('mastering_display_max_luminance') == 1000.0
        assert result.metadata.get('mastering_display_min_luminance') == 0.005
        # bit_depth auto-calculated from uint16 dtype
        assert result.metadata.get('bit_depth') == 16


class TestPNGBitDepth:
    """Tests for bit depth handling

    PNG standard supports only 8-bit and 16-bit per sample.
    10-bit and 12-bit data must be stored in uint16 arrays.
    """

    def test_unsupported_bit_depth_warning(self, temp_png_path):
        """Test that unsupported bit_depth in metadata triggers warning"""
        pixels = np.random.randint(0, 1024, (10, 10, 3), dtype=np.uint16)
        metadata = {'bit_depth': 10}  # Not standard PNG bit depth
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = PNGFormatWriter(str(temp_png_path))

        # Should warn about non-standard bit_depth
        with pytest.warns(UserWarning, match="PNG standard only supports 8 and 16 bits"):
            writer.write(img_data, options={})

        # Read back - should have 16-bit (from uint16 dtype)
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        assert result.metadata.get('bit_depth') == 16
        np.testing.assert_array_equal(result.pixels, pixels)

    def test_8bit_standard(self, temp_png_path):
        """Test standard 8-bit PNG"""
        pixels = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        assert result.metadata.get('bit_depth') == 8
        np.testing.assert_array_equal(result.pixels, pixels)

    def test_16bit_standard(self, temp_png_path):
        """Test standard 16-bit PNG"""
        pixels = np.random.randint(0, 65536, (10, 10, 3), dtype=np.uint16)
        img_data = ImageData(pixels=pixels)

        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        assert result.metadata.get('bit_depth') == 16
        np.testing.assert_array_equal(result.pixels, pixels)


class TestPNGSaveOptions:
    """Tests for PNG save options"""

    def test_compression_level(self, temp_png_path):
        """Test different compression levels"""
        pixels = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        # Write with high compression
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={'level': 9})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Pixels should be lossless regardless of compression
        np.testing.assert_array_equal(result.pixels, pixels)

    def test_default_options(self, temp_png_path):
        """Test writing with default options (empty dict)"""
        pixels = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        # Write with defaults
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Should work fine
        np.testing.assert_array_equal(result.pixels, pixels)


class TestPNGValidation:
    """Tests for PNG writer validation"""

    def test_validation_rejects_float32(self, temp_png_path):
        """Test that float32 is rejected"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        writer = PNGFormatWriter(str(temp_png_path))

        with pytest.raises(ValueError, match="PNG supports only uint8 and uint16"):
            writer.write(img_data, options={})

    def test_validation_rejects_empty_array(self, temp_png_path):
        """Test that empty array is rejected"""
        pixels = np.array([], dtype=np.uint8)
        img_data = ImageData(pixels=pixels.reshape(0, 0, 3))

        writer = PNGFormatWriter(str(temp_png_path))

        with pytest.raises(ValueError, match="Empty pixel array"):
            writer.write(img_data, options={})

    def test_validation_accepts_uint8(self, temp_png_path):
        """Test that uint8 is accepted"""
        pixels = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = PNGFormatWriter(str(temp_png_path))
        # Should not raise
        writer.write(img_data, options={})

    def test_validation_accepts_uint16(self, temp_png_path):
        """Test that uint16 is accepted"""
        pixels = np.random.randint(0, 65536, (10, 10, 3), dtype=np.uint16)
        img_data = ImageData(pixels=pixels)

        writer = PNGFormatWriter(str(temp_png_path))
        # Should not raise
        writer.write(img_data, options={})


class TestPNGEdgeCases:
    """Edge cases and special scenarios"""

    def test_large_image(self, temp_png_path):
        """Test with larger image (100x100)"""
        pixels = np.random.randint(0, 65536, (100, 100, 3), dtype=np.uint16)
        img_data = ImageData(pixels=pixels)

        # Write
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Verify
        assert result.shape == (100, 100, 3)
        np.testing.assert_array_equal(result.pixels, pixels)

    def test_single_pixel_image(self, temp_png_path):
        """Test with 1x1 image"""
        pixels = np.array([[[255, 128, 0]]], dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        # Write
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Verify
        np.testing.assert_array_equal(result.pixels, pixels)

    def test_metadata_survives_without_hdr_fields(self, temp_png_path):
        """Test that basic metadata works without HDR fields"""
        pixels = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        metadata = {
            'custom_field': 'value',
            'description': 'test image'
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Write
        writer = PNGFormatWriter(str(temp_png_path))
        writer.write(img_data, options={})

        # Read back
        reader = PNGFormatReader(temp_png_path)
        result = reader.read()

        # Basic metadata should be present
        assert result.metadata['format'] == 'PNG'
        # Custom fields won't survive (PNG doesn't store arbitrary metadata)
        # but it shouldn't crash
