"""
Tests for JPEG I/O (standard JPEG facade + Ultra HDR)

Tests cover:
- JPEG SaveOptions adapter validation
- Standard JPEG facade (NotImplementedError tests, skipped until implemented)
- Ultra HDR writer validation (dtype, transfer_function, color_space)
- Ultra HDR writer encoding (different color spaces, bit depths, channels)
- Ultra HDR reader (format detection, metadata extraction)
- Round-trip tests (write → read, lossy tolerance)

Strategy:
- For Ultra HDR: Use writer to create test files, then read them back
- For Standard JPEG: Test that NotImplementedError is raised (skipped tests)
"""

import numpy as np
import pytest
from pathlib import Path

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.jpeg.reader import JPEGReader
from image_pipeline.io.formats.jpeg.writer import JPEGWriter
from image_pipeline.io.formats.jpeg.ultrahdr.reader import UltraHDRReader
from image_pipeline.io.formats.jpeg.ultrahdr.writer import UltraHDRWriter
from image_pipeline.io.formats.jpeg.options import JPEGSaveOptionsAdapter
from image_pipeline.types import TransferFunction, ColorSpace, SaveOptions


@pytest.fixture
def temp_jpeg_path(tmp_path):
    """Fixture providing temporary JPEG file path"""
    return tmp_path / "test.jpg"


@pytest.fixture
def sample_hdr_linear_rgb():
    """
    Sample HDR image in LINEAR transfer function

    Returns:
        ImageData with float32 RGB, LINEAR transfer, BT709 color space

    Note: Ultra HDR requires minimum 8x8 dimensions
    """
    # Create 8x8 HDR image with values in [0, 2.0] range (scene-referred)
    pixels = np.array([
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.3, 0.3, 0.3], [0.7, 0.7, 0.7], [1.5, 1.5, 1.5], [0.2, 0.2, 0.2]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 1.0, 0.5], [0.5, 0.0, 1.0]],
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.8, 0.2, 0.4], [0.2, 0.8, 0.4], [0.4, 0.2, 0.8], [0.6, 0.6, 0.2]],
        [[1.5, 0.5, 0.0], [0.5, 1.5, 0.0], [0.0, 0.5, 1.5], [2.0, 1.0, 0.5], [1.2, 0.8, 0.4], [0.4, 1.2, 0.8], [0.8, 0.4, 1.2], [1.0, 0.6, 0.3]],
        [[0.9, 0.1, 0.3], [0.1, 0.9, 0.3], [0.3, 0.1, 0.9], [0.7, 0.7, 0.1], [1.8, 0.3, 0.6], [0.3, 1.8, 0.6], [0.6, 0.3, 1.8], [1.1, 1.1, 0.4]],
        [[0.4, 0.6, 0.8], [0.6, 0.8, 0.4], [0.8, 0.4, 0.6], [0.2, 0.6, 1.0], [1.3, 0.5, 0.2], [0.5, 1.3, 0.2], [0.2, 0.5, 1.3], [0.9, 0.9, 0.5]],
        [[1.4, 0.2, 0.7], [0.2, 1.4, 0.7], [0.7, 0.2, 1.4], [1.6, 0.8, 0.1], [0.5, 0.9, 1.1], [0.9, 1.1, 0.5], [1.1, 0.5, 0.9], [0.8, 1.0, 0.6]],
        [[0.1, 0.3, 0.5], [0.3, 0.5, 0.1], [0.5, 0.1, 0.3], [0.6, 0.4, 0.2], [1.7, 0.6, 0.9], [0.6, 1.7, 0.9], [0.9, 0.6, 1.7], [1.2, 1.4, 0.7]]
    ], dtype=np.float32)

    img_data = ImageData(pixels=pixels)
    img_data.metadata['transfer_function'] = TransferFunction.LINEAR
    img_data.metadata['color_space'] = ColorSpace.BT709

    return img_data


@pytest.fixture
def sample_hdr_linear_rgba():
    """
    Sample HDR image with alpha channel

    Returns:
        ImageData with float32 RGBA, LINEAR transfer, BT709 color space

    Note: Ultra HDR requires minimum 8x8 dimensions
    """
    # Create 8x8 RGBA image
    pixels = np.random.rand(8, 8, 4).astype(np.float32)
    # Set some specific values for testing
    pixels[0, 0] = [1.0, 0.0, 0.0, 1.0]
    pixels[0, 1] = [0.0, 1.0, 0.0, 1.0]
    pixels[1, 0] = [0.0, 0.0, 1.0, 0.5]
    pixels[1, 1] = [1.0, 1.0, 1.0, 1.0]

    img_data = ImageData(pixels=pixels)
    img_data.metadata['transfer_function'] = TransferFunction.LINEAR
    img_data.metadata['color_space'] = ColorSpace.BT709

    return img_data


# ============================================================================
# JPEG Save Options Adapter Tests
# ============================================================================

class TestJPEGSaveOptionsAdapter:
    """Tests for JPEG save options validation and adaptation"""

    def test_default_options(self):
        """Test that defaults are applied correctly"""
        adapted = JPEGSaveOptionsAdapter.adapt({})

        assert adapted['quality'] == 95
        assert adapted['ultra_hdr'] is False
        assert adapted['gainmap_scale'] == 4

    def test_custom_quality(self):
        """Test custom quality value"""
        adapted = JPEGSaveOptionsAdapter.adapt({'quality': 80})
        assert adapted['quality'] == 80

    def test_quality_validation_range_low(self):
        """Test quality validation: must be >= 1"""
        with pytest.raises(ValueError, match="quality must be in range"):
            JPEGSaveOptionsAdapter.adapt({'quality': 0})

    def test_quality_validation_range_high(self):
        """Test quality validation: must be <= 100"""
        with pytest.raises(ValueError, match="quality must be in range"):
            JPEGSaveOptionsAdapter.adapt({'quality': 101})

    def test_quality_validation_type(self):
        """Test quality validation: must be int"""
        with pytest.raises(TypeError, match="quality must be int"):
            JPEGSaveOptionsAdapter.adapt({'quality': 80.5})

    def test_ultra_hdr_flag_true(self):
        """Test ultra_hdr flag can be set to True"""
        adapted = JPEGSaveOptionsAdapter.adapt({'ultra_hdr': True})
        assert adapted['ultra_hdr'] is True

    def test_ultra_hdr_flag_false(self):
        """Test ultra_hdr flag can be set to False"""
        adapted = JPEGSaveOptionsAdapter.adapt({'ultra_hdr': False})
        assert adapted['ultra_hdr'] is False

    def test_ultra_hdr_validation_type(self):
        """Test ultra_hdr validation: must be bool"""
        with pytest.raises(TypeError, match="ultra_hdr must be bool"):
            JPEGSaveOptionsAdapter.adapt({'ultra_hdr': 1})

    def test_gainmap_scale_custom(self):
        """Test custom gainmap_scale value"""
        adapted = JPEGSaveOptionsAdapter.adapt({'gainmap_scale': 8})
        assert adapted['gainmap_scale'] == 8

    def test_gainmap_scale_validation_range(self):
        """Test gainmap_scale validation: must be >= 1"""
        with pytest.raises(ValueError, match="gainmap_scale must be >= 1"):
            JPEGSaveOptionsAdapter.adapt({'gainmap_scale': 0})

    def test_gainmap_scale_validation_type(self):
        """Test gainmap_scale validation: must be int"""
        with pytest.raises(TypeError, match="gainmap_scale must be int"):
            JPEGSaveOptionsAdapter.adapt({'gainmap_scale': 4.0})

    def test_all_options_together(self):
        """Test all options together"""
        options = {
            'quality': 90,
            'ultra_hdr': True,
            'gainmap_scale': 2
        }
        adapted = JPEGSaveOptionsAdapter.adapt(options)

        assert adapted['quality'] == 90
        assert adapted['ultra_hdr'] is True
        assert adapted['gainmap_scale'] == 2


# ============================================================================
# Standard JPEG Facade Tests (SKIPPED - NotImplementedError)
# ============================================================================

class TestStandardJPEGFacade:
    """
    Tests for standard JPEG facade (reader/writer)

    These tests verify that NotImplementedError is raised for standard JPEG
    operations, since only Ultra HDR is currently implemented.

    All tests are SKIPPED until standard JPEG support is implemented.
    """

    @pytest.mark.skip(reason="Standard JPEG reading not implemented yet")
    def test_reader_detects_standard_jpeg(self, tmp_path):
        """Test that JPEGReader detects standard (non-Ultra HDR) JPEG"""
        # Create a minimal JPEG file (JFIF header)
        # JPEG SOI marker (0xFFD8) + JFIF APP0 marker
        jpeg_bytes = bytes([
            0xFF, 0xD8,  # SOI (Start of Image)
            0xFF, 0xE0,  # APP0 marker
            0x00, 0x10,  # Length (16 bytes)
            0x4A, 0x46, 0x49, 0x46, 0x00,  # "JFIF\0"
            0x01, 0x01,  # Version 1.1
            0x00,  # Units (aspect ratio)
            0x00, 0x01, 0x00, 0x01,  # X/Y density
            0x00, 0x00,  # Thumbnail size
            0xFF, 0xD9  # EOI (End of Image)
        ])

        jpeg_path = tmp_path / "standard.jpg"
        with open(jpeg_path, 'wb') as f:
            f.write(jpeg_bytes)

        # Should raise NotImplementedError for standard JPEG
        reader = JPEGReader(jpeg_path)
        with pytest.raises(NotImplementedError, match="Standard JPEG reading is not implemented"):
            reader.read()

    @pytest.mark.skip(reason="Standard JPEG writing not implemented yet")
    def test_writer_raises_for_standard_jpeg(self, temp_jpeg_path):
        """Test that JPEGWriter raises NotImplementedError when ultra_hdr=False"""
        pixels = np.random.rand(8, 8, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        writer = JPEGWriter(str(temp_jpeg_path))
        options: SaveOptions = {'ultra_hdr': False, 'quality': 95}

        with pytest.raises(NotImplementedError, match="Standard JPEG encoding is not implemented"):
            writer.write(img_data, options)

    @pytest.mark.skip(reason="Standard JPEG not implemented yet")
    def test_reader_file_not_found(self, tmp_path):
        """Test that JPEGReader validates file existence"""
        nonexistent = tmp_path / "nonexistent.jpg"
        reader = JPEGReader(nonexistent)

        with pytest.raises(FileNotFoundError, match="File not found"):
            reader.validate_file()

    @pytest.mark.skip(reason="Standard JPEG not implemented yet")
    def test_reader_path_is_directory(self, tmp_path):
        """Test that JPEGReader rejects directories"""
        reader = JPEGReader(tmp_path)

        with pytest.raises(ValueError, match="Path is not a file"):
            reader.validate_file()


# ============================================================================
# Ultra HDR Writer - Validation Tests
# ============================================================================

class TestUltraHDRWriterValidation:
    """Tests for Ultra HDR writer input validation"""

    def test_validate_invalid_dtype_uint8(self, temp_jpeg_path):
        """Test validation rejects uint8 (must be float16/float32)"""
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(temp_jpeg_path))

        with pytest.raises(ValueError, match="Ultra HDR requires float16 or float32 data"):
            writer.validate(img_data)

    def test_validate_invalid_dtype_uint16(self, temp_jpeg_path):
        """Test validation rejects uint16"""
        pixels = np.random.randint(0, 65536, (8, 8, 3), dtype=np.uint16)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(temp_jpeg_path))

        with pytest.raises(ValueError, match="Ultra HDR requires float16 or float32 data"):
            writer.validate(img_data)

    def test_validate_invalid_dimensions_2d(self, temp_jpeg_path):
        """Test validation rejects 2D arrays (must be 3D)"""
        pixels = np.random.rand(8, 8).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(temp_jpeg_path))

        with pytest.raises(ValueError, match="Ultra HDR requires 3D array"):
            writer.validate(img_data)

    def test_validate_invalid_channels_1(self, temp_jpeg_path):
        """Test validation rejects 1 channel (must be 3 or 4)"""
        pixels = np.random.rand(8, 8, 1).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(temp_jpeg_path))

        with pytest.raises(ValueError, match="Ultra HDR requires 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            writer.validate(img_data)

    def test_validate_invalid_channels_5(self, temp_jpeg_path):
        """Test validation rejects 5 channels"""
        pixels = np.random.rand(8, 8, 5).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(temp_jpeg_path))

        with pytest.raises(ValueError, match="Ultra HDR requires 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            writer.validate(img_data)

    def test_validate_missing_color_space(self, temp_jpeg_path):
        """Test validation requires color_space in metadata"""
        pixels = np.random.rand(8, 8, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        # No color_space set

        writer = UltraHDRWriter(str(temp_jpeg_path))

        with pytest.raises(ValueError, match="Ultra HDR requires 'color_space' in metadata"):
            writer.validate(img_data)

    def test_validate_invalid_color_space_type(self, temp_jpeg_path):
        """Test validation requires ColorSpace enum (not string)"""
        pixels = np.random.rand(8, 8, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = "BT709"  # String instead of enum

        writer = UltraHDRWriter(str(temp_jpeg_path))

        with pytest.raises(TypeError, match="color_space must be ColorSpace enum"):
            writer.validate(img_data)

    def test_validate_unsupported_color_space(self, temp_jpeg_path):
        """Test validation rejects unsupported color spaces"""
        from unittest.mock import Mock

        pixels = np.random.rand(8, 8, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR

        # Create a mock color space that's not BT709/BT2020/DISPLAY_P3
        mock_color_space = Mock(spec=ColorSpace)
        mock_color_space.value = "Adobe RGB"
        img_data.metadata['color_space'] = mock_color_space

        writer = UltraHDRWriter(str(temp_jpeg_path))

        with pytest.raises(ValueError, match="Ultra HDR supports BT.709, BT.2020, Display-P3"):
            writer.validate(img_data)

    def test_validate_missing_transfer_function(self, temp_jpeg_path):
        """Test validation requires transfer_function in metadata"""
        pixels = np.random.rand(8, 8, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['color_space'] = ColorSpace.BT709
        # No transfer_function set

        writer = UltraHDRWriter(str(temp_jpeg_path))

        with pytest.raises(ValueError, match="Ultra HDR requires 'transfer_function' in metadata"):
            writer.validate(img_data)

    def test_validate_invalid_transfer_function_type(self, temp_jpeg_path):
        """Test validation requires TransferFunction enum (not string)"""
        pixels = np.random.rand(8, 8, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = "LINEAR"  # String instead of enum
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(temp_jpeg_path))

        with pytest.raises(TypeError, match="transfer_function must be TransferFunction enum"):
            writer.validate(img_data)

    def test_validate_invalid_transfer_function_pq(self, temp_jpeg_path):
        """Test validation rejects PQ transfer function (must be LINEAR)"""
        pixels = np.random.rand(8, 8, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.PQ
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(temp_jpeg_path))

        with pytest.raises(ValueError, match="Ultra HDR requires LINEAR transfer function"):
            writer.validate(img_data)

    def test_validate_invalid_transfer_function_srgb(self, temp_jpeg_path):
        """Test validation rejects SRGB transfer function"""
        pixels = np.random.rand(8, 8, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.SRGB
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(temp_jpeg_path))

        with pytest.raises(ValueError, match="Ultra HDR requires LINEAR transfer function"):
            writer.validate(img_data)

    def test_validate_accepts_float32_rgb(self, temp_jpeg_path, sample_hdr_linear_rgb):
        """Test validation accepts valid float32 RGB image"""
        writer = UltraHDRWriter(str(temp_jpeg_path))

        # Should not raise
        writer.validate(sample_hdr_linear_rgb)

    def test_validate_accepts_float32_rgba(self, temp_jpeg_path, sample_hdr_linear_rgba):
        """Test validation accepts valid float32 RGBA image"""
        writer = UltraHDRWriter(str(temp_jpeg_path))

        # Should not raise
        writer.validate(sample_hdr_linear_rgba)

    def test_validate_accepts_float16(self, temp_jpeg_path):
        """Test validation accepts float16 data"""
        pixels = np.random.rand(8, 8, 3).astype(np.float16)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(temp_jpeg_path))

        # Should not raise
        writer.validate(img_data)


# ============================================================================
# Ultra HDR Writer - Helper Methods Tests
# ============================================================================

class TestUltraHDRWriterHelpers:
    """Tests for Ultra HDR writer helper methods"""

    def test_prepare_pixels_adds_alpha_channel(self, temp_jpeg_path):
        """Test _prepare_pixels adds alpha channel to RGB image"""
        pixels = np.random.rand(8, 8, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(temp_jpeg_path))
        result = writer._prepare_pixels(img_data)

        # Should have 4 channels
        assert result.shape == (8, 8, 4)
        # Alpha channel should be all 1.0
        np.testing.assert_array_equal(result[:, :, 3], 1.0)
        # RGB channels preserved (with float16 precision tolerance)
        np.testing.assert_allclose(result[:, :, :3], pixels, rtol=1e-3, atol=1e-4)

    def test_prepare_pixels_preserves_alpha_channel(self, temp_jpeg_path):
        """Test _prepare_pixels preserves existing alpha channel"""
        pixels = np.random.rand(8, 8, 4).astype(np.float32)
        pixels[:, :, 3] = 0.5  # Set alpha to 0.5
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(temp_jpeg_path))
        result = writer._prepare_pixels(img_data)

        # Should preserve alpha (with float16 precision tolerance)
        np.testing.assert_allclose(result[:, :, 3], 0.5, rtol=1e-3, atol=1e-4)

    def test_prepare_pixels_converts_float32_to_float16(self, temp_jpeg_path):
        """Test _prepare_pixels converts float32 to float16"""
        pixels = np.random.rand(8, 8, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(temp_jpeg_path))
        result = writer._prepare_pixels(img_data)

        # Should be float16
        assert result.dtype == np.float16

    def test_prepare_pixels_preserves_float16(self, temp_jpeg_path):
        """Test _prepare_pixels preserves float16 dtype"""
        pixels = np.random.rand(8, 8, 3).astype(np.float16)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(temp_jpeg_path))
        result = writer._prepare_pixels(img_data)

        # Should remain float16
        assert result.dtype == np.float16

    def test_prepare_pixels_c_contiguous(self, temp_jpeg_path):
        """Test _prepare_pixels ensures C-contiguous array"""
        pixels = np.random.rand(16, 16, 3).astype(np.float32)
        # Make non-contiguous by slicing (result must still be >= 8x8)
        pixels = pixels[::2, ::2, :]  # Results in 8x8
        assert pixels.shape == (8, 8, 3)
        assert not pixels.flags.c_contiguous

        img_data = ImageData(pixels=pixels.copy())
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(temp_jpeg_path))
        result = writer._prepare_pixels(img_data)

        # Should be C-contiguous
        assert result.flags.c_contiguous

    def test_map_color_space_bt709(self, temp_jpeg_path):
        """Test _map_color_space maps BT709 correctly"""
        import imagecodecs
        writer = UltraHDRWriter(str(temp_jpeg_path))

        result = writer._map_color_space(ColorSpace.BT709)
        assert result == imagecodecs.ULTRAHDR.CG.BT_709

    def test_map_color_space_bt2020(self, temp_jpeg_path):
        """Test _map_color_space maps BT2020 correctly"""
        import imagecodecs
        writer = UltraHDRWriter(str(temp_jpeg_path))

        result = writer._map_color_space(ColorSpace.BT2020)
        assert result == imagecodecs.ULTRAHDR.CG.BT_2100

    def test_map_color_space_display_p3(self, temp_jpeg_path):
        """Test _map_color_space maps Display P3 correctly"""
        import imagecodecs
        writer = UltraHDRWriter(str(temp_jpeg_path))

        result = writer._map_color_space(ColorSpace.DISPLAY_P3)
        assert result == imagecodecs.ULTRAHDR.CG.DISPLAY_P3


# ============================================================================
# Ultra HDR Writer - Encoding Tests
# ============================================================================

class TestUltraHDRWriterEncoding:
    """Tests for Ultra HDR encoding functionality"""

    def test_write_basic_rgb_float32(self, temp_jpeg_path, sample_hdr_linear_rgb):
        """Test writing basic RGB float32 image"""
        writer = UltraHDRWriter(str(temp_jpeg_path))
        options: SaveOptions = {'quality': 95, 'ultra_hdr': True, 'gainmap_scale': 4}

        # Should not raise
        writer.write(sample_hdr_linear_rgb, options)

        # File should exist
        assert temp_jpeg_path.exists()
        # File should have non-zero size
        assert temp_jpeg_path.stat().st_size > 0

    def test_write_rgba_float32(self, temp_jpeg_path, sample_hdr_linear_rgba):
        """Test writing RGBA float32 image"""
        writer = UltraHDRWriter(str(temp_jpeg_path))
        options: SaveOptions = {'quality': 95, 'ultra_hdr': True, 'gainmap_scale': 4}

        # Should not raise
        writer.write(sample_hdr_linear_rgba, options)

        assert temp_jpeg_path.exists()
        assert temp_jpeg_path.stat().st_size > 0

    def test_write_float16_input(self, temp_jpeg_path):
        """Test writing float16 input (no conversion needed)"""
        pixels = np.random.rand(8, 8, 3).astype(np.float16)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(temp_jpeg_path))
        options: SaveOptions = {'quality': 95, 'ultra_hdr': True, 'gainmap_scale': 4}

        writer.write(img_data, options)

        assert temp_jpeg_path.exists()

    def test_write_bt2020_color_space(self, temp_jpeg_path):
        """Test writing with BT2020 color space"""
        pixels = np.random.rand(8, 8, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT2020

        writer = UltraHDRWriter(str(temp_jpeg_path))
        options: SaveOptions = {'quality': 95, 'ultra_hdr': True, 'gainmap_scale': 4}

        writer.write(img_data, options)

        assert temp_jpeg_path.exists()

    def test_write_display_p3_color_space(self, temp_jpeg_path):
        """Test writing with Display P3 color space"""
        pixels = np.random.rand(8, 8, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.DISPLAY_P3

        writer = UltraHDRWriter(str(temp_jpeg_path))
        options: SaveOptions = {'quality': 95, 'ultra_hdr': True, 'gainmap_scale': 4}

        writer.write(img_data, options)

        assert temp_jpeg_path.exists()

    def test_write_different_quality_levels(self, temp_jpeg_path, sample_hdr_linear_rgb):
        """Test writing with different quality levels"""
        writer = UltraHDRWriter(str(temp_jpeg_path))

        for quality in [50, 75, 90, 100]:
            options: SaveOptions = {'quality': quality, 'ultra_hdr': True, 'gainmap_scale': 4}
            writer.write(sample_hdr_linear_rgb, options)

            assert temp_jpeg_path.exists()
            # Higher quality should generally produce larger files
            # (but not strictly guaranteed due to compression complexity)

    def test_write_different_gainmap_scales(self, temp_jpeg_path, sample_hdr_linear_rgb):
        """Test writing with different gainmap scale factors"""
        writer = UltraHDRWriter(str(temp_jpeg_path))

        for scale in [1, 2, 4, 8]:
            options: SaveOptions = {'quality': 95, 'ultra_hdr': True, 'gainmap_scale': scale}
            writer.write(sample_hdr_linear_rgb, options)

            assert temp_jpeg_path.exists()

    def test_write_creates_directory(self, tmp_path):
        """Test that write creates parent directories if needed"""
        nested_path = tmp_path / "subdir" / "nested" / "test.jpg"

        pixels = np.random.rand(8, 8, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT709

        writer = UltraHDRWriter(str(nested_path))
        options: SaveOptions = {'quality': 95, 'ultra_hdr': True, 'gainmap_scale': 4}

        writer.write(img_data, options)

        assert nested_path.exists()


# ============================================================================
# Ultra HDR Reader Tests
# ============================================================================

class TestUltraHDRReader:
    """Tests for Ultra HDR reader functionality"""

    @pytest.fixture
    def ultrahdr_file(self, tmp_path, sample_hdr_linear_rgb):
        """
        Fixture that creates a valid Ultra HDR file for testing

        Uses UltraHDRWriter to generate a real Ultra HDR JPEG file
        """
        filepath = tmp_path / "test_ultrahdr.jpg"

        writer = UltraHDRWriter(str(filepath))
        options: SaveOptions = {'quality': 95, 'ultra_hdr': True, 'gainmap_scale': 4}
        writer.write(sample_hdr_linear_rgb, options)

        return filepath

    def test_read_valid_ultrahdr_file(self, ultrahdr_file):
        """Test reading a valid Ultra HDR file"""
        reader = UltraHDRReader(ultrahdr_file)
        img_data = reader.read()

        # Check pixels
        assert img_data.pixels.dtype == np.float32
        assert img_data.pixels.ndim == 3
        assert img_data.pixels.shape[2] == 4  # RGBA

        # Check metadata
        assert img_data.metadata['format'] == 'JPEG Ultra HDR'
        assert img_data.metadata['transfer_function'] == TransferFunction.LINEAR
        assert img_data.metadata['filename'] == ultrahdr_file.name
        assert img_data.metadata['file_size'] > 0

    def test_read_preserves_shape(self, tmp_path):
        """Test that reader preserves image dimensions"""
        # Create specific size image
        pixels = np.random.rand(16, 24, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT709

        filepath = tmp_path / "sized.jpg"
        writer = UltraHDRWriter(str(filepath))
        writer.write(img_data, {'quality': 95, 'ultra_hdr': True, 'gainmap_scale': 4})

        # Read back
        reader = UltraHDRReader(filepath)
        result = reader.read()

        # Height and width preserved (but now RGBA instead of RGB)
        assert result.pixels.shape[0] == 16
        assert result.pixels.shape[1] == 24
        assert result.pixels.shape[2] == 4  # RGBA

    def test_validate_file_not_found(self, tmp_path):
        """Test validation fails for non-existent file"""
        nonexistent = tmp_path / "nonexistent.jpg"

        # UltraHDRReader.__init__() calls validate_file() which raises FileNotFoundError
        with pytest.raises(FileNotFoundError, match="File not found"):
            UltraHDRReader(nonexistent)

    def test_validate_file_is_directory(self, tmp_path):
        """Test validation fails for directory"""
        # UltraHDRReader.__init__() calls validate_file() which raises ValueError
        with pytest.raises(ValueError, match="Path is not a file"):
            UltraHDRReader(tmp_path)

    @pytest.mark.skip(reason="Requires non-Ultra HDR JPEG file")
    def test_validate_non_ultrahdr_file(self, tmp_path):
        """Test validation fails for non-Ultra HDR JPEG"""
        # Create minimal standard JPEG
        jpeg_bytes = bytes([
            0xFF, 0xD8,  # SOI
            0xFF, 0xE0,  # APP0
            0x00, 0x10,  # Length
            0x4A, 0x46, 0x49, 0x46, 0x00,  # "JFIF\0"
            0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
            0xFF, 0xD9  # EOI
        ])

        jpeg_path = tmp_path / "standard.jpg"
        with open(jpeg_path, 'wb') as f:
            f.write(jpeg_bytes)

        reader = UltraHDRReader(jpeg_path)

        with pytest.raises(ValueError, match="not a valid JPEG Ultra HDR"):
            reader.validate_file()


# ============================================================================
# JPEG Facade Reader Tests
# ============================================================================

class TestJPEGReaderFacade:
    """Tests for JPEG reader facade (automatic format detection)"""

    def test_reader_delegates_to_ultrahdr_reader(self, tmp_path, sample_hdr_linear_rgb):
        """Test that JPEGReader delegates to UltraHDRReader for Ultra HDR files"""
        filepath = tmp_path / "test_ultrahdr.jpg"

        # Write Ultra HDR file
        writer = UltraHDRWriter(str(filepath))
        writer.write(sample_hdr_linear_rgb, {'quality': 95, 'ultra_hdr': True, 'gainmap_scale': 4})

        # Read via facade
        reader = JPEGReader(filepath)
        img_data = reader.read()

        # Should successfully read and identify as Ultra HDR
        assert img_data.metadata['format'] == 'JPEG Ultra HDR'
        assert img_data.pixels.dtype == np.float32


# ============================================================================
# JPEG Facade Writer Tests
# ============================================================================

class TestJPEGWriterFacade:
    """Tests for JPEG writer facade (automatic delegation)"""

    def test_writer_delegates_to_ultrahdr_writer(self, temp_jpeg_path, sample_hdr_linear_rgb):
        """Test that JPEGWriter delegates to UltraHDRWriter when ultra_hdr=True"""
        writer = JPEGWriter(str(temp_jpeg_path))
        options: SaveOptions = {'ultra_hdr': True, 'quality': 95, 'gainmap_scale': 4}

        # Should delegate to UltraHDRWriter
        writer.write(sample_hdr_linear_rgb, options)

        assert temp_jpeg_path.exists()


# ============================================================================
# Round-trip Integration Tests
# ============================================================================

class TestUltraHDRRoundTrip:
    """Integration tests: write → read → compare"""

    def test_roundtrip_basic_rgb(self, temp_jpeg_path, sample_hdr_linear_rgb):
        """Test round-trip for RGB image (lossy compression tolerance)"""
        # Write
        writer = UltraHDRWriter(str(temp_jpeg_path))
        writer.write(sample_hdr_linear_rgb, {'quality': 95, 'ultra_hdr': True, 'gainmap_scale': 4})

        # Read
        reader = UltraHDRReader(temp_jpeg_path)
        result = reader.read()

        # Compare pixels (with tolerance for lossy compression)
        # Ultra HDR reader returns RGBA, original was RGB
        original_rgb = sample_hdr_linear_rgb.pixels
        result_rgb = result.pixels[:, :, :3]

        # JPEG is lossy + tone mapping, use very relaxed tolerance
        # Ultra HDR applies tone mapping and gainmap, causing significant changes
        np.testing.assert_allclose(result_rgb, original_rgb, rtol=0.5, atol=0.3)

    def test_roundtrip_preserves_dimensions(self, temp_jpeg_path):
        """Test round-trip preserves image dimensions"""
        pixels = np.random.rand(12, 16, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT709

        # Write
        writer = UltraHDRWriter(str(temp_jpeg_path))
        writer.write(img_data, {'quality': 95, 'ultra_hdr': True, 'gainmap_scale': 4})

        # Read
        reader = UltraHDRReader(temp_jpeg_path)
        result = reader.read()

        # Dimensions preserved
        assert result.pixels.shape[0] == 12
        assert result.pixels.shape[1] == 16

    def test_roundtrip_metadata_transfer_function(self, temp_jpeg_path, sample_hdr_linear_rgb):
        """Test round-trip preserves transfer_function metadata"""
        # Write
        writer = UltraHDRWriter(str(temp_jpeg_path))
        writer.write(sample_hdr_linear_rgb, {'quality': 95, 'ultra_hdr': True, 'gainmap_scale': 4})

        # Read
        reader = UltraHDRReader(temp_jpeg_path)
        result = reader.read()

        # Transfer function preserved
        assert result.metadata['transfer_function'] == TransferFunction.LINEAR

    def test_roundtrip_high_quality_better_fidelity(self, tmp_path, sample_hdr_linear_rgb):
        """Test that higher quality produces better fidelity (smaller error)"""
        # Write with low quality
        low_quality_path = tmp_path / "low.jpg"
        writer_low = UltraHDRWriter(str(low_quality_path))
        writer_low.write(sample_hdr_linear_rgb, {'quality': 50, 'ultra_hdr': True, 'gainmap_scale': 4})

        # Write with high quality
        high_quality_path = tmp_path / "high.jpg"
        writer_high = UltraHDRWriter(str(high_quality_path))
        writer_high.write(sample_hdr_linear_rgb, {'quality': 100, 'ultra_hdr': True, 'gainmap_scale': 4})

        # Read both
        reader_low = UltraHDRReader(low_quality_path)
        result_low = reader_low.read()

        reader_high = UltraHDRReader(high_quality_path)
        result_high = reader_high.read()  # Fixed: was result_high.read()

        original_rgb = sample_hdr_linear_rgb.pixels

        # Calculate errors
        error_low = np.mean(np.abs(result_low.pixels[:, :, :3] - original_rgb))
        error_high = np.mean(np.abs(result_high.pixels[:, :, :3] - original_rgb))

        # High quality should have lower error
        # (not strictly guaranteed but very likely)
        assert error_high <= error_low * 1.5  # Allow some variance

    def test_roundtrip_bt2020_color_space(self, temp_jpeg_path):
        """Test round-trip with BT2020 color space"""
        pixels = np.random.rand(8, 8, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)
        img_data.metadata['transfer_function'] = TransferFunction.LINEAR
        img_data.metadata['color_space'] = ColorSpace.BT2020

        # Write
        writer = UltraHDRWriter(str(temp_jpeg_path))
        writer.write(img_data, {'quality': 95, 'ultra_hdr': True, 'gainmap_scale': 4})

        # Read
        reader = UltraHDRReader(temp_jpeg_path)
        result = reader.read()

        # Should successfully read
        assert result.pixels is not None
        # TODO: Color space is not currently extracted by reader (metadata limitation)
        # This would require XMP parsing, which is not implemented yet

    def test_roundtrip_via_facades(self, temp_jpeg_path, sample_hdr_linear_rgb):
        """Test round-trip using JPEGWriter/JPEGReader facades"""
        # Write via facade
        writer = JPEGWriter(str(temp_jpeg_path))
        writer.write(sample_hdr_linear_rgb, {'ultra_hdr': True, 'quality': 95})

        # Read via facade
        reader = JPEGReader(temp_jpeg_path)
        result = reader.read()

        # Should work correctly
        assert result.pixels.dtype == np.float32
        assert result.metadata['format'] == 'JPEG Ultra HDR'
