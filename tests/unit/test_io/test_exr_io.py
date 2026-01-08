"""Comprehensive tests for OpenEXR I/O with metadata"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import OpenEXR

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.exr import EXRFormatReader, EXRFormatWriter
from image_pipeline.types import ColorSpace, TransferFunction


@pytest.fixture
def temp_exr_path():
    """Create temporary EXR file path"""
    with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as f:
        path = Path(f.name)
    yield path
    if path.exists():
        path.unlink()


class TestEXRBasicIO:
    """Tests for basic EXR I/O without metadata"""

    def test_write_read_float32_rgb(self, temp_exr_path):
        """Test float32 RGB EXR round-trip"""
        pixels = np.random.rand(20, 20, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        # Write
        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={'pixel_type': 'float'}) 

        # Read back
        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()

        # Verify pixels (should be lossless for FLOAT type)
        np.testing.assert_array_almost_equal(result.pixels, pixels, decimal=5)
        assert result.pixels.shape == (20, 20, 3)
        assert result.pixels.dtype == np.float32

    def test_write_read_half_rgb(self, temp_exr_path):
        """Test HALF (float16) RGB EXR round-trip"""
        pixels = np.random.rand(20, 20, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        # Write with HALF pixel type (default)
        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={'pixel_type': 'half'}) 

        # Read back
        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()

        # HALF has less precision, so use lower tolerance
        np.testing.assert_array_almost_equal(result.pixels, pixels, decimal=3)
        assert result.pixels.dtype == np.float32  # Reader converts to float32

    def test_write_read_rgba(self, temp_exr_path):
        """Test RGBA (with alpha channel) round-trip"""
        pixels = np.random.rand(15, 15, 4).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        # Write
        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={'pixel_type': 'float'}) 

        # Read back
        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()

        # Verify all 4 channels
        np.testing.assert_array_almost_equal(result.pixels, pixels, decimal=5)
        assert result.pixels.shape == (15, 15, 4)
        assert result.metadata.get('channels') == 4

    def test_hdr_values_preserved(self, temp_exr_path):
        """Test that HDR values > 1.0 are preserved"""
        # Create HDR data with values up to 10.0
        pixels = np.random.rand(10, 10, 3).astype(np.float32) * 10.0
        img_data = ImageData(pixels=pixels)

        # Write and read
        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={'pixel_type': 'float'}) 

        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()

        # Verify HDR values are preserved
        np.testing.assert_array_almost_equal(result.pixels, pixels, decimal=5)
        assert result.pixels.max() > 1.0  # Ensure HDR range preserved


class TestEXRMetadata:
    """Tests for EXR metadata handling"""

    def test_chromaticities_bt709(self, temp_exr_path):
        """Test BT.709 chromaticities write and read"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        metadata = {'color_space': ColorSpace.BT709} 
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Write
        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={})

        # Read raw header to verify
        exr_file = OpenEXR.InputFile(str(temp_exr_path))
        header = exr_file.header()
        assert 'chromaticities' in header

        chroma = header['chromaticities']
        # BT.709 red primary
        assert abs(chroma.red.x - 0.64) < 0.01
        assert abs(chroma.red.y - 0.33) < 0.01

        # Read back via reader
        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()
        assert result.metadata.get('color_space') == ColorSpace.BT709

    def test_chromaticities_bt2020(self, temp_exr_path):
        """Test BT.2020 chromaticities write and read"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        metadata = {'color_space': ColorSpace.BT2020} 
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Write
        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={})

        # Read back
        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()
        assert result.metadata.get('color_space') == ColorSpace.BT2020

    def test_chromaticities_display_p3(self, temp_exr_path):
        """Test Display P3 chromaticities write and read"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        metadata = {'color_space': ColorSpace.DISPLAY_P3} 
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Write
        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={})

        # Read back
        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()
        assert result.metadata.get('color_space') == ColorSpace.DISPLAY_P3

    def test_white_luminance(self, temp_exr_path):
        """Test whiteLuminance (paper_white) write and read"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        metadata = {'paper_white': 203.0}
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Write
        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={})

        # Verify in raw header
        exr_file = OpenEXR.InputFile(str(temp_exr_path))
        header = exr_file.header()
        assert 'whiteLuminance' in header
        assert abs(header['whiteLuminance'] - 203.0) < 0.01

        # Read back
        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()
        assert result.metadata.get('paper_white') == pytest.approx(203.0)

    def test_combined_metadata(self, temp_exr_path):
        """Test writing both chromaticities and whiteLuminance"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        metadata = { 
            'color_space': ColorSpace.BT2020,
            'paper_white': 100.0,
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Write
        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={})

        # Read back
        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()

        # Verify both attributes
        assert result.metadata.get('color_space') == ColorSpace.BT2020
        assert result.metadata.get('paper_white') == pytest.approx(100.0)
        assert result.metadata.get('transfer_function') == TransferFunction.LINEAR

    def test_default_metadata_when_missing(self, temp_exr_path):
        """Test default metadata values when not specified"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)  # No metadata

        # Write
        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={})

        # Read back
        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()

        # Should have defaults
        assert result.metadata.get('color_space') == ColorSpace.BT709  # Default
        assert result.metadata.get('transfer_function') == TransferFunction.LINEAR
        assert result.metadata.get('format') == 'OpenEXR'

    def test_custom_primaries(self, temp_exr_path):
        """Test custom color primaries (not matching standard spaces)"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        custom_primaries = {
            'red': (0.7, 0.3),
            'green': (0.2, 0.7),
            'blue': (0.14, 0.05),
            'white': (0.31, 0.32)
        }
        metadata = {'color_primaries': custom_primaries} 
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Write
        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={})

        # Read back
        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()

        # Should store as custom primaries (not matched to standard)
        assert result.metadata.get('color_space') is None
        primaries = result.metadata.get('color_primaries')
        assert primaries is not None
        assert abs(primaries['red'][0] - 0.7) < 0.01


class TestEXRCompression:
    """Tests for different compression methods"""

    def test_compression_zip(self, temp_exr_path):
        """Test ZIP compression"""
        pixels = np.random.rand(20, 20, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={'compression': 'zip'})

        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()
        np.testing.assert_array_almost_equal(result.pixels, pixels, decimal=3)

    def test_compression_piz(self, temp_exr_path):
        """Test PIZ compression"""
        pixels = np.random.rand(20, 20, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={'compression': 'piz'})

        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()
        np.testing.assert_array_almost_equal(result.pixels, pixels, decimal=3)

    def test_compression_none(self, temp_exr_path):
        """Test no compression"""
        pixels = np.random.rand(20, 20, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={'compression': 'none'})

        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()
        # Default pixel_type is 'half' (float16), so use lower precision
        np.testing.assert_array_almost_equal(result.pixels, pixels, decimal=3)


class TestEXREdgeCases:
    """Tests for edge cases and error handling"""

    def test_very_small_image(self, temp_exr_path):
        """Test 1x1 pixel image"""
        pixels = np.array([[[0.5, 0.7, 0.9]]], dtype=np.float32)
        img_data = ImageData(pixels=pixels)

        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={})

        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()
        # Default pixel_type is 'half' (float16), so use lower precision
        np.testing.assert_array_almost_equal(result.pixels, pixels, decimal=3)

    def test_large_image(self, temp_exr_path):
        """Test larger image (100x100)"""
        pixels = np.random.rand(100, 100, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={'pixel_type': 'half'}) 

        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()
        assert result.pixels.shape == (100, 100, 3)

    def test_zero_values(self, temp_exr_path):
        """Test image with all zero values"""
        pixels = np.zeros((10, 10, 3), dtype=np.float32)
        img_data = ImageData(pixels=pixels)

        writer = EXRFormatWriter(str(temp_exr_path))
        writer.write(img_data, options={})

        reader = EXRFormatReader(temp_exr_path)
        result = reader.read()
        np.testing.assert_array_equal(result.pixels, pixels)

    def test_invalid_channel_count(self):
        """Test that invalid channel count raises error"""
        pixels = np.random.rand(10, 10, 2).astype(np.float32)  # 2 channels (invalid)
        img_data = ImageData(pixels=pixels)

        with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as f:
            path = Path(f.name)

        try:
            writer = EXRFormatWriter(str(path))
            with pytest.raises(ValueError, match="3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
                writer.write(img_data, options={})
        finally:
            if path.exists():
                path.unlink()
