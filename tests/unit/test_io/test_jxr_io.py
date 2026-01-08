"""
Tests for JPEG XR (JXR) I/O (round-trip: write → read)

Tests cover:
- JXR SaveOptions adapter validation (lossless, quality, photometric, resolution)
- Writer validation (uint8/uint16/float16/float32, 1/3/4 channels)
- Writer encoding (lossless/lossy, different quality levels, float support)
- Reader functionality (basic reading, metadata extraction, fp2int parameter)
- Round-trip tests (write → read, lossless for uint, minimal loss for float)

Strategy: Write ImageData to temp file, read back, verify pixels and metadata

Note: JPEG XR supports HDR (float16/float32) with minimal precision loss (~6e-5).
      uint8/uint16 can achieve true lossless compression at level=100.
"""

import numpy as np
import pytest
from pathlib import Path

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.jxr.reader import JXRFormatReader
from image_pipeline.io.formats.jxr.writer import JXRFormatWriter
from image_pipeline.io.formats.jxr.options import JXRSaveOptionsAdapter, JXRSaveOptions


@pytest.fixture
def temp_jxr_path(tmp_path):
    """Fixture providing temporary JXR file path"""
    return tmp_path / "test.jxr"


@pytest.fixture
def sample_uint8_rgb():
    """Sample 8-bit RGB image"""
    pixels = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
        [[255, 0, 255], [0, 255, 255], [128, 128, 128], [64, 64, 64]],
        [[192, 64, 32], [32, 192, 64], [64, 32, 192], [100, 150, 200]],
        [[50, 100, 150], [150, 50, 100], [100, 150, 50], [200, 100, 50]]
    ], dtype=np.uint8)
    return ImageData(pixels=pixels)


@pytest.fixture
def sample_uint16_rgb():
    """Sample 16-bit RGB image"""
    pixels = np.array([
        [[65535, 0, 0], [0, 65535, 0], [0, 0, 65535], [65535, 65535, 0]],
        [[65535, 0, 65535], [0, 65535, 65535], [32768, 32768, 32768], [16384, 16384, 16384]],
        [[49152, 16384, 8192], [8192, 49152, 16384], [16384, 8192, 49152], [25600, 38400, 51200]]
    ], dtype=np.uint16)
    return ImageData(pixels=pixels)


@pytest.fixture
def sample_float32_rgb():
    """Sample float32 RGB image (HDR, values 0-2.0)"""
    pixels = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [2.0, 1.5, 0.5]],
        [[1.5, 0.5, 2.0], [0.5, 2.0, 1.5], [0.8, 0.8, 0.8], [1.2, 1.2, 1.2]],
        [[0.3, 0.6, 0.9], [1.8, 1.5, 1.2], [0.1, 0.2, 0.3], [1.9, 1.7, 1.6]]
    ], dtype=np.float32)
    return ImageData(pixels=pixels)


@pytest.fixture
def sample_float16_rgb():
    """Sample float16 RGB image (HDR)"""
    pixels = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.5, 0.5, 1.2], [0.8, 1.8, 0.3], [1.1, 0.9, 1.3]]
    ], dtype=np.float16)
    return ImageData(pixels=pixels)


@pytest.fixture
def sample_uint8_rgba():
    """Sample 8-bit RGBA image"""
    pixels = np.array([
        [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 128], [255, 255, 0, 64]],
        [[255, 0, 255, 192], [0, 255, 255, 255], [128, 128, 128, 128], [64, 64, 64, 255]]
    ], dtype=np.uint8)
    return ImageData(pixels=pixels)


# ============================================================================
# JXR Save Options Adapter Tests
# ============================================================================

class TestJXROptionsAdapter:
    """Tests for JXR save options validation and adaptation"""

    def test_default_options(self):
        """Test that defaults are applied correctly"""
        adapter = JXRSaveOptionsAdapter()
        validated = adapter.validate({})

        assert validated['lossless'] is True  # JXR defaults to lossless
        assert validated['quality'] == 90
        assert 'photometric' not in validated  # No default
        assert 'resolution' not in validated  # No default

    def test_custom_quality(self):
        """Test custom quality value"""
        adapter = JXRSaveOptionsAdapter()
        validated = adapter.validate({'quality': 95})
        assert validated['quality'] == 95

    def test_quality_validation_range_low(self):
        """Test quality validation: must be >= 1"""
        adapter = JXRSaveOptionsAdapter()
        with pytest.raises(ValueError, match="quality must be in range"):
            adapter.validate({'quality': 0})

    def test_quality_validation_range_high(self):
        """Test quality validation: must be <= 100"""
        adapter = JXRSaveOptionsAdapter()
        with pytest.raises(ValueError, match="quality must be in range"):
            adapter.validate({'quality': 101})

    def test_quality_validation_type(self):
        """Test quality validation: must be int"""
        adapter = JXRSaveOptionsAdapter()
        with pytest.raises(TypeError, match="quality must be int"):
            adapter.validate({'quality': 90.5})

    def test_lossless_true(self):
        """Test lossless flag can be set to True"""
        adapter = JXRSaveOptionsAdapter()
        validated = adapter.validate({'lossless': True})
        assert validated['lossless'] is True

    def test_lossless_false(self):
        """Test lossless flag can be set to False"""
        adapter = JXRSaveOptionsAdapter()
        validated = adapter.validate({'lossless': False})
        assert validated['lossless'] is False

    def test_lossless_validation_type(self):
        """Test lossless validation: must be bool"""
        adapter = JXRSaveOptionsAdapter()
        with pytest.raises(TypeError, match="lossless must be bool"):
            adapter.validate({'lossless': 1})

    def test_photometric_custom(self):
        """Test custom photometric value"""
        adapter = JXRSaveOptionsAdapter()
        validated = adapter.validate({'photometric': 2})
        assert validated['photometric'] == 2

    def test_photometric_none(self):
        """Test photometric can be None (auto-detect)"""
        adapter = JXRSaveOptionsAdapter()
        validated = adapter.validate({'photometric': None})
        assert validated['photometric'] is None

    def test_photometric_validation_type(self):
        """Test photometric validation: must be int or None"""
        adapter = JXRSaveOptionsAdapter()
        with pytest.raises(TypeError, match="photometric must be int or None"):
            adapter.validate({'photometric': '2'})

    def test_resolution_custom(self):
        """Test custom resolution value"""
        adapter = JXRSaveOptionsAdapter()
        validated = adapter.validate({'resolution': (300.0, 300.0)})
        assert validated['resolution'] == (300.0, 300.0)

    def test_resolution_list(self):
        """Test resolution accepts list"""
        adapter = JXRSaveOptionsAdapter()
        validated = adapter.validate({'resolution': [72.0, 72.0]})
        assert validated['resolution'] == (72.0, 72.0)

    def test_resolution_none(self):
        """Test resolution can be None (no metadata)"""
        adapter = JXRSaveOptionsAdapter()
        validated = adapter.validate({'resolution': None})
        # None resolution should not be added to validated dict
        assert 'resolution' not in validated

    def test_resolution_validation_length(self):
        """Test resolution validation: must have 2 values"""
        adapter = JXRSaveOptionsAdapter()
        with pytest.raises(ValueError, match="resolution must have 2 values"):
            adapter.validate({'resolution': (300.0,)})

    def test_resolution_validation_positive(self):
        """Test resolution validation: values must be positive"""
        adapter = JXRSaveOptionsAdapter()
        with pytest.raises(ValueError, match="resolution DPI values must be positive"):
            adapter.validate({'resolution': (300.0, -72.0)})

    def test_resolution_validation_type(self):
        """Test resolution validation: must be tuple/list or None"""
        adapter = JXRSaveOptionsAdapter()
        with pytest.raises(TypeError, match="resolution must be tuple/list or None"):
            adapter.validate({'resolution': 300})

    def test_all_options_together(self):
        """Test all options together"""
        adapter = JXRSaveOptionsAdapter()
        options = {
            'lossless': False,
            'quality': 85,
            'photometric': 2,
            'resolution': (300.0, 300.0)
        }
        validated = adapter.validate(options)

        assert validated['lossless'] is False
        assert validated['quality'] == 85
        assert validated['photometric'] == 2
        assert validated['resolution'] == (300.0, 300.0)


# ============================================================================
# JXR Writer - Validation Tests
# ============================================================================

class TestJXRWriterValidation:
    """Tests for JXR writer input validation"""

    def test_validate_accepts_uint8(self, temp_jxr_path, sample_uint8_rgb):
        """Test validation accepts uint8"""
        writer = JXRFormatWriter(str(temp_jxr_path))
        writer.validate(sample_uint8_rgb)  # Should not raise

    def test_validate_accepts_uint16(self, temp_jxr_path, sample_uint16_rgb):
        """Test validation accepts uint16"""
        writer = JXRFormatWriter(str(temp_jxr_path))
        writer.validate(sample_uint16_rgb)  # Should not raise

    def test_validate_accepts_float32(self, temp_jxr_path, sample_float32_rgb):
        """Test validation accepts float32 (HDR)"""
        writer = JXRFormatWriter(str(temp_jxr_path))
        writer.validate(sample_float32_rgb)  # Should not raise

    def test_validate_accepts_float16(self, temp_jxr_path, sample_float16_rgb):
        """Test validation accepts float16 (HDR)"""
        writer = JXRFormatWriter(str(temp_jxr_path))
        writer.validate(sample_float16_rgb)  # Should not raise

    def test_validate_rejects_int32(self, temp_jxr_path):
        """Test validation rejects int32"""
        pixels = np.random.randint(0, 1000, (8, 8, 3), dtype=np.int32)
        img_data = ImageData(pixels=pixels)

        writer = JXRFormatWriter(str(temp_jxr_path))

        with pytest.raises(ValueError, match="JPEG XR format supports uint8, uint16, float16, float32"):
            writer.validate(img_data)

    def test_validate_rejects_float64(self, temp_jxr_path):
        """Test validation rejects float64"""
        pixels = np.random.rand(8, 8, 3).astype(np.float64)
        img_data = ImageData(pixels=pixels)

        writer = JXRFormatWriter(str(temp_jxr_path))

        with pytest.raises(ValueError, match="JPEG XR format supports uint8, uint16, float16, float32"):
            writer.validate(img_data)

    def test_validate_invalid_dimensions_1d(self, temp_jxr_path):
        """Test validation rejects 1D arrays"""
        pixels = np.random.randint(0, 256, 100, dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = JXRFormatWriter(str(temp_jxr_path))

        with pytest.raises(ValueError, match="JPEG XR requires 2D or 3D array"):
            writer.validate(img_data)

    def test_validate_invalid_channels_2(self, temp_jxr_path):
        """Test validation rejects 2 channels"""
        pixels = np.random.randint(0, 256, (8, 8, 2), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = JXRFormatWriter(str(temp_jxr_path))

        with pytest.raises(ValueError, match="JPEG XR supports 1 \\(grayscale\\), 3 \\(RGB\\), or 4 \\(RGBA\\) channels"):
            writer.validate(img_data)

    def test_validate_invalid_channels_5(self, temp_jxr_path):
        """Test validation rejects 5 channels"""
        pixels = np.random.randint(0, 256, (8, 8, 5), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = JXRFormatWriter(str(temp_jxr_path))

        with pytest.raises(ValueError, match="JPEG XR supports 1 \\(grayscale\\), 3 \\(RGB\\), or 4 \\(RGBA\\) channels"):
            writer.validate(img_data)

    def test_validate_empty_array(self, temp_jxr_path):
        """Test validation rejects empty arrays"""
        pixels = np.array([], dtype=np.uint8)
        img_data = ImageData(pixels=pixels.reshape(0, 0, 3))

        writer = JXRFormatWriter(str(temp_jxr_path))

        with pytest.raises(ValueError, match="Empty pixel array"):
            writer.validate(img_data)

    def test_validate_accepts_rgba(self, temp_jxr_path, sample_uint8_rgba):
        """Test validation accepts RGBA"""
        writer = JXRFormatWriter(str(temp_jxr_path))
        writer.validate(sample_uint8_rgba)  # Should not raise

    def test_validate_accepts_grayscale_2d(self, temp_jxr_path):
        """Test validation accepts grayscale (2D)"""
        pixels = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = JXRFormatWriter(str(temp_jxr_path))
        writer.validate(img_data)  # Should not raise


# ============================================================================
# JXR Writer - Encoding Tests
# ============================================================================

class TestJXRWriterEncoding:
    """Tests for JXR encoding functionality"""

    def test_write_basic_uint8_lossless(self, temp_jxr_path, sample_uint8_rgb):
        """Test writing uint8 RGB with lossless compression"""
        writer = JXRFormatWriter(str(temp_jxr_path))
        options: JXRSaveOptions = {'lossless': True}

        writer.write(sample_uint8_rgb, options)

        assert temp_jxr_path.exists()
        assert temp_jxr_path.stat().st_size > 0

    def test_write_basic_uint8_lossy(self, temp_jxr_path, sample_uint8_rgb):
        """Test writing uint8 RGB with lossy compression"""
        writer = JXRFormatWriter(str(temp_jxr_path))
        options: JXRSaveOptions = {'lossless': False, 'quality': 80}

        writer.write(sample_uint8_rgb, options)

        assert temp_jxr_path.exists()
        assert temp_jxr_path.stat().st_size > 0

    def test_write_uint16_lossless(self, temp_jxr_path, sample_uint16_rgb):
        """Test writing uint16 RGB with lossless compression"""
        writer = JXRFormatWriter(str(temp_jxr_path))
        options: JXRSaveOptions = {'lossless': True}

        writer.write(sample_uint16_rgb, options)

        assert temp_jxr_path.exists()

    def test_write_float32_hdr(self, temp_jxr_path, sample_float32_rgb):
        """Test writing float32 HDR image"""
        writer = JXRFormatWriter(str(temp_jxr_path))
        options: JXRSaveOptions = {'lossless': True}

        writer.write(sample_float32_rgb, options)

        assert temp_jxr_path.exists()

    def test_write_float16_hdr(self, temp_jxr_path, sample_float16_rgb):
        """Test writing float16 HDR image"""
        writer = JXRFormatWriter(str(temp_jxr_path))
        options: JXRSaveOptions = {'lossless': True}

        writer.write(sample_float16_rgb, options)

        assert temp_jxr_path.exists()

    def test_write_rgba(self, temp_jxr_path, sample_uint8_rgba):
        """Test writing RGBA image"""
        writer = JXRFormatWriter(str(temp_jxr_path))
        options: JXRSaveOptions = {'lossless': True}

        writer.write(sample_uint8_rgba, options)

        assert temp_jxr_path.exists()

    def test_write_different_quality_levels(self, temp_jxr_path):
        """Test writing with different quality levels (lossy)"""
        # Note: Use larger image to avoid jxrlib errors on small images
        # "Image width must be at least 2 MB wide for subsampled chroma..."
        pixels = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = JXRFormatWriter(str(temp_jxr_path))

        for quality in [50, 80, 100]:  # Skip very low quality (10) - causes issues on small images
            options: JXRSaveOptions = {'lossless': False, 'quality': quality}
            writer.write(img_data, options)

            assert temp_jxr_path.exists()

    def test_write_with_resolution(self, temp_jxr_path, sample_uint8_rgb):
        """Test writing with resolution metadata"""
        writer = JXRFormatWriter(str(temp_jxr_path))
        options: JXRSaveOptions = {'lossless': True, 'resolution': (300.0, 300.0)}

        writer.write(sample_uint8_rgb, options)

        assert temp_jxr_path.exists()

    def test_lossless_larger_than_lossy(self, tmp_path, sample_uint8_rgb):
        """Test that lossless produces larger files than lossy (generally)"""
        lossy_path = tmp_path / "lossy.jxr"
        lossless_path = tmp_path / "lossless.jxr"

        # Write lossy
        writer_lossy = JXRFormatWriter(str(lossy_path))
        writer_lossy.write(sample_uint8_rgb, {'lossless': False, 'quality': 50})

        # Write lossless
        writer_lossless = JXRFormatWriter(str(lossless_path))
        writer_lossless.write(sample_uint8_rgb, {'lossless': True})

        lossy_size = lossy_path.stat().st_size
        lossless_size = lossless_path.stat().st_size

        # Lossless should be significantly larger for lossy quality=50
        assert lossless_size > lossy_size


# ============================================================================
# JXR Reader Tests
# ============================================================================

class TestJXRReader:
    """Tests for JXR reader functionality"""

    @pytest.fixture
    def jxr_uint8_file(self, tmp_path, sample_uint8_rgb):
        """Fixture that creates a valid JXR file (uint8) for testing"""
        filepath = tmp_path / "test_uint8.jxr"

        writer = JXRFormatWriter(str(filepath))
        options: JXRSaveOptions = {'lossless': True}
        writer.write(sample_uint8_rgb, options)

        return filepath

    @pytest.fixture
    def jxr_float32_file(self, tmp_path, sample_float32_rgb):
        """Fixture that creates a valid JXR file (float32) for testing"""
        filepath = tmp_path / "test_float32.jxr"

        writer = JXRFormatWriter(str(filepath))
        options: JXRSaveOptions = {'lossless': True}
        writer.write(sample_float32_rgb, options)

        return filepath

    def test_read_valid_uint8_file(self, jxr_uint8_file):
        """Test reading a valid uint8 JXR file"""
        reader = JXRFormatReader(jxr_uint8_file)
        img_data = reader.read()

        # Check pixels
        assert img_data.pixels.dtype == np.uint8
        assert img_data.pixels.ndim == 3

        # Check metadata
        assert img_data.metadata['format'] == 'JXR'
        assert img_data.metadata['filename'] == jxr_uint8_file.name
        assert img_data.metadata['file_size'] > 0

    def test_read_valid_float32_file(self, jxr_float32_file):
        """Test reading a valid float32 JXR file"""
        reader = JXRFormatReader(jxr_float32_file)
        img_data = reader.read()

        # Check pixels
        assert img_data.pixels.dtype == np.float32
        assert img_data.pixels.ndim == 3

        # Check metadata
        assert img_data.metadata['format'] == 'JXR'
        assert img_data.metadata['filename'] == jxr_float32_file.name
        assert img_data.metadata['file_size'] > 0

    def test_read_preserves_shape(self, tmp_path):
        """Test that reader preserves image dimensions"""
        # Create specific size image
        pixels = np.random.randint(0, 256, (16, 24, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        filepath = tmp_path / "sized.jxr"
        writer = JXRFormatWriter(str(filepath))
        writer.write(img_data, {'lossless': True})

        # Read back
        reader = JXRFormatReader(filepath)
        result = reader.read()

        # Dimensions preserved
        assert result.pixels.shape[0] == 16
        assert result.pixels.shape[1] == 24
        assert result.pixels.shape[2] == 3

    def test_read_invalid_file(self, tmp_path):
        """Test reading an invalid file raises IOError"""
        invalid_path = tmp_path / "invalid.jxr"

        # Create a file with invalid JXR data
        with open(invalid_path, 'wb') as f:
            f.write(b'NOT A JXR FILE')

        reader = JXRFormatReader(invalid_path)

        with pytest.raises(IOError, match="Error reading JPEG XR"):
            reader.read()

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading non-existent file raises FileNotFoundError"""
        nonexistent = tmp_path / "nonexistent.jxr"

        with pytest.raises(FileNotFoundError):
            JXRFormatReader(nonexistent)


# ============================================================================
# JXR File Extension Tests
# ============================================================================

class TestJXRFileExtensions:
    """Test that all JXR extensions (.jxr, .wdp, .hdp) work correctly"""

    @pytest.mark.parametrize("extension", [".jxr", ".wdp", ".hdp"])
    def test_write_read_all_extensions(self, tmp_path, sample_uint8_rgb, extension):
        """Test write and read work for all JXR extensions"""
        filepath = tmp_path / f"test{extension}"

        # Write
        writer = JXRFormatWriter(str(filepath))
        writer.write(sample_uint8_rgb, {'lossless': True})

        assert filepath.exists()

        # Read
        reader = JXRFormatReader(filepath)
        result = reader.read()

        assert result.pixels.dtype == np.uint8
        assert result.metadata['format'] == 'JXR'


# ============================================================================
# Round-trip Integration Tests
# ============================================================================

class TestJXRRoundTrip:
    """Integration tests: write → read → compare"""

    def test_roundtrip_lossless_uint8_rgb(self, temp_jxr_path, sample_uint8_rgb):
        """Test lossless round-trip for uint8 RGB (exact match)"""
        # Write lossless
        writer = JXRFormatWriter(str(temp_jxr_path))
        writer.write(sample_uint8_rgb, {'lossless': True})

        # Read
        reader = JXRFormatReader(temp_jxr_path)
        result = reader.read()

        # Lossless uint8 should be exact
        np.testing.assert_array_equal(result.pixels, sample_uint8_rgb.pixels)

    def test_roundtrip_lossless_uint16_rgb(self, temp_jxr_path, sample_uint16_rgb):
        """Test lossless round-trip for uint16 RGB (exact match)"""
        # Write lossless
        writer = JXRFormatWriter(str(temp_jxr_path))
        writer.write(sample_uint16_rgb, {'lossless': True})

        # Read
        reader = JXRFormatReader(temp_jxr_path)
        result = reader.read()

        # Lossless uint16 should be exact
        np.testing.assert_array_equal(result.pixels, sample_uint16_rgb.pixels)

    def test_roundtrip_lossless_rgba(self, temp_jxr_path, sample_uint8_rgba):
        """Test lossless round-trip for RGBA image"""
        # Write lossless
        writer = JXRFormatWriter(str(temp_jxr_path))
        writer.write(sample_uint8_rgba, {'lossless': True})

        # Read
        reader = JXRFormatReader(temp_jxr_path)
        result = reader.read()

        # Lossless RGBA should be exact
        np.testing.assert_array_equal(result.pixels, sample_uint8_rgba.pixels)

    def test_roundtrip_float32_minimal_loss(self, temp_jxr_path, sample_float32_rgb):
        """Test round-trip for float32 (minimal precision loss)"""
        # Write lossless (note: float always has minimal loss ~6e-5)
        writer = JXRFormatWriter(str(temp_jxr_path))
        writer.write(sample_float32_rgb, {'lossless': True})

        # Read
        reader = JXRFormatReader(temp_jxr_path)
        result = reader.read()

        # float32 should have minimal loss
        max_diff = np.max(np.abs(result.pixels - sample_float32_rgb.pixels))
        assert max_diff < 1e-4  # Allow small precision loss

    def test_roundtrip_float16_minimal_loss(self, temp_jxr_path, sample_float16_rgb):
        """Test round-trip for float16 (minimal precision loss)"""
        # Write lossless
        writer = JXRFormatWriter(str(temp_jxr_path))
        writer.write(sample_float16_rgb, {'lossless': True})

        # Read
        reader = JXRFormatReader(temp_jxr_path)
        result = reader.read()

        # float16 should have minimal loss
        max_diff = np.max(np.abs(result.pixels - sample_float16_rgb.pixels))
        assert max_diff < 1e-3  # Allow slightly higher tolerance for float16

    def test_roundtrip_lossy_uint8(self, temp_jxr_path, sample_uint8_rgb):
        """Test lossy round-trip for uint8 RGB"""
        # Write lossy
        writer = JXRFormatWriter(str(temp_jxr_path))
        writer.write(sample_uint8_rgb, {'lossless': False, 'quality': 80})

        # Read
        reader = JXRFormatReader(temp_jxr_path)
        result = reader.read()

        # Lossy should have some difference, but same shape and dtype
        assert result.pixels.shape == sample_uint8_rgb.pixels.shape
        assert result.pixels.dtype == np.uint8

    def test_roundtrip_preserves_dimensions(self, temp_jxr_path):
        """Test round-trip preserves image dimensions"""
        pixels = np.random.randint(0, 256, (12, 16, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        # Write
        writer = JXRFormatWriter(str(temp_jxr_path))
        writer.write(img_data, {'lossless': True})

        # Read
        reader = JXRFormatReader(temp_jxr_path)
        result = reader.read()

        # Dimensions preserved
        assert result.pixels.shape == (12, 16, 3)

    def test_roundtrip_metadata_format(self, temp_jxr_path, sample_uint8_rgb):
        """Test round-trip preserves format metadata"""
        # Write
        writer = JXRFormatWriter(str(temp_jxr_path))
        writer.write(sample_uint8_rgb, {'lossless': True})

        # Read
        reader = JXRFormatReader(temp_jxr_path)
        result = reader.read()

        # Format preserved
        assert result.metadata['format'] == 'JXR'

    def test_roundtrip_high_quality_better_fidelity(self, tmp_path):
        """Test that higher quality produces better fidelity"""
        # Use larger image to avoid jxrlib errors on small images with low quality
        pixels = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        # Write with medium-low quality
        low_quality_path = tmp_path / "low.jxr"
        writer_low = JXRFormatWriter(str(low_quality_path))
        writer_low.write(img_data, {'lossless': False, 'quality': 40})

        # Write with high quality
        high_quality_path = tmp_path / "high.jxr"
        writer_high = JXRFormatWriter(str(high_quality_path))
        writer_high.write(img_data, {'lossless': False, 'quality': 100})

        # Read both
        reader_low = JXRFormatReader(low_quality_path)
        result_low = reader_low.read()

        reader_high = JXRFormatReader(high_quality_path)
        result_high = reader_high.read()

        original = pixels

        # Calculate errors
        error_low = np.mean(np.abs(result_low.pixels.astype(np.int16) - original.astype(np.int16)))
        error_high = np.mean(np.abs(result_high.pixels.astype(np.int16) - original.astype(np.int16)))

        # High quality should have lower or equal error
        assert error_high <= error_low

    def test_roundtrip_large_random_image(self, temp_jxr_path):
        """Test round-trip with larger random image"""
        # Create larger random image
        pixels = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        # Write lossless
        writer = JXRFormatWriter(str(temp_jxr_path))
        writer.write(img_data, {'lossless': True})

        # Read
        reader = JXRFormatReader(temp_jxr_path)
        result = reader.read()

        # Should be exact for uint8 lossless
        np.testing.assert_array_equal(result.pixels, pixels)
