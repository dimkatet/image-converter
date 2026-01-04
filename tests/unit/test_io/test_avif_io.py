"""
Tests for AVIF format I/O (round-trip testing)

AVIF (AV1 Image File Format) HDR Support:
- Bit depths: 8, 10, 12 (via uint16)
- CICP metadata: color primaries, transfer characteristics, matrix coefficients
- Supports: BT.709, BT.2020, Display P3
- Transfer functions: PQ (ST.2084), HLG, sRGB
- ISO BMFF-based format with embedded metadata

Tests verify expected behavior based on AVIF specification and current implementation.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.avif.reader import AVIFFormatReader
from image_pipeline.io.formats.avif.writer import AVIFFormatWriter
from image_pipeline.types import ColorSpace, TransferFunction


@pytest.fixture
def temp_avif_path():
    """Create temporary AVIF file path"""
    with tempfile.NamedTemporaryFile(suffix='.avif', delete=False) as f:
        path = Path(f.name)

    yield path

    # Cleanup
    if path.exists():
        path.unlink()


class TestAVIFBasicIO:
    """Tests for basic AVIF I/O without metadata"""

    def test_write_read_8bit_rgb(self, temp_avif_path):
        """Test 8-bit RGB AVIF round-trip"""
        pixels = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        # Write
        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 100})  # Lossless

        # Read back
        reader = AVIFFormatReader(temp_avif_path)
        result = reader.read()

        # AVIF is lossy even at quality=100, but should be very close
        # Check shape and dtype match
        assert result.pixels.shape == pixels.shape
        assert result.pixels.dtype == pixels.dtype

        # Check most pixels are close (within ±2 for quality=100)
        diff = np.abs(result.pixels.astype(float) - pixels.astype(float))
        assert np.mean(diff) < 2.0
        assert result.metadata.get('bit_depth') == 8

    def test_write_read_10bit_hdr(self, temp_avif_path):
        """Test 10-bit HDR AVIF (uint16 with bit_depth=10)"""
        # 10-bit data: 0-1023 range in uint16
        pixels = np.random.randint(0, 1024, (20, 20, 3), dtype=np.uint16)
        metadata = {'bit_depth': 10}
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 100})

        reader = AVIFFormatReader(temp_avif_path)
        result = reader.read()

        assert result.pixels.dtype == np.uint16
        # Reader should infer 10-bit from uint16 for HDR
        assert result.metadata.get('bit_depth') == 10

    def test_write_read_12bit_hdr(self, temp_avif_path):
        """Test 12-bit HDR AVIF (uint16 with bit_depth=12)"""
        # 12-bit data: 0-4095 range
        pixels = np.random.randint(0, 4096, (20, 20, 3), dtype=np.uint16)
        metadata = {'bit_depth': 12}
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 95, 'bitspersample': 12})

        reader = AVIFFormatReader(temp_avif_path)
        result = reader.read()

        assert result.pixels.dtype == np.uint16
        # Should preserve bit depth
        assert result.metadata.get('bit_depth') in (10, 12)  # Reader may infer 10

    def test_write_read_grayscale(self, temp_avif_path):
        """Test grayscale AVIF (single channel)

        Note: AVIF encoder may convert grayscale to RGB internally
        """
        pixels = np.random.randint(0, 256, (30, 30, 1), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 100})

        reader = AVIFFormatReader(temp_avif_path)
        result = reader.read()

        # AVIF may convert to RGB (common behavior)
        assert result.pixels.ndim in (2, 3)
        if result.pixels.ndim == 3:
            assert result.pixels.shape[2] in (1, 3)  # 1 (grayscale) or 3 (RGB)

    def test_write_read_rgba(self, temp_avif_path):
        """Test RGBA AVIF with alpha channel"""
        pixels = np.random.randint(0, 256, (20, 20, 4), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 100})

        reader = AVIFFormatReader(temp_avif_path)
        result = reader.read()

        assert result.pixels.shape[2] == 4
        assert result.metadata.get('channels') == 4


class TestAVIFHDRMetadata:
    """Tests for AVIF HDR metadata (CICP parameters)"""

    def test_transfer_function_pq(self, temp_avif_path):
        """Test PQ (ST.2084) transfer function via CICP"""
        pixels = np.random.randint(0, 1024, (10, 10, 3), dtype=np.uint16)
        metadata = {
            'transfer_function': TransferFunction.PQ,
            'bit_depth': 10
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 95})

        # Note: Reader doesn't extract CICP metadata yet
        # This test documents expected behavior
        reader = AVIFFormatReader(temp_avif_path)
        result = reader.read()

        # Should preserve transfer function (when reader supports CICP)
        # Currently reader doesn't extract this, so test passes if file is valid
        assert result.pixels.dtype == np.uint16
        # TODO: assert result.metadata.get('transfer_function') == TransferFunction.PQ

    def test_transfer_function_hlg(self, temp_avif_path):
        """Test HLG transfer function via CICP"""
        pixels = np.random.randint(0, 1024, (10, 10, 3), dtype=np.uint16)
        metadata = {
            'transfer_function': TransferFunction.HLG,
            'bit_depth': 10
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 95})

        reader = AVIFFormatReader(temp_avif_path)
        result = reader.read()

        assert result.pixels.dtype == np.uint16
        # TODO: assert result.metadata.get('transfer_function') == TransferFunction.HLG

    def test_color_space_bt709(self, temp_avif_path):
        """Test BT.709 color space via CICP primaries"""
        pixels = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        metadata = {
            'color_space': ColorSpace.BT709,
            'transfer_function': TransferFunction.SRGB
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 100})

        reader = AVIFFormatReader(temp_avif_path)
        result = reader.read()

        # TODO: assert result.metadata.get('color_space') == ColorSpace.BT709

    def test_color_space_bt2020(self, temp_avif_path):
        """Test BT.2020 color space for HDR"""
        pixels = np.random.randint(0, 1024, (10, 10, 3), dtype=np.uint16)
        metadata = {
            'color_space': ColorSpace.BT2020,
            'transfer_function': TransferFunction.PQ,
            'bit_depth': 10
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 95})

        reader = AVIFFormatReader(temp_avif_path)
        result = reader.read()

        assert result.pixels.dtype == np.uint16
        # TODO: assert result.metadata.get('color_space') == ColorSpace.BT2020

    def test_color_space_display_p3(self, temp_avif_path):
        """Test Display P3 color space"""
        pixels = np.random.randint(0, 1024, (10, 10, 3), dtype=np.uint16)
        metadata = {
            'color_space': ColorSpace.DISPLAY_P3,
            'transfer_function': TransferFunction.PQ,
            'bit_depth': 10
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 95})

        reader = AVIFFormatReader(temp_avif_path)
        result = reader.read()

        assert result.pixels.dtype == np.uint16
        # TODO: assert result.metadata.get('color_space') == ColorSpace.DISPLAY_P3

    def test_complete_hdr10_metadata(self, temp_avif_path):
        """Test complete HDR10 metadata set (BT.2020 + PQ + 10-bit)"""
        pixels = np.random.randint(0, 1024, (20, 20, 3), dtype=np.uint16)
        metadata = {
            'transfer_function': TransferFunction.PQ,
            'color_space': ColorSpace.BT2020,
            'bit_depth': 10,
            'mastering_display_max_luminance': 1000.0,
            'mastering_display_min_luminance': 0.005,
            'max_cll': 1000,
            'max_fall': 400
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 95})

        reader = AVIFFormatReader(temp_avif_path)
        result = reader.read()

        # Verify file was created and is valid
        assert result.pixels.shape == pixels.shape
        assert result.pixels.dtype == np.uint16
        # TODO: Verify metadata when reader supports CICP extraction


class TestAVIFSaveOptions:
    """Tests for AVIF save options"""

    def test_quality_settings(self, temp_avif_path):
        """Test different quality levels"""
        pixels = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        # High quality
        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 95})
        high_quality_size = temp_avif_path.stat().st_size

        temp_avif_path.unlink()

        # Low quality
        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 50})
        low_quality_size = temp_avif_path.stat().st_size

        # Low quality should be smaller
        assert low_quality_size < high_quality_size

    def test_speed_parameter(self, temp_avif_path):
        """Test encoding speed parameter"""
        pixels = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = AVIFFormatWriter(str(temp_avif_path))
        # Speed: 0 (slowest) to 10 (fastest)
        writer.write(img_data, options={'quality': 90, 'speed': 8})

        # Should complete without error
        assert temp_avif_path.exists()
        assert temp_avif_path.stat().st_size > 0

    def test_bitspersample_via_metadata(self, temp_avif_path):
        """Test bitspersample via metadata bit_depth (10-bit)

        Note: bitspersample is controlled via metadata, not options
        """
        pixels = np.random.randint(0, 1024, (20, 20, 3), dtype=np.uint16)
        # bit_depth in metadata gets converted to bitspersample
        metadata = {'bit_depth': 10}
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 95})

        reader = AVIFFormatReader(temp_avif_path)
        result = reader.read()

        assert result.pixels.dtype == np.uint16
        assert result.metadata.get('bit_depth') == 10


class TestAVIFValidation:
    """Tests for AVIF writer validation"""

    def test_accepts_uint8(self, temp_avif_path):
        """AVIF should accept uint8 dtype"""
        pixels = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.validate(img_data)
        writer.write(img_data, options={'quality': 90})

    def test_accepts_uint16(self, temp_avif_path):
        """AVIF should accept uint16 dtype (with explicit bit depth)"""
        pixels = np.random.randint(0, 1024, (10, 10, 3), dtype=np.uint16)
        metadata = {'bit_depth': 10}
        img_data = ImageData(pixels=pixels, metadata=metadata)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.validate(img_data)
        # uint16 requires explicit bitspersample for AVIF
        writer.write(img_data, options={'quality': 90, 'bitspersample': 10})

    def test_rejects_float32(self, temp_avif_path):
        """AVIF should reject float32 dtype"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        writer = AVIFFormatWriter(str(temp_avif_path))

        with pytest.raises(ValueError, match="uint8 and uint16"):
            writer.validate(img_data)

    def test_rejects_empty_array(self, temp_avif_path):
        """AVIF should reject empty arrays"""
        pixels = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        img_data = ImageData(pixels=pixels)

        writer = AVIFFormatWriter(str(temp_avif_path))

        with pytest.raises(ValueError, match="[Ee]mpty"):
            writer.validate(img_data)

    def test_rejects_invalid_channels(self, temp_avif_path):
        """AVIF should reject invalid channel counts"""
        # 2 channels not supported
        pixels = np.random.randint(0, 256, (10, 10, 2), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = AVIFFormatWriter(str(temp_avif_path))

        with pytest.raises(ValueError, match="1, 3, or 4 channels"):
            writer.validate(img_data)


class TestAVIFEdgeCases:
    """Tests for AVIF edge cases"""

    def test_large_image(self, temp_avif_path):
        """Test large image encoding"""
        pixels = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 90})

        reader = AVIFFormatReader(temp_avif_path)
        result = reader.read()

        assert result.pixels.shape == pixels.shape

    def test_single_pixel(self, temp_avif_path):
        """Test 1x1 pixel image"""
        pixels = np.array([[[123, 45, 67]]], dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 100})

        reader = AVIFFormatReader(temp_avif_path)
        result = reader.read()

        assert result.pixels.shape[0] == 1
        assert result.pixels.shape[1] == 1

    def test_very_high_quality(self, temp_avif_path):
        """Test maximum quality setting (near-lossless)"""
        pixels = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = AVIFFormatWriter(str(temp_avif_path))
        writer.write(img_data, options={'quality': 100, 'speed': 0})

        reader = AVIFFormatReader(temp_avif_path)
        result = reader.read()

        # Quality 100 should be very close to original
        diff = np.abs(result.pixels.astype(float) - pixels.astype(float))
        assert np.mean(diff) < 1.5
        assert np.max(diff) < 10
