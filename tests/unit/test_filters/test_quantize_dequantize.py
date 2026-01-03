"""
Tests for Quantize/Dequantize filters

Tests cover:
- Basic quantization/dequantization for common bit depths (8, 10, 12, 16)
- Round-trip conversion (dequantize → quantize ≈ identity)
- Parameter validation
- Metadata updates
"""

import numpy as np
import pytest
import warnings

from image_pipeline.filters.quantize import QuantizeFilter
from image_pipeline.filters.dequantize import DequantizeFilter
from image_pipeline.core.image_data import ImageData


class TestQuantizeFilter:
    """Tests for QuantizeFilter"""

    def test_quantize_8bit(self):
        """Test 8-bit quantization [0,1] → [0,255]"""
        quantizer = QuantizeFilter(bit_depth=8)

        # Test known values
        pixels = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        result = quantizer.apply(pixels)

        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 128  # 0.5 * 255 ≈ 128
        assert result[0, 0, 2] == 255

    def test_quantize_16bit(self):
        """Test 16-bit quantization [0,1] → [0,65535]"""
        quantizer = QuantizeFilter(bit_depth=16)

        pixels = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        result = quantizer.apply(pixels)

        assert result.dtype == np.uint16
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 32768  # 0.5 * 65535 ≈ 32768
        assert result[0, 0, 2] == 65535

    def test_quantize_10bit(self):
        """Test 10-bit quantization (stored in uint16)"""
        quantizer = QuantizeFilter(bit_depth=10)

        pixels = np.array([[[0.0, 1.0]]], dtype=np.float32)
        result = quantizer.apply(pixels)

        assert result.dtype == np.uint16
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 1023  # 2^10 - 1

    def test_quantize_12bit(self):
        """Test 12-bit quantization (stored in uint16)"""
        quantizer = QuantizeFilter(bit_depth=12)

        pixels = np.array([[[1.0]]], dtype=np.float32)
        result = quantizer.apply(pixels)

        assert result.dtype == np.uint16
        assert result[0, 0, 0] == 4095  # 2^12 - 1

    def test_quantize_clips_out_of_range(self):
        """Test that values outside [0,1] are clipped with warning"""
        quantizer = QuantizeFilter(bit_depth=8)

        pixels = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float32)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = quantizer.apply(pixels)

            # Should warn about clipping
            assert len(w) == 1
            assert "clipped" in str(w[0].message).lower()

        # Values should be clipped
        assert result[0, 0, 0] == 0    # -0.5 → 0
        assert result[0, 0, 1] == 128  # 0.5 → 128
        assert result[0, 0, 2] == 255  # 1.5 → 255

    def test_invalid_dtype_raises(self):
        """Test that integer input raises error"""
        quantizer = QuantizeFilter(bit_depth=8)
        pixels = np.array([[[100]]], dtype=np.uint8)

        with pytest.raises(ValueError, match="requires dtype"):
            quantizer.apply(pixels)

    def test_invalid_bit_depth_raises(self):
        """Test that unsupported bit depths raise error"""
        with pytest.raises(ValueError, match="Unsupported bit depth"):
            QuantizeFilter(bit_depth=24)

    def test_metadata_update(self):
        """Test that quantization updates bit_depth metadata"""
        quantizer = QuantizeFilter(bit_depth=16)
        pixels = np.array([[[0.5]]], dtype=np.float32)
        img_data = ImageData(pixels)

        result = quantizer(img_data)

        assert result.metadata['bit_depth'] == 16
        assert 'uint16' in result.metadata['dtype']


class TestDequantizeFilter:
    """Tests for DequantizeFilter"""

    def test_dequantize_8bit(self):
        """Test 8-bit dequantization [0,255] → [0,1]"""
        dequantizer = DequantizeFilter(bit_depth=8)

        pixels = np.array([[[0, 128, 255]]], dtype=np.uint8)
        result = dequantizer.apply(pixels)

        assert result.dtype == np.float32
        assert np.isclose(result[0, 0, 0], 0.0)
        assert np.isclose(result[0, 0, 1], 128/255, atol=0.001)
        assert np.isclose(result[0, 0, 2], 1.0)

    def test_dequantize_16bit(self):
        """Test 16-bit dequantization [0,65535] → [0,1]"""
        dequantizer = DequantizeFilter(bit_depth=16)

        pixels = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
        result = dequantizer.apply(pixels)

        assert result.dtype == np.float32
        assert np.isclose(result[0, 0, 0], 0.0)
        assert np.isclose(result[0, 0, 1], 0.5, atol=0.001)
        assert np.isclose(result[0, 0, 2], 1.0)

    def test_dequantize_10bit(self):
        """Test 10-bit dequantization"""
        dequantizer = DequantizeFilter(bit_depth=10)

        pixels = np.array([[[0, 1023]]], dtype=np.uint16)
        result = dequantizer.apply(pixels)

        assert np.isclose(result[0, 0, 0], 0.0)
        assert np.isclose(result[0, 0, 1], 1.0)

    def test_invalid_dtype_raises(self):
        """Test that float input raises error"""
        dequantizer = DequantizeFilter(bit_depth=8)
        pixels = np.array([[[0.5]]], dtype=np.float32)

        with pytest.raises(ValueError, match="requires dtype"):
            dequantizer.apply(pixels)

    def test_invalid_bit_depth_raises(self):
        """Test validation of bit_depth parameter"""
        with pytest.raises(ValueError, match="must be positive"):
            DequantizeFilter(bit_depth=0)

        with pytest.raises(ValueError, match="must be <= 32"):
            DequantizeFilter(bit_depth=64)

    def test_invalid_type_raises(self):
        """Test that non-integer bit_depth raises TypeError"""
        with pytest.raises(TypeError, match="must be int"):
            DequantizeFilter(bit_depth=8.5)


class TestRoundTrip:
    """Tests for quantize → dequantize round-trip"""

    @pytest.mark.parametrize("bit_depth", [8, 10, 12, 16])
    def test_roundtrip_identity(self, bit_depth):
        """Test that dequantize(quantize(x)) ≈ x for various bit depths"""
        quantizer = QuantizeFilter(bit_depth=bit_depth)
        dequantizer = DequantizeFilter(bit_depth=bit_depth)

        # Test with evenly spaced values
        original = np.linspace(0.0, 1.0, 11).reshape(1, 11, 1).astype(np.float32)

        quantized = quantizer.apply(original)
        recovered = dequantizer.apply(quantized)

        # Error should be less than half a quantization step
        max_error = 1.0 / (2 ** bit_depth)
        assert np.allclose(recovered, original, atol=max_error)

    def test_roundtrip_preserves_edges(self):
        """Test that 0.0 and 1.0 survive round-trip exactly"""
        quantizer = QuantizeFilter(bit_depth=8)
        dequantizer = DequantizeFilter(bit_depth=8)

        edges = np.array([[[0.0, 1.0]]], dtype=np.float32)

        quantized = quantizer.apply(edges)
        recovered = dequantizer.apply(quantized)

        assert np.isclose(recovered[0, 0, 0], 0.0, atol=1e-6)
        assert np.isclose(recovered[0, 0, 1], 1.0, atol=1e-6)

    def test_roundtrip_with_random_data(self):
        """Test round-trip with random values"""
        np.random.seed(42)
        quantizer = QuantizeFilter(bit_depth=16)
        dequantizer = DequantizeFilter(bit_depth=16)

        original = np.random.rand(8, 8, 3).astype(np.float32)

        quantized = quantizer.apply(original)
        recovered = dequantizer.apply(quantized)

        # 16-bit should have very high precision
        max_error = 1.0 / (2 ** 16)
        assert np.allclose(recovered, original, atol=max_error)


class TestEdgeCases:
    """Edge case tests"""

    def test_quantize_uniform_values(self):
        """Test quantization of uniform array"""
        quantizer = QuantizeFilter(bit_depth=8)
        pixels = np.full((4, 4, 3), 0.5, dtype=np.float32)

        result = quantizer.apply(pixels)

        assert np.all(result == 128)

    def test_dequantize_zeros(self):
        """Test dequantization of all-zero array"""
        dequantizer = DequantizeFilter(bit_depth=8)
        pixels = np.zeros((4, 4, 3), dtype=np.uint8)

        result = dequantizer.apply(pixels)

        assert np.allclose(result, 0.0)

    def test_quantize_single_pixel(self):
        """Test with single pixel"""
        quantizer = QuantizeFilter(bit_depth=8)
        pixel = np.array([[[0.5]]], dtype=np.float32)

        result = quantizer.apply(pixel)

        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0] == 128
