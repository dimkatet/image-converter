"""
Tests for sRGB encode/decode filters

Tests cover:
- Basic encoding/decoding with known values
- Round-trip conversion (encode → decode ≈ identity)
- Correct transfer function (gamma 2.4 with linear segment)
- Edge cases (zero, one, linear threshold)
- Metadata updates (transfer_function)
- Shape preservation and dtype validation
"""

import numpy as np
import pytest

from image_pipeline.filters.srgb_encode import SRGBEncodeFilter
from image_pipeline.filters.srgb_decode import SRGBDecodeFilter
from image_pipeline.core.image_data import ImageData
from image_pipeline.types import TransferFunction


class TestSRGBEncodeFilter:
    """Tests for SRGBEncodeFilter"""

    def test_basic_encoding(self):
        """Test sRGB encoding with known reference values"""
        encoder = SRGBEncodeFilter()

        # Test value: 0.5 linear should encode to ~0.7353
        # Formula: 1.055 * (0.5)^(1/2.4) - 0.055 ≈ 0.7353
        pixels = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        encoded = encoder.apply(pixels)

        assert encoded.shape == pixels.shape
        assert encoded.dtype == np.float32
        assert np.allclose(encoded[0, 0, 0], 0.7353, atol=0.001)

    def test_encoding_linear_segment(self):
        """Test sRGB encoding uses linear segment for dark values"""
        encoder = SRGBEncodeFilter()

        # Test value in linear segment: 0.001 should encode to 0.001 * 12.92 = 0.01292
        pixels = np.array([[[0.001, 0.001, 0.001]]], dtype=np.float32)
        encoded = encoder.apply(pixels)

        expected = 0.001 * 12.92
        assert np.allclose(encoded[0, 0, 0], expected, atol=1e-5)

    def test_encoding_boundary_values(self):
        """Test sRGB encoding at boundary values (0, 1)"""
        encoder = SRGBEncodeFilter()

        # Test black (0.0) → should encode to 0.0
        black = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        encoded_black = encoder.apply(black)
        assert np.allclose(encoded_black, 0.0, atol=1e-6)

        # Test white (1.0) → should encode to very close to 1.0
        white = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
        encoded_white = encoder.apply(white)
        assert np.allclose(encoded_white, 1.0, atol=1e-5)

    def test_encoding_clips_negative_values(self):
        """Test that negative values are clipped to 0"""
        encoder = SRGBEncodeFilter()

        pixels = np.array([[[-0.5, 0.0, 0.5]]], dtype=np.float32)
        encoded = encoder.apply(pixels)

        # Negative should be clipped to 0, then encoded to 0
        assert encoded[0, 0, 0] >= 0.0
        assert np.allclose(encoded[0, 0, 0], 0.0, atol=1e-6)

    def test_encoding_clips_values_above_one(self):
        """Test that values above 1.0 are clipped"""
        encoder = SRGBEncodeFilter()

        # Value above 1.0 (out of SDR range)
        pixels = np.array([[[1.5, 1.0, 0.5]]], dtype=np.float32)
        encoded = encoder.apply(pixels)

        # 1.5 should be clipped to 1.0, then encoded to ~1.0
        assert np.allclose(encoded[0, 0, 0], 1.0, atol=1e-5)

    def test_encoding_preserves_shape(self):
        """Test that encoding preserves array shape"""
        encoder = SRGBEncodeFilter()

        # Different shapes
        shapes = [
            (8, 8, 3),      # RGB image
            (16, 16, 1),    # Grayscale
            (4, 4, 4),      # RGBA (though sRGB is typically RGB)
        ]

        for shape in shapes:
            pixels = np.random.rand(*shape).astype(np.float32)  # 0-1 range
            encoded = encoder.apply(pixels)
            assert encoded.shape == shape

    def test_invalid_dtype_raises_error(self):
        """Test that non-float dtypes raise ValueError"""
        encoder = SRGBEncodeFilter()

        # Integer array should be rejected
        pixels = np.array([[[100, 100, 100]]], dtype=np.uint8)

        with pytest.raises(ValueError, match="requires dtype"):
            encoder.apply(pixels)

    def test_metadata_update(self):
        """Test that encoding updates transfer_function metadata"""
        encoder = SRGBEncodeFilter()

        pixels = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        img_data = ImageData(pixels, metadata={'transfer_function': TransferFunction.LINEAR})

        result = encoder(img_data)

        assert result.metadata['transfer_function'] == TransferFunction.SRGB

    def test_encoding_threshold_continuity(self):
        """Test that encoding is continuous at linear threshold"""
        encoder = SRGBEncodeFilter()

        # Values around threshold 0.0031308
        threshold = 0.0031308
        below = np.array([[[threshold * 0.99, 0.0, 0.0]]], dtype=np.float32)
        above = np.array([[[threshold * 1.01, 0.0, 0.0]]], dtype=np.float32)

        encoded_below = encoder.apply(below)
        encoded_above = encoder.apply(above)

        # Should be relatively close (continuous function)
        diff = abs(encoded_above[0, 0, 0] - encoded_below[0, 0, 0])
        assert diff < 0.01  # Difference should be small


class TestSRGBDecodeFilter:
    """Tests for SRGBDecodeFilter"""

    def test_basic_decoding(self):
        """Test sRGB decoding with known reference values"""
        decoder = SRGBDecodeFilter()

        # Input: sRGB-encoded value 0.7353 (corresponds to 0.5 linear)
        pixels = np.array([[[0.7353, 0.7353, 0.7353]]], dtype=np.float32)
        decoded = decoder.apply(pixels)

        # Should decode back to approximately 0.5 linear
        assert decoded.shape == pixels.shape
        assert decoded.dtype == np.float32
        assert np.allclose(decoded[0, 0, 0], 0.5, atol=0.01)

    def test_decoding_linear_segment(self):
        """Test sRGB decoding uses linear segment for dark values"""
        decoder = SRGBDecodeFilter()

        # Test value in linear segment: 0.01292 should decode to 0.001
        # Formula: 0.01292 / 12.92 = 0.001
        pixels = np.array([[[0.01292, 0.01292, 0.01292]]], dtype=np.float32)
        decoded = decoder.apply(pixels)

        expected = 0.001
        assert np.allclose(decoded[0, 0, 0], expected, atol=1e-5)

    def test_decoding_boundary_values(self):
        """Test sRGB decoding at boundary values (0, 1)"""
        decoder = SRGBDecodeFilter()

        # Test 0.0 → should decode to 0.0 linear
        black = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        decoded_black = decoder.apply(black)
        assert np.allclose(decoded_black, 0.0, atol=1e-6)

        # Test 1.0 → should decode to very close to 1.0 linear
        white = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
        decoded_white = decoder.apply(white)
        assert np.allclose(decoded_white, 1.0, atol=1e-5)

    def test_decoding_rejects_out_of_range(self):
        """Test that values outside [0, 1] raise ValueError"""
        decoder = SRGBDecodeFilter()

        # Value > 1.0 should raise error
        pixels_high = np.array([[[1.5, 0.5, 0.5]]], dtype=np.float32)
        with pytest.raises(ValueError, match="expects values in"):
            decoder.apply(pixels_high)

        # Value < 0.0 should raise error
        pixels_low = np.array([[[-0.1, 0.5, 0.5]]], dtype=np.float32)
        with pytest.raises(ValueError, match="expects values in"):
            decoder.apply(pixels_low)

    def test_metadata_update(self):
        """Test that decoding updates transfer_function to LINEAR"""
        decoder = SRGBDecodeFilter()

        pixels = np.array([[[0.7, 0.7, 0.7]]], dtype=np.float32)
        img_data = ImageData(pixels, metadata={'transfer_function': TransferFunction.SRGB})

        result = decoder(img_data)

        assert result.metadata['transfer_function'] == TransferFunction.LINEAR


class TestSRGBRoundTrip:
    """Tests for sRGB encode → decode round-trip conversion"""

    def test_roundtrip_identity(self):
        """Test that encode(decode(x)) ≈ x within tolerance"""
        encoder = SRGBEncodeFilter()
        decoder = SRGBDecodeFilter()

        # Test with various linear values (0 to 1 for SDR)
        test_values = [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 1.0]

        for value in test_values:
            original = np.array([[[value, value, value]]], dtype=np.float32)

            encoded = encoder.apply(original)
            decoded = decoder.apply(encoded)

            # Should be very close to original (float32 precision)
            assert np.allclose(decoded, original, atol=1e-6), \
                f"Round-trip failed for {value}: got {decoded[0,0,0]}"

    def test_roundtrip_with_random_data(self):
        """Test round-trip with random SDR values"""
        encoder = SRGBEncodeFilter()
        decoder = SRGBDecodeFilter()

        # Random values in SDR range (0-1)
        np.random.seed(42)  # For reproducibility
        original = np.random.rand(8, 8, 3).astype(np.float32)

        encoded = encoder.apply(original)
        decoded = decoder.apply(encoded)

        # Should match within float32 precision
        assert np.allclose(decoded, original, atol=1e-6)

    def test_roundtrip_preserves_zeros(self):
        """Test that zero values survive round-trip exactly"""
        encoder = SRGBEncodeFilter()
        decoder = SRGBDecodeFilter()

        original = np.zeros((4, 4, 3), dtype=np.float32)
        encoded = encoder.apply(original)
        decoded = decoder.apply(encoded)

        assert np.allclose(decoded, 0.0, atol=1e-6)

    def test_roundtrip_preserves_ones(self):
        """Test that one values survive round-trip"""
        encoder = SRGBEncodeFilter()
        decoder = SRGBDecodeFilter()

        original = np.ones((4, 4, 3), dtype=np.float32)
        encoded = encoder.apply(original)
        decoded = decoder.apply(encoded)

        assert np.allclose(decoded, 1.0, atol=1e-5)


class TestEdgeCases:
    """Tests for edge cases and special scenarios"""

    def test_empty_array_raises(self):
        """Test that empty arrays raise ValueError"""
        encoder = SRGBEncodeFilter()

        empty = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="empty pixel array"):
            encoder.apply(empty)

    def test_single_pixel(self):
        """Test encoding/decoding single pixel"""
        encoder = SRGBEncodeFilter()
        decoder = SRGBDecodeFilter()

        # Single pixel (1, 1, 3) shape
        pixel = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)

        encoded = encoder.apply(pixel)
        decoded = decoder.apply(encoded)

        assert encoded.shape == (1, 1, 3)
        assert np.allclose(decoded, pixel, atol=1e-6)

    def test_grayscale_image(self):
        """Test with single-channel (grayscale) image"""
        encoder = SRGBEncodeFilter()
        decoder = SRGBDecodeFilter()

        # Grayscale image (H, W, 1)
        gray = np.random.rand(8, 8, 1).astype(np.float32)

        encoded = encoder.apply(gray)
        decoded = decoder.apply(encoded)

        assert encoded.shape == (8, 8, 1)
        assert np.allclose(decoded, gray, atol=1e-6)


class TestCorrectTransferFunction:
    """Tests to verify correct sRGB transfer function implementation"""

    def test_linear_segment_formula(self):
        """Verify linear segment uses correct formula: 12.92 * linear"""
        encoder = SRGBEncodeFilter()

        # Values below threshold should use linear formula
        test_values = [0.0001, 0.001, 0.003]

        for linear_val in test_values:
            pixels = np.array([[[linear_val, 0.0, 0.0]]], dtype=np.float32)
            encoded = encoder.apply(pixels)

            expected = 12.92 * linear_val
            assert np.allclose(encoded[0, 0, 0], expected, atol=1e-6)

    def test_power_segment_formula(self):
        """Verify power segment uses correct formula with gamma 2.4"""
        encoder = SRGBEncodeFilter()

        # Values above threshold should use power formula
        # Formula: 1.055 * linear^(1/2.4) - 0.055
        test_values = [0.01, 0.1, 0.5, 0.9]

        for linear_val in test_values:
            pixels = np.array([[[linear_val, 0.0, 0.0]]], dtype=np.float32)
            encoded = encoder.apply(pixels)

            expected = 1.055 * (linear_val ** (1.0 / 2.4)) - 0.055
            assert np.allclose(encoded[0, 0, 0], expected, atol=1e-5)

    def test_decode_linear_segment_formula(self):
        """Verify decode linear segment: srgb / 12.92"""
        decoder = SRGBDecodeFilter()

        # Values below sRGB threshold should use linear formula
        test_values = [0.001, 0.01, 0.04]

        for srgb_val in test_values:
            pixels = np.array([[[srgb_val, 0.0, 0.0]]], dtype=np.float32)
            decoded = decoder.apply(pixels)

            expected = srgb_val / 12.92
            assert np.allclose(decoded[0, 0, 0], expected, atol=1e-6)

    def test_decode_power_segment_formula(self):
        """Verify decode power segment with gamma 2.4"""
        decoder = SRGBDecodeFilter()

        # Values above sRGB threshold should use power formula
        # Formula: ((srgb + 0.055) / 1.055)^2.4
        test_values = [0.1, 0.5, 0.7, 0.9]

        for srgb_val in test_values:
            pixels = np.array([[[srgb_val, 0.0, 0.0]]], dtype=np.float32)
            decoded = decoder.apply(pixels)

            expected = ((srgb_val + 0.055) / 1.055) ** 2.4
            assert np.allclose(decoded[0, 0, 0], expected, atol=1e-5)

    def test_not_simple_gamma_22(self):
        """Verify that sRGB is NOT simple gamma 2.2"""
        encoder = SRGBEncodeFilter()

        # Compare sRGB encoding vs simple gamma 2.2
        test_val = 0.5
        pixels = np.array([[[test_val, 0.0, 0.0]]], dtype=np.float32)

        srgb_encoded = encoder.apply(pixels)[0, 0, 0]
        simple_gamma_22 = test_val ** (1.0 / 2.2)

        # They should be different (sRGB ≠ simple gamma 2.2)
        assert not np.allclose(srgb_encoded, simple_gamma_22, atol=0.001)
