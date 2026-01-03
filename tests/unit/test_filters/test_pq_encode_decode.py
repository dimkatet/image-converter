"""
Tests for PQ (Perceptual Quantizer ST.2084) encode/decode filters

Tests cover:
- Basic encoding/decoding with known values
- Round-trip conversion (encode → decode ≈ identity)
- Parameter validation (invalid types, negative values, warnings)
- Edge cases (zero, reference_peak, out of range)
- Metadata updates (transfer_function)
"""

import numpy as np
import pytest
import warnings

from image_pipeline.filters.pq_encode import PQEncodeFilter
from image_pipeline.filters.pq_decode import PQDecodeFilter
from image_pipeline.core.image_data import ImageData
from image_pipeline.types import TransferFunction


class TestPQEncodeFilter:
    """Tests for PQEncodeFilter"""

    def test_basic_encoding(self):
        """Test PQ encoding with known reference values"""
        # Test with a simple case: 100 nits (typical SDR white)
        encoder = PQEncodeFilter(reference_peak=10000.0)

        # Input: 100 nits in display-referred linear
        pixels = np.array([[[100.0, 100.0, 100.0]]], dtype=np.float32)
        encoded = encoder.apply(pixels)

        # PQ encoding should map 100 nits to 0.5081 (exact value from ST.2084 spec)
        # Using tolerance for floating point comparison
        assert encoded.shape == pixels.shape
        assert encoded.dtype == np.float32
        assert np.allclose(encoded[0, 0, 0], 0.5081, atol=0.001)

    def test_encoding_boundary_values(self):
        """Test PQ encoding at boundary values (0, reference_peak)"""
        encoder = PQEncodeFilter(reference_peak=10000.0)

        # Test black (0 nits) → should encode to 0
        black = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        encoded_black = encoder.apply(black)
        assert np.allclose(encoded_black, 0.0, atol=1e-6)

        # Test reference_peak (10000 nits) → should encode to 1.0
        peak = np.array([[[10000.0, 10000.0, 10000.0]]], dtype=np.float32)
        encoded_peak = encoder.apply(peak)
        assert np.allclose(encoded_peak, 1.0, atol=1e-6)

    def test_encoding_clips_negative_values(self):
        """Test that negative values are clipped to 0"""
        encoder = PQEncodeFilter(reference_peak=10000.0)

        pixels = np.array([[[-100.0, 0.0, 100.0]]], dtype=np.float32)
        encoded = encoder.apply(pixels)

        # Negative should be clipped to 0, then encoded to 0
        assert encoded[0, 0, 0] >= 0.0
        assert np.allclose(encoded[0, 0, 0], 0.0, atol=1e-6)

    def test_encoding_clips_values_above_peak(self):
        """Test that values above reference_peak are clipped"""
        encoder = PQEncodeFilter(reference_peak=10000.0)

        # Value above reference_peak
        pixels = np.array([[[15000.0, 10000.0, 5000.0]]], dtype=np.float32)
        encoded = encoder.apply(pixels)

        # 15000 should be clipped to 10000, then encoded to 1.0
        assert np.allclose(encoded[0, 0, 0], 1.0, atol=1e-6)

    def test_encoding_preserves_shape(self):
        """Test that encoding preserves array shape"""
        encoder = PQEncodeFilter(reference_peak=10000.0)

        # Different shapes
        shapes = [
            (8, 8, 3),      # RGB image
            (16, 16, 1),    # Grayscale
            (4, 4, 4),      # RGBA
        ]

        for shape in shapes:
            pixels = np.random.rand(*shape).astype(np.float32) * 1000  # 0-1000 nits
            encoded = encoder.apply(pixels)
            assert encoded.shape == shape

    def test_invalid_dtype_raises_error(self):
        """Test that non-float dtypes raise ValueError"""
        encoder = PQEncodeFilter(reference_peak=10000.0)

        # Integer array should be rejected
        pixels = np.array([[[100, 100, 100]]], dtype=np.uint8)

        with pytest.raises(ValueError, match="requires dtype"):
            encoder.apply(pixels)

    def test_metadata_update(self):
        """Test that encoding updates transfer_function metadata"""
        encoder = PQEncodeFilter(reference_peak=10000.0)

        pixels = np.array([[[100.0, 100.0, 100.0]]], dtype=np.float32)
        img_data = ImageData(pixels, metadata={'transfer_function': TransferFunction.LINEAR})

        result = encoder(img_data)

        assert result.metadata['transfer_function'] == TransferFunction.PQ

    def test_custom_reference_peak(self):
        """Test encoding with non-standard reference_peak"""
        # Lower reference_peak means higher encoded values for same input
        encoder_4000 = PQEncodeFilter(reference_peak=4000.0)
        encoder_10000 = PQEncodeFilter(reference_peak=10000.0)

        pixels = np.array([[[1000.0, 1000.0, 1000.0]]], dtype=np.float32)

        encoded_4000 = encoder_4000.apply(pixels)
        encoded_10000 = encoder_10000.apply(pixels)

        # 1000 nits is a larger fraction of 4000 than 10000
        # So encoded value should be higher for reference_peak=4000
        assert encoded_4000[0, 0, 0] > encoded_10000[0, 0, 0]


class TestPQDecodeFilter:
    """Tests for PQDecodeFilter"""

    def test_basic_decoding(self):
        """Test PQ decoding with known reference values"""
        decoder = PQDecodeFilter(peak_luminance=10000.0)

        # Input: PQ-encoded value 0.5081 (corresponds to 100 nits)
        pixels = np.array([[[0.5081, 0.5081, 0.5081]]], dtype=np.float32)
        decoded = decoder.apply(pixels)

        # Should decode back to approximately 100 nits
        assert decoded.shape == pixels.shape
        assert decoded.dtype == np.float32
        assert np.allclose(decoded[0, 0, 0], 100.0, atol=10.0)  # Tolerance for round-trip

    def test_decoding_boundary_values(self):
        """Test PQ decoding at boundary values (0, 1)"""
        decoder = PQDecodeFilter(peak_luminance=10000.0)

        # Test 0.0 → should decode to 0 nits
        black = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        decoded_black = decoder.apply(black)
        assert np.allclose(decoded_black, 0.0, atol=1e-6)

        # Test 1.0 → should decode to peak_luminance (10000 nits)
        peak = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
        decoded_peak = decoder.apply(peak)
        assert np.allclose(decoded_peak, 10000.0, atol=1.0)

    def test_decoding_rejects_out_of_range(self):
        """Test that values outside [0, 1] raise ValueError"""
        decoder = PQDecodeFilter(peak_luminance=10000.0)

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
        decoder = PQDecodeFilter(peak_luminance=10000.0)

        pixels = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        img_data = ImageData(pixels, metadata={'transfer_function': TransferFunction.PQ})

        result = decoder(img_data)

        assert result.metadata['transfer_function'] == TransferFunction.LINEAR


class TestPQRoundTrip:
    """Tests for PQ encode → decode round-trip conversion"""

    def test_roundtrip_identity(self):
        """Test that encode(decode(x)) ≈ x within tolerance"""
        encoder = PQEncodeFilter(reference_peak=10000.0)
        decoder = PQDecodeFilter(peak_luminance=10000.0)

        # Test with various luminance values (0 to 10000 nits)
        test_values = [0.0, 1.0, 10.0, 100.0, 1000.0, 5000.0, 10000.0]

        for value in test_values:
            original = np.array([[[value, value, value]]], dtype=np.float32)

            encoded = encoder.apply(original)
            decoded = decoder.apply(encoded)

            # Should be very close to original (within 0.1% or 1 nit, whichever is larger)
            tolerance = max(value * 0.001, 1.0)
            assert np.allclose(decoded, original, atol=tolerance), \
                f"Round-trip failed for {value} nits: got {decoded[0,0,0]}"

    def test_roundtrip_with_random_data(self):
        """Test round-trip with random HDR values"""
        encoder = PQEncodeFilter(reference_peak=10000.0)
        decoder = PQDecodeFilter(peak_luminance=10000.0)

        # Random values in typical HDR range (0-5000 nits)
        np.random.seed(42)  # For reproducibility
        original = np.random.rand(8, 8, 3).astype(np.float32) * 5000.0

        encoded = encoder.apply(original)
        decoded = decoder.apply(encoded)

        # Should match within reasonable tolerance
        # Use relative tolerance of 0.1% for most values, 1 nit absolute for near-zero
        assert np.allclose(decoded, original, rtol=0.001, atol=1.0)

    def test_roundtrip_preserves_zeros(self):
        """Test that zero values survive round-trip exactly"""
        encoder = PQEncodeFilter(reference_peak=10000.0)
        decoder = PQDecodeFilter(peak_luminance=10000.0)

        original = np.zeros((4, 4, 3), dtype=np.float32)
        encoded = encoder.apply(original)
        decoded = decoder.apply(encoded)

        assert np.allclose(decoded, 0.0, atol=1e-6)


class TestParameterValidation:
    """Tests for parameter validation in both filters"""

    def test_encode_negative_reference_peak_raises(self):
        """Test that negative reference_peak raises ValueError"""
        with pytest.raises(ValueError, match="must be positive"):
            PQEncodeFilter(reference_peak=-1000.0)

    def test_encode_zero_reference_peak_raises(self):
        """Test that zero reference_peak raises ValueError"""
        with pytest.raises(ValueError, match="must be positive"):
            PQEncodeFilter(reference_peak=0.0)

    def test_encode_invalid_type_raises(self):
        """Test that non-numeric reference_peak raises TypeError"""
        with pytest.raises(TypeError, match="must be numeric"):
            PQEncodeFilter(reference_peak="10000")

    def test_encode_high_reference_peak_warns(self):
        """Test that reference_peak > 10000 triggers warning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PQEncodeFilter(reference_peak=15000.0)

            assert len(w) == 1
            assert "exceeds ST.2084 standard" in str(w[0].message)

    def test_decode_negative_peak_luminance_raises(self):
        """Test that negative peak_luminance raises ValueError"""
        with pytest.raises(ValueError, match="must be positive"):
            PQDecodeFilter(peak_luminance=-1000.0)

    def test_decode_zero_peak_luminance_raises(self):
        """Test that zero peak_luminance raises ValueError"""
        with pytest.raises(ValueError, match="must be positive"):
            PQDecodeFilter(peak_luminance=0.0)

    def test_decode_invalid_type_raises(self):
        """Test that non-numeric peak_luminance raises TypeError"""
        with pytest.raises(TypeError, match="must be numeric"):
            PQDecodeFilter(peak_luminance="10000")

    def test_decode_high_peak_luminance_warns(self):
        """Test that peak_luminance > 10000 triggers warning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PQDecodeFilter(peak_luminance=15000.0)

            assert len(w) == 1
            assert "exceeds ST.2084 standard" in str(w[0].message)


class TestEdgeCases:
    """Tests for edge cases and special scenarios"""

    def test_empty_array_raises(self):
        """Test that empty arrays raise ValueError"""
        encoder = PQEncodeFilter(reference_peak=10000.0)

        empty = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="empty pixel array"):
            encoder.apply(empty)

    def test_single_pixel(self):
        """Test encoding/decoding single pixel"""
        encoder = PQEncodeFilter(reference_peak=10000.0)
        decoder = PQDecodeFilter(peak_luminance=10000.0)

        # Single pixel (1, 1, 3) shape
        pixel = np.array([[[100.0, 200.0, 300.0]]], dtype=np.float32)

        encoded = encoder.apply(pixel)
        decoded = decoder.apply(encoded)

        assert encoded.shape == (1, 1, 3)
        assert np.allclose(decoded, pixel, atol=1.0)

    def test_grayscale_image(self):
        """Test with single-channel (grayscale) image"""
        encoder = PQEncodeFilter(reference_peak=10000.0)
        decoder = PQDecodeFilter(peak_luminance=10000.0)

        # Grayscale image (H, W, 1)
        gray = np.random.rand(8, 8, 1).astype(np.float32) * 1000.0

        encoded = encoder.apply(gray)
        decoded = decoder.apply(encoded)

        assert encoded.shape == (8, 8, 1)
        assert np.allclose(decoded, gray, atol=1.0)
