"""
Tests for NormalizeFilter

Tests cover:
- Basic normalization to [0, 1] range
- Custom output ranges [min_val, max_val]
- Explicit input ranges [min_in, max_in]
- Edge case: all pixels have same value (division by zero)
- Parameter validation (min_val < max_val, type checks)
- Dtype preservation (output is always float32)
"""

import numpy as np
import pytest

from image_pipeline.filters.normalize import NormalizeFilter
from image_pipeline.core.image_data import ImageData


class TestNormalizeFilter:
    """Tests for NormalizeFilter"""

    def test_basic_normalization_to_zero_one(self):
        """Test default normalization to [0, 1] range"""
        filter_obj = NormalizeFilter()

        # Input range [100, 200]
        pixels = np.array([
            [[100.0, 150.0, 200.0]],
            [[125.0, 175.0, 150.0]]
        ], dtype=np.float32)

        result = filter_obj.apply(pixels)

        # Should be normalized to [0, 1]
        assert result.dtype == np.float32
        assert result.shape == pixels.shape

        # 100 → 0.0, 200 → 1.0, 150 → 0.5
        assert np.allclose(result[0, 0, 0], 0.0, atol=1e-6)
        assert np.allclose(result[0, 0, 2], 1.0, atol=1e-6)
        assert np.allclose(result[0, 0, 1], 0.5, atol=1e-6)

    def test_normalization_to_custom_range(self):
        """Test normalization to custom [min_val, max_val] range"""
        filter_obj = NormalizeFilter(min_val=-1.0, max_val=1.0)

        # Input range [0, 100]
        pixels = np.array([[[0.0, 50.0, 100.0]]], dtype=np.float32)

        result = filter_obj.apply(pixels)

        # Should be normalized to [-1, 1]
        assert np.allclose(result[0, 0, 0], -1.0, atol=1e-6)
        assert np.allclose(result[0, 0, 1], 0.0, atol=1e-6)
        assert np.allclose(result[0, 0, 2], 1.0, atol=1e-6)

    def test_normalization_with_explicit_input_range(self):
        """Test normalization with explicit min_in/max_in"""
        filter_obj = NormalizeFilter(min_val=0.0, max_val=1.0, min_in=0.0, max_in=255.0)

        # Input values [0, 128, 255]
        pixels = np.array([[[0.0, 128.0, 255.0]]], dtype=np.float32)

        result = filter_obj.apply(pixels)

        # Should use explicit range [0, 255]
        assert np.allclose(result[0, 0, 0], 0.0, atol=1e-6)
        assert np.allclose(result[0, 0, 1], 128.0 / 255.0, atol=1e-6)
        assert np.allclose(result[0, 0, 2], 1.0, atol=1e-6)

    def test_normalization_with_partial_explicit_range(self):
        """Test that explicit min_in/max_in overrides auto-detection"""
        filter_obj = NormalizeFilter(min_val=0.0, max_val=1.0, min_in=0.0, max_in=1000.0)

        # Actual data range is [100, 200], but we specify [0, 1000]
        pixels = np.array([[[100.0, 150.0, 200.0]]], dtype=np.float32)

        result = filter_obj.apply(pixels)

        # Should use explicit range [0, 1000], not auto-detected [100, 200]
        assert np.allclose(result[0, 0, 0], 0.1, atol=1e-6)  # 100/1000
        assert np.allclose(result[0, 0, 1], 0.15, atol=1e-6)  # 150/1000
        assert np.allclose(result[0, 0, 2], 0.2, atol=1e-6)  # 200/1000

    def test_constant_image_returns_min_val(self):
        """Test that constant image (all same values) returns min_val everywhere"""
        filter_obj = NormalizeFilter(min_val=0.0, max_val=1.0)

        # All pixels are 42.0
        pixels = np.full((4, 4, 3), 42.0, dtype=np.float32)

        result = filter_obj.apply(pixels)

        # Should return min_val (0.0) everywhere to avoid division by zero
        assert result.dtype == np.float32
        assert result.shape == pixels.shape
        assert np.allclose(result, 0.0, atol=1e-6)

    def test_constant_image_with_custom_min_val(self):
        """Test constant image with custom min_val"""
        filter_obj = NormalizeFilter(min_val=0.5, max_val=1.0)

        pixels = np.full((2, 2, 3), 100.0, dtype=np.float32)

        result = filter_obj.apply(pixels)

        # Should return min_val (0.5) everywhere
        assert np.allclose(result, 0.5, atol=1e-6)

    def test_preserves_shape(self):
        """Test that normalization preserves array shape"""
        filter_obj = NormalizeFilter()

        shapes = [
            (8, 8, 3),      # RGB
            (16, 16, 1),    # Grayscale with channel
            (4, 4, 4),      # RGBA
            (10, 10),       # Grayscale 2D
        ]

        for shape in shapes:
            pixels = np.random.rand(*shape).astype(np.float32) * 1000
            result = filter_obj.apply(pixels)
            assert result.shape == shape

    def test_output_is_always_float32(self):
        """Test that output dtype is always float32 regardless of input"""
        filter_obj = NormalizeFilter()

        # Test different input dtypes
        dtypes = [np.float32, np.float64, np.uint8, np.uint16]

        for dtype in dtypes:
            pixels = np.array([[[100, 150, 200]]], dtype=dtype)
            result = filter_obj.apply(pixels)
            assert result.dtype == np.float32

    def test_validation_min_val_must_be_numeric(self):
        """Test that min_val must be numeric"""
        with pytest.raises(TypeError, match="min_val must be numeric"):
            NormalizeFilter(min_val="0.0", max_val=1.0)

    def test_validation_max_val_must_be_numeric(self):
        """Test that max_val must be numeric"""
        with pytest.raises(TypeError, match="max_val must be numeric"):
            NormalizeFilter(min_val=0.0, max_val="1.0")

    def test_validation_min_val_less_than_max_val(self):
        """Test that min_val must be less than max_val"""
        with pytest.raises(ValueError, match="min_val must be less than max_val"):
            NormalizeFilter(min_val=1.0, max_val=0.0)

    def test_validation_min_val_equal_to_max_val(self):
        """Test that min_val cannot equal max_val"""
        with pytest.raises(ValueError, match="min_val must be less than max_val"):
            NormalizeFilter(min_val=0.5, max_val=0.5)

    def test_repr(self):
        """Test string representation"""
        filter_obj = NormalizeFilter(min_val=0.0, max_val=1.0)
        assert repr(filter_obj) == "NormalizeFilter(min=0.0, max=1.0)"

    def test_repr_with_custom_range(self):
        """Test string representation with custom range"""
        filter_obj = NormalizeFilter(min_val=-1.0, max_val=2.0)
        assert repr(filter_obj) == "NormalizeFilter(min=-1.0, max=2.0)"

    def test_hdr_to_sdr_normalization(self):
        """Test realistic HDR to SDR normalization scenario"""
        # Normalize HDR values [0, 10000] to SDR [0, 1]
        filter_obj = NormalizeFilter(min_val=0.0, max_val=1.0, min_in=0.0, max_in=10000.0)

        # HDR values: black, 100 nits, 1000 nits, 10000 nits
        pixels = np.array([[[0.0, 100.0, 1000.0, 10000.0]]], dtype=np.float32)

        result = filter_obj.apply(pixels)

        assert np.allclose(result[0, 0, 0], 0.0, atol=1e-6)
        assert np.allclose(result[0, 0, 1], 0.01, atol=1e-6)  # 100/10000
        assert np.allclose(result[0, 0, 2], 0.1, atol=1e-6)   # 1000/10000
        assert np.allclose(result[0, 0, 3], 1.0, atol=1e-6)

    def test_negative_input_values(self):
        """Test normalization with negative input values"""
        filter_obj = NormalizeFilter(min_val=0.0, max_val=1.0)

        # Input range [-100, 100]
        pixels = np.array([[[-100.0, 0.0, 100.0]]], dtype=np.float32)

        result = filter_obj.apply(pixels)

        # Should normalize to [0, 1]
        assert np.allclose(result[0, 0, 0], 0.0, atol=1e-6)
        assert np.allclose(result[0, 0, 1], 0.5, atol=1e-6)
        assert np.allclose(result[0, 0, 2], 1.0, atol=1e-6)

    def test_with_image_data_object(self):
        """Test that filter works with ImageData object"""
        filter_obj = NormalizeFilter(min_val=0.0, max_val=1.0)

        # Create ImageData with range [100, 200]
        pixels = np.array([[[100.0, 150.0, 200.0]]], dtype=np.float32)
        img_data = ImageData(pixels=pixels)

        # Apply filter
        result_pixels = filter_obj.apply(img_data.pixels)

        # Create new ImageData
        result_data = ImageData(pixels=result_pixels)

        # Check normalization worked
        assert np.allclose(result_data.pixels[0, 0, 0], 0.0, atol=1e-6)
        assert np.allclose(result_data.pixels[0, 0, 1], 0.5, atol=1e-6)
        assert np.allclose(result_data.pixels[0, 0, 2], 1.0, atol=1e-6)

        # Metadata should auto-sync
        assert result_data.metadata['dtype'] == 'float32'
        assert result_data.metadata['shape'] == (1, 1, 3)
