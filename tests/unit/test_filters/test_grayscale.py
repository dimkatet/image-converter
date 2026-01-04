"""
Tests for GrayscaleFilter

Tests cover:
- Three conversion methods: 'luminosity', 'average', 'lightness'
- Already grayscale images (2D arrays)
- Single-channel 3D images
- RGBA images (uses only RGB channels)
- Parameter validation (method type and value)
- Dtype preservation
"""

import numpy as np
import pytest

from image_pipeline.filters.grayscale import GrayscaleFilter
from image_pipeline.core.image_data import ImageData


class TestGrayscaleFilter:
    """Tests for GrayscaleFilter"""

    @pytest.mark.parametrize("method,expected", [
        # For RGB [1.0, 0.5, 0.0]:
        ('luminosity', 0.299 * 1.0 + 0.587 * 0.5 + 0.114 * 0.0),  # ≈ 0.5925
        ('average', (1.0 + 0.5 + 0.0) / 3),                        # = 0.5
        ('lightness', (1.0 + 0.0) / 2),                            # = 0.5
    ])
    def test_conversion_methods(self, method, expected):
        """Test all three conversion methods with known values"""
        filter_obj = GrayscaleFilter(method=method)

        # RGB pixel: R=1.0, G=0.5, B=0.0
        rgb = np.array([[[1.0, 0.5, 0.0]]], dtype=np.float32)

        result = filter_obj.apply(rgb)

        # Result should be 2D (H, W) grayscale
        assert result.shape == (1, 1)
        assert result.dtype == np.float32
        assert np.isclose(result[0, 0], expected, atol=1e-6)

    def test_luminosity_method_detailed(self):
        """Test luminosity method with multiple pixels"""
        filter_obj = GrayscaleFilter(method='luminosity')

        # Different RGB values
        rgb = np.array([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # Pure R, G, B
            [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]   # White, gray, black
        ], dtype=np.float32)

        result = filter_obj.apply(rgb)

        assert result.shape == (2, 3)

        # Pure red: 0.299
        assert np.isclose(result[0, 0], 0.299, atol=1e-6)
        # Pure green: 0.587
        assert np.isclose(result[0, 1], 0.587, atol=1e-6)
        # Pure blue: 0.114
        assert np.isclose(result[0, 2], 0.114, atol=1e-6)
        # White: 1.0
        assert np.isclose(result[1, 0], 1.0, atol=1e-6)
        # Gray: 0.5
        assert np.isclose(result[1, 1], 0.5, atol=1e-6)
        # Black: 0.0
        assert np.isclose(result[1, 2], 0.0, atol=1e-6)

    def test_average_method(self):
        """Test average method"""
        filter_obj = GrayscaleFilter(method='average')

        # RGB values that average to specific values
        rgb = np.array([
            [[0.6, 0.3, 0.3]],  # avg = 0.4
            [[1.0, 0.5, 0.0]]   # avg = 0.5
        ], dtype=np.float32)

        result = filter_obj.apply(rgb)

        assert result.shape == (2, 1)
        assert np.isclose(result[0, 0], 0.4, atol=1e-6)
        assert np.isclose(result[1, 0], 0.5, atol=1e-6)

    def test_lightness_method(self):
        """Test lightness method (min + max) / 2"""
        filter_obj = GrayscaleFilter(method='lightness')

        # RGB values with specific min/max
        rgb = np.array([
            [[1.0, 0.5, 0.0]],  # max=1.0, min=0.0 → 0.5
            [[0.8, 0.2, 0.6]],  # max=0.8, min=0.2 → 0.5
            [[0.3, 0.3, 0.3]]   # max=0.3, min=0.3 → 0.3
        ], dtype=np.float32)

        result = filter_obj.apply(rgb)

        assert result.shape == (3, 1)
        assert np.isclose(result[0, 0], 0.5, atol=1e-6)
        assert np.isclose(result[1, 0], 0.5, atol=1e-6)
        assert np.isclose(result[2, 0], 0.3, atol=1e-6)

    def test_preserves_already_grayscale_2d(self):
        """Test that 2D grayscale images are returned unchanged"""
        filter_obj = GrayscaleFilter()

        # 2D grayscale array
        gray = np.array([
            [0.0, 0.5, 1.0],
            [0.3, 0.7, 0.2]
        ], dtype=np.float32)

        result = filter_obj.apply(gray)

        # Should be identical
        assert result.shape == gray.shape
        np.testing.assert_array_equal(result, gray)

    def test_squeezes_single_channel_3d(self):
        """Test that single-channel 3D images are squeezed to 2D"""
        filter_obj = GrayscaleFilter()

        # Single channel (H, W, 1)
        single = np.array([
            [[0.0], [0.5]],
            [[0.8], [1.0]]
        ], dtype=np.float32)

        result = filter_obj.apply(single)

        # Should be squeezed to 2D
        assert result.shape == (2, 2)
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.5
        assert result[1, 0] == 0.8
        assert result[1, 1] == 1.0

    def test_handles_rgba_ignores_alpha(self):
        """Test that RGBA images are converted using only RGB channels"""
        filter_obj = GrayscaleFilter(method='average')

        # RGBA with different alpha values
        rgba = np.array([
            [[1.0, 0.5, 0.0, 0.3]],  # avg of RGB = 0.5, alpha ignored
            [[0.6, 0.3, 0.3, 1.0]]   # avg of RGB = 0.4, alpha ignored
        ], dtype=np.float32)

        result = filter_obj.apply(rgba)

        assert result.shape == (2, 1)
        assert np.isclose(result[0, 0], 0.5, atol=1e-6)
        assert np.isclose(result[1, 0], 0.4, atol=1e-6)

    def test_preserves_dtype_float32(self):
        """Test that dtype is preserved (float32)"""
        filter_obj = GrayscaleFilter()

        rgb = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        result = filter_obj.apply(rgb)

        assert result.dtype == np.float32

    def test_preserves_dtype_float64(self):
        """Test that dtype is preserved (float64)"""
        filter_obj = GrayscaleFilter()

        rgb = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float64)
        result = filter_obj.apply(rgb)

        assert result.dtype == np.float64

    def test_preserves_dtype_uint8(self):
        """Test that dtype is preserved (uint8)"""
        filter_obj = GrayscaleFilter(method='average')

        rgb = np.array([[[255, 128, 0]]], dtype=np.uint8)
        result = filter_obj.apply(rgb)

        assert result.dtype == np.uint8
        # (255 + 128 + 0) / 3 ≈ 127.67 → 127 (integer)
        assert result[0, 0] == 127

    def test_preserves_dtype_uint16(self):
        """Test that dtype is preserved (uint16)"""
        filter_obj = GrayscaleFilter(method='average')

        rgb = np.array([[[65535, 32768, 0]]], dtype=np.uint16)
        result = filter_obj.apply(rgb)

        assert result.dtype == np.uint16

    def test_handles_large_image(self):
        """Test with larger image dimensions"""
        filter_obj = GrayscaleFilter(method='luminosity')

        # 100x100 RGB image
        rgb = np.random.rand(100, 100, 3).astype(np.float32)
        result = filter_obj.apply(rgb)

        # Should be 2D grayscale
        assert result.shape == (100, 100)
        assert result.dtype == np.float32

    def test_validation_method_must_be_string(self):
        """Test that method parameter must be a string"""
        with pytest.raises(TypeError, match="method must be str"):
            GrayscaleFilter(method=123)

    def test_validation_method_must_be_valid(self):
        """Test that method must be one of valid methods"""
        with pytest.raises(ValueError, match="method must be one of"):
            GrayscaleFilter(method='invalid')

    def test_validation_accepts_all_valid_methods(self):
        """Test that all valid methods are accepted"""
        valid_methods = ['luminosity', 'average', 'lightness']

        for method in valid_methods:
            # Should not raise
            filter_obj = GrayscaleFilter(method=method)
            assert filter_obj.method == method

    def test_repr_default(self):
        """Test string representation with default method"""
        filter_obj = GrayscaleFilter()
        assert repr(filter_obj) == "GrayscaleFilter(method='luminosity')"

    def test_repr_custom_method(self):
        """Test string representation with custom method"""
        filter_obj = GrayscaleFilter(method='average')
        assert repr(filter_obj) == "GrayscaleFilter(method='average')"

    def test_with_image_data_object(self):
        """Test that filter works with ImageData object"""
        filter_obj = GrayscaleFilter(method='luminosity')

        # Create RGB ImageData
        rgb = np.array([[[1.0, 0.5, 0.0]]], dtype=np.float32)
        img_data = ImageData(pixels=rgb)

        # Initial state
        assert img_data.metadata['channels'] == 3
        assert img_data.metadata['shape'] == (1, 1, 3)

        # Apply filter
        result_pixels = filter_obj.apply(img_data.pixels)

        # Create new ImageData
        result_data = ImageData(pixels=result_pixels)

        # Should be grayscale (2D)
        assert result_data.pixels.shape == (1, 1)
        assert result_data.metadata['shape'] == (1, 1)
        # For 2D arrays, channels should be 1 or might not be set
        # (depends on ImageData implementation)
