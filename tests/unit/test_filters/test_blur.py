"""
Tests for BlurFilter

Tests cover:
- Gaussian blur with various sigma values
- Edge case: sigma=0 returns unchanged image
- RGB and grayscale images
- Parameter validation (sigma >= 0, type checks)
- Blur effect verification (variance reduction)
- Dtype preservation
"""

import numpy as np
import pytest

from image_pipeline.filters.blur import BlurFilter
from image_pipeline.core.image_data import ImageData


class TestBlurFilter:
    """Tests for BlurFilter"""

    def test_basic_blur_reduces_variance(self):
        """Test that blur reduces image variance"""
        filter_obj = BlurFilter(sigma=2.0)

        # Create a noisy image with high variance
        np.random.seed(42)
        noisy = np.random.rand(20, 20, 3).astype(np.float32)

        original_variance = np.var(noisy)
        result = filter_obj.apply(noisy)

        # Blur should reduce variance
        blurred_variance = np.var(result)
        assert blurred_variance < original_variance
        assert result.shape == noisy.shape
        assert result.dtype == np.float32

    def test_sigma_zero_returns_unchanged(self):
        """Test that sigma=0 returns image unchanged"""
        filter_obj = BlurFilter(sigma=0.0)

        pixels = np.array([
            [[0.0, 0.5, 1.0], [0.2, 0.8, 0.3]],
            [[0.9, 0.1, 0.6], [0.4, 0.7, 0.5]]
        ], dtype=np.float32)

        result = filter_obj.apply(pixels)

        # Should be identical
        np.testing.assert_array_equal(result, pixels)

    def test_small_sigma_minimal_blur(self):
        """Test that small sigma produces minimal blur"""
        filter_obj = BlurFilter(sigma=0.5)

        # Sharp edge image
        pixels = np.zeros((10, 10, 3), dtype=np.float32)
        pixels[:5, :, :] = 1.0  # Top half white, bottom half black

        result = filter_obj.apply(pixels)

        # Result should be close to original but slightly blurred
        assert result.shape == pixels.shape
        # Check that edge is slightly softened (not exactly 0 or 1)
        assert 0.0 < result[5, 5, 0] < 1.0

    def test_large_sigma_heavy_blur(self):
        """Test that large sigma produces heavy blur"""
        filter_obj = BlurFilter(sigma=5.0)

        # Single bright pixel in center
        pixels = np.zeros((21, 21, 3), dtype=np.float32)
        pixels[10, 10, :] = 1.0

        result = filter_obj.apply(pixels)

        # Blur should spread the bright pixel
        assert result.shape == pixels.shape
        # Center should still be brightest but less than original
        assert result[10, 10, 0] < 1.0
        # Neighbors should now have some brightness
        assert result[10, 11, 0] > 0.0
        assert result[11, 10, 0] > 0.0

    def test_grayscale_2d_image(self):
        """Test blur on 2D grayscale image"""
        filter_obj = BlurFilter(sigma=1.0)

        # 2D grayscale array
        gray = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],  # Bright pixel in center
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ], dtype=np.float32)

        result = filter_obj.apply(gray)

        assert result.shape == gray.shape
        # Center pixel should be blurred (less than 1.0)
        assert result[2, 2] < 1.0
        # Neighbors should have some brightness
        assert result[2, 1] > 0.0
        assert result[2, 3] > 0.0
        assert result[1, 2] > 0.0
        assert result[3, 2] > 0.0

    def test_rgb_image_channels_blurred_separately(self):
        """Test that RGB channels are blurred independently"""
        filter_obj = BlurFilter(sigma=1.0)

        # Different pattern in each channel
        rgb = np.zeros((5, 5, 3), dtype=np.float32)
        rgb[2, 2, 0] = 1.0  # Red center
        rgb[1, 1, 1] = 1.0  # Green upper-left
        rgb[3, 3, 2] = 1.0  # Blue lower-right

        result = filter_obj.apply(rgb)

        assert result.shape == rgb.shape

        # Each channel should have blur centered around its bright pixel
        # Red channel: center should be brightest in red
        assert result[2, 2, 0] > result[1, 1, 0]
        # Green channel: upper-left should be brightest in green
        assert result[1, 1, 1] > result[2, 2, 1]
        # Blue channel: lower-right should be brightest in blue
        assert result[3, 3, 2] > result[2, 2, 2]

    def test_rgba_image(self):
        """Test blur on RGBA image"""
        filter_obj = BlurFilter(sigma=1.0)

        # RGBA image with different values
        rgba = np.zeros((5, 5, 4), dtype=np.float32)
        rgba[2, 2, :] = [1.0, 0.8, 0.6, 1.0]  # Bright pixel with full alpha

        result = filter_obj.apply(rgba)

        assert result.shape == rgba.shape
        # All channels including alpha should be blurred
        for channel in range(4):
            assert result[2, 2, channel] < rgba[2, 2, channel]
            assert result[2, 1, channel] > 0.0  # Neighbor has some value

    def test_preserves_dtype_float32(self):
        """Test that dtype is preserved (float32)"""
        filter_obj = BlurFilter(sigma=1.0)

        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        result = filter_obj.apply(pixels)

        assert result.dtype == np.float32

    def test_preserves_dtype_float64(self):
        """Test that dtype is preserved (float64)"""
        filter_obj = BlurFilter(sigma=1.0)

        pixels = np.random.rand(10, 10, 3).astype(np.float64)
        result = filter_obj.apply(pixels)

        assert result.dtype == np.float64

    def test_validation_sigma_must_be_numeric(self):
        """Test that sigma must be numeric"""
        with pytest.raises(TypeError, match="sigma must be numeric"):
            BlurFilter(sigma="1.0")

    def test_validation_sigma_must_be_non_negative(self):
        """Test that sigma must be non-negative"""
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            BlurFilter(sigma=-1.0)

    def test_repr_default(self):
        """Test string representation with default sigma"""
        filter_obj = BlurFilter()
        assert repr(filter_obj) == "BlurFilter(sigma=1.0)"

    def test_repr_custom_sigma(self):
        """Test string representation with custom sigma"""
        filter_obj = BlurFilter(sigma=2.5)
        assert repr(filter_obj) == "BlurFilter(sigma=2.5)"

    def test_with_image_data_object(self):
        """Test that filter works with ImageData object"""
        filter_obj = BlurFilter(sigma=1.0)

        # Create noisy ImageData
        np.random.seed(42)
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        original_variance = np.var(img_data.pixels)

        # Apply filter
        result_pixels = filter_obj.apply(img_data.pixels)

        # Create new ImageData
        result_data = ImageData(pixels=result_pixels)

        # Blur should reduce variance
        blurred_variance = np.var(result_data.pixels)
        assert blurred_variance < original_variance

        # Metadata should auto-sync
        assert result_data.metadata['dtype'] == 'float32'
        assert result_data.metadata['shape'] == (10, 10, 3)

    def test_negative_sigma_integer_raises_error(self):
        """Test that negative sigma as integer raises error"""
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            BlurFilter(sigma=-5)

    def test_very_large_image(self):
        """Test blur on larger image"""
        filter_obj = BlurFilter(sigma=2.0)

        # 100x100 RGB image
        pixels = np.random.rand(100, 100, 3).astype(np.float32)
        result = filter_obj.apply(pixels)

        assert result.shape == pixels.shape
        assert result.dtype == np.float32
        # Variance should be reduced
        assert np.var(result) < np.var(pixels)
