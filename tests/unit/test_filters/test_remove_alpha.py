"""
Tests for RemoveAlphaFilter

Tests cover:
- Removing alpha channel from RGBA images
- Handling images without alpha channel (RGB, grayscale)
- Preserving dtype and shape
- Edge cases (single channel with alpha, already RGB)
"""

import numpy as np
import pytest

from image_pipeline.filters.remove_alpha import RemoveAlphaFilter
from image_pipeline.core.image_data import ImageData


class TestRemoveAlphaFilter:
    """Tests for RemoveAlphaFilter"""

    def test_removes_alpha_from_rgba(self):
        """Test removing alpha channel from RGBA image"""
        filter_obj = RemoveAlphaFilter()

        # RGBA image: 2x2 with different alpha values
        rgba = np.array([
            [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.8]],
            [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.3]]
        ], dtype=np.float32)

        result = filter_obj.apply(rgba)

        # Should have only RGB channels
        assert result.shape == (2, 2, 3)
        assert result.dtype == np.float32

        # RGB values should be preserved
        np.testing.assert_array_equal(result[0, 0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(result[0, 1], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(result[1, 0], [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(result[1, 1], [1.0, 1.0, 0.0])

    def test_preserves_rgb_image(self):
        """Test that RGB images are returned unchanged"""
        filter_obj = RemoveAlphaFilter()

        # RGB image (no alpha)
        rgb = np.array([
            [[1.0, 0.5, 0.0], [0.2, 0.8, 0.3]],
            [[0.1, 0.1, 0.9], [0.7, 0.6, 0.4]]
        ], dtype=np.float32)

        result = filter_obj.apply(rgb)

        # Should be identical
        assert result.shape == rgb.shape
        np.testing.assert_array_equal(result, rgb)

    def test_preserves_grayscale_image(self):
        """Test that grayscale images are returned unchanged"""
        filter_obj = RemoveAlphaFilter()

        # Grayscale image (2D array)
        gray = np.array([
            [0.0, 0.5],
            [0.8, 1.0]
        ], dtype=np.float32)

        result = filter_obj.apply(gray)

        # Should be identical
        assert result.shape == gray.shape
        np.testing.assert_array_equal(result, gray)

    def test_preserves_single_channel_image(self):
        """Test that single-channel 3D images are returned unchanged"""
        filter_obj = RemoveAlphaFilter()

        # Single channel (H, W, 1)
        single = np.array([
            [[0.0], [0.5]],
            [[0.8], [1.0]]
        ], dtype=np.float32)

        result = filter_obj.apply(single)

        # Should be identical
        assert result.shape == single.shape
        np.testing.assert_array_equal(result, single)

    def test_preserves_dtype_float64(self):
        """Test that dtype is preserved (float64)"""
        filter_obj = RemoveAlphaFilter()

        rgba = np.array([[[1.0, 0.5, 0.0, 0.8]]], dtype=np.float64)
        result = filter_obj.apply(rgba)

        assert result.dtype == np.float64
        assert result.shape == (1, 1, 3)

    def test_preserves_dtype_uint8(self):
        """Test that dtype is preserved (uint8)"""
        filter_obj = RemoveAlphaFilter()

        rgba = np.array([[[255, 128, 0, 200]]], dtype=np.uint8)
        result = filter_obj.apply(rgba)

        assert result.dtype == np.uint8
        assert result.shape == (1, 1, 3)
        np.testing.assert_array_equal(result[0, 0], [255, 128, 0])

    def test_preserves_dtype_uint16(self):
        """Test that dtype is preserved (uint16)"""
        filter_obj = RemoveAlphaFilter()

        rgba = np.array([[[65535, 32768, 0, 40000]]], dtype=np.uint16)
        result = filter_obj.apply(rgba)

        assert result.dtype == np.uint16
        assert result.shape == (1, 1, 3)
        np.testing.assert_array_equal(result[0, 0], [65535, 32768, 0])

    def test_handles_large_image(self):
        """Test with larger image dimensions"""
        filter_obj = RemoveAlphaFilter()

        # 100x100 RGBA image
        rgba = np.random.rand(100, 100, 4).astype(np.float32)
        result = filter_obj.apply(rgba)

        assert result.shape == (100, 100, 3)
        # First 3 channels should match
        np.testing.assert_array_equal(result, rgba[..., :3])

    def test_repr(self):
        """Test string representation"""
        filter_obj = RemoveAlphaFilter()
        assert repr(filter_obj) == "RemoveAlphaFilter()"

    def test_with_image_data_object(self):
        """Test that filter works with ImageData object through metadata sync"""
        filter_obj = RemoveAlphaFilter()

        # Create RGBA ImageData
        rgba = np.array([[[1.0, 0.5, 0.0, 0.8]]], dtype=np.float32)
        img_data = ImageData(pixels=rgba)

        # Initial state
        assert img_data.metadata['channels'] == 4
        assert img_data.metadata['shape'] == (1, 1, 4)

        # Apply filter
        result_pixels = filter_obj.apply(img_data.pixels)

        # Create new ImageData with result
        result_data = ImageData(pixels=result_pixels)

        # Metadata should auto-sync
        assert result_data.metadata['channels'] == 3
        assert result_data.metadata['shape'] == (1, 1, 3)
