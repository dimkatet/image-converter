"""
Tests for ColorConvertFilter

Tests cover:
- Basic color space conversions (BT.709 ↔ BT.2020, Display P3)
- Identity transformation (same source/target)
- Alpha channel preservation
- Metadata updates
- Parameter validation
"""

import numpy as np
import pytest
import warnings

from image_pipeline.filters.color_convert import ColorConvertFilter
from image_pipeline.core.image_data import ImageData
from image_pipeline.types import ColorSpace, TransferFunction


class TestColorConvertFilter:
    """Tests for ColorConvertFilter"""

    def test_identity_conversion_warns(self):
        """Test that same source/target warns and returns unchanged"""
        converter = ColorConvertFilter(
            source=ColorSpace.BT709,
            target=ColorSpace.BT709
        )

        pixels = np.random.rand(4, 4, 3).astype(np.float32)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = converter.apply(pixels)

            assert len(w) == 1
            assert "identical" in str(w[0].message).lower()

        # Pixels should be unchanged
        assert np.array_equal(result, pixels)

    def test_bt709_to_bt2020_conversion(self):
        """Test BT.709 → BT.2020 conversion"""
        converter = ColorConvertFilter(
            source=ColorSpace.BT709,
            target=ColorSpace.BT2020
        )

        # Pure red in BT.709
        pixels = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        result = converter.apply(pixels)

        assert result.shape == pixels.shape
        assert result.dtype == np.float32
        # Red should still be primarily red channel
        assert result[0, 0, 0] > 0.5

    def test_bt2020_to_bt709_conversion(self):
        """Test BT.2020 → BT.709 conversion (reverse)"""
        converter = ColorConvertFilter(
            source=ColorSpace.BT2020,
            target=ColorSpace.BT709
        )

        pixels = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        result = converter.apply(pixels)

        assert result.shape == pixels.shape
        assert result.dtype == np.float32

    def test_preserves_alpha_channel(self):
        """Test that RGBA images preserve alpha"""
        converter = ColorConvertFilter(
            source=ColorSpace.BT709,
            target=ColorSpace.BT2020
        )

        # RGBA image
        pixels = np.random.rand(4, 4, 4).astype(np.float32)
        original_alpha = pixels[..., 3].copy()

        result = converter.apply(pixels)

        assert result.shape == (4, 4, 4)
        # Alpha channel should be unchanged
        assert np.array_equal(result[..., 3], original_alpha)

    def test_preserves_hdr_values(self):
        """Test that values > 1.0 are preserved (not clipped)"""
        converter = ColorConvertFilter(
            source=ColorSpace.BT709,
            target=ColorSpace.BT2020
        )

        # HDR values beyond 1.0
        pixels = np.array([[[2.0, 1.5, 3.0]]], dtype=np.float32)
        result = converter.apply(pixels)

        # Should not clip - some channels may still be > 1.0
        assert np.any(result > 1.0) or np.any(result < 0.0)

    def test_invalid_dtype_raises(self):
        """Test that integer dtype raises error"""
        converter = ColorConvertFilter(
            source=ColorSpace.BT709,
            target=ColorSpace.BT2020
        )

        pixels = np.array([[[128, 128, 128]]], dtype=np.uint8)

        with pytest.raises(ValueError, match="requires dtype"):
            converter.apply(pixels)

    def test_invalid_source_type_raises(self):
        """Test that non-ColorSpace source raises TypeError"""
        with pytest.raises(TypeError, match="must be ColorSpace enum"):
            ColorConvertFilter(source="BT.709", target=ColorSpace.BT2020)

    def test_invalid_target_type_raises(self):
        """Test that non-ColorSpace target raises TypeError"""
        with pytest.raises(TypeError, match="must be ColorSpace enum"):
            ColorConvertFilter(source=ColorSpace.BT709, target="BT.2020")

    def test_metadata_update(self):
        """Test that conversion updates color_space metadata"""
        converter = ColorConvertFilter(
            source=ColorSpace.BT709,
            target=ColorSpace.BT2020
        )

        pixels = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        img_data = ImageData(pixels, metadata={
            'transfer_function': TransferFunction.LINEAR,
            'color_space': ColorSpace.BT709
        })

        result = converter(img_data)

        assert result.metadata['color_space'] == ColorSpace.BT2020
        assert 'color_primaries' in result.metadata

    def test_nonlinear_transfer_function_raises(self):
        """Test that non-linear transfer function raises error"""
        converter = ColorConvertFilter(
            source=ColorSpace.BT709,
            target=ColorSpace.BT2020
        )

        pixels = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        img_data = ImageData(pixels, metadata={
            'transfer_function': TransferFunction.PQ  # Non-linear!
        })

        with pytest.raises(ValueError, match="requires linear RGB"):
            converter(img_data)


class TestDisplayP3Conversions:
    """Tests for Display P3 color space"""

    def test_bt709_to_displayp3(self):
        """Test BT.709 → Display P3 conversion"""
        converter = ColorConvertFilter(
            source=ColorSpace.BT709,
            target=ColorSpace.DISPLAY_P3
        )

        pixels = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
        result = converter.apply(pixels)

        assert result.shape == pixels.shape

    def test_displayp3_to_bt2020(self):
        """Test Display P3 → BT.2020 conversion"""
        converter = ColorConvertFilter(
            source=ColorSpace.DISPLAY_P3,
            target=ColorSpace.BT2020
        )

        pixels = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        result = converter.apply(pixels)

        assert result.shape == pixels.shape


class TestEdgeCases:
    """Edge case tests"""

    def test_zero_values(self):
        """Test conversion of all-zero image"""
        converter = ColorConvertFilter(
            source=ColorSpace.BT709,
            target=ColorSpace.BT2020
        )

        pixels = np.zeros((4, 4, 3), dtype=np.float32)
        result = converter.apply(pixels)

        # Black should remain black
        assert np.allclose(result, 0.0, atol=1e-6)

    def test_single_pixel(self):
        """Test with single pixel"""
        converter = ColorConvertFilter(
            source=ColorSpace.BT709,
            target=ColorSpace.BT2020
        )

        pixel = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        result = converter.apply(pixel)

        assert result.shape == (1, 1, 3)

    def test_large_image_shape(self):
        """Test that large images maintain shape"""
        converter = ColorConvertFilter(
            source=ColorSpace.BT709,
            target=ColorSpace.BT2020
        )

        pixels = np.random.rand(1080, 1920, 3).astype(np.float32)
        result = converter.apply(pixels)

        assert result.shape == (1080, 1920, 3)
        assert result.dtype == np.float32
