"""
Tests for AbsoluteLuminanceFilter and RelativeLuminanceFilter

Tests cover:
- Scene-referred → Display-referred conversion (absolute luminance)
- Display-referred → Scene-referred conversion (relative luminance)
- Round-trip conversion
- Metadata updates (paper_white, max_cll, max_fall)
- Parameter validation
"""

import numpy as np
import pytest

from image_pipeline.filters.absolute_luminance import AbsoluteLuminanceFilter
from image_pipeline.filters.relative_luminance import RelativeLuminanceFilter
from image_pipeline.core.image_data import ImageData


class TestAbsoluteLuminanceFilter:
    """Tests for AbsoluteLuminanceFilter (scene → display)"""

    def test_basic_conversion(self):
        """Test scene-referred → display-referred conversion"""
        filter = AbsoluteLuminanceFilter(paper_white=100.0)

        # Scene-referred: 0.2 relative to paper white
        pixels = np.array([[[0.2, 0.5, 1.0]]], dtype=np.float32)
        result = filter.apply(pixels)

        # Display-referred: absolute nits
        assert result.dtype == np.float32
        assert np.isclose(result[0, 0, 0], 20.0)   # 0.2 * 100
        assert np.isclose(result[0, 0, 1], 50.0)   # 0.5 * 100
        assert np.isclose(result[0, 0, 2], 100.0)  # 1.0 * 100

    def test_different_paper_white(self):
        """Test with non-standard paper_white"""
        filter = AbsoluteLuminanceFilter(paper_white=203.0)

        pixels = np.array([[[1.0]]], dtype=np.float32)
        result = filter.apply(pixels)

        # 1.0 * 203 = 203 nits
        assert np.isclose(result[0, 0, 0], 203.0)

    def test_hdr_values_above_one(self):
        """Test that HDR values > 1.0 are handled correctly"""
        filter = AbsoluteLuminanceFilter(paper_white=100.0)

        # Scene values can exceed 1.0 (specular highlights, bright objects)
        pixels = np.array([[[2.0, 5.0, 10.0]]], dtype=np.float32)
        result = filter.apply(pixels)

        assert np.isclose(result[0, 0, 0], 200.0)   # 2.0 * 100
        assert np.isclose(result[0, 0, 1], 500.0)   # 5.0 * 100
        assert np.isclose(result[0, 0, 2], 1000.0)  # 10.0 * 100

    def test_metadata_update(self):
        """Test that metadata is updated with paper_white, max_cll, max_fall"""
        filter = AbsoluteLuminanceFilter(paper_white=100.0)

        pixels = np.array([
            [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
            [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]
        ], dtype=np.float32)
        img_data = ImageData(pixels)

        result = filter(img_data)

        # Check metadata
        assert result.metadata['paper_white'] == 100.0
        assert result.metadata['max_cll'] == 100  # max value = 1.0 * 100
        assert 'max_fall' in result.metadata

    def test_invalid_paper_white_raises(self):
        """Test that invalid paper_white raises error"""
        with pytest.raises(ValueError, match="must be positive"):
            AbsoluteLuminanceFilter(paper_white=0.0)

        with pytest.raises(ValueError, match="must be positive"):
            AbsoluteLuminanceFilter(paper_white=-100.0)

    def test_invalid_dtype_raises(self):
        """Test that integer dtype raises error"""
        filter = AbsoluteLuminanceFilter(paper_white=100.0)
        pixels = np.array([[[128]]], dtype=np.uint8)

        with pytest.raises(ValueError, match="requires dtype"):
            filter.apply(pixels)


class TestRelativeLuminanceFilter:
    """Tests for RelativeLuminanceFilter (display → scene)"""

    def test_basic_conversion(self):
        """Test display-referred → scene-referred conversion"""
        filter = RelativeLuminanceFilter(paper_white=100.0)

        # Display-referred: absolute nits
        pixels = np.array([[[20.0, 50.0, 100.0]]], dtype=np.float32)
        result = filter.apply(pixels)

        # Scene-referred: relative to paper white
        assert result.dtype == np.float32
        assert np.isclose(result[0, 0, 0], 0.2)  # 20 / 100
        assert np.isclose(result[0, 0, 1], 0.5)  # 50 / 100
        assert np.isclose(result[0, 0, 2], 1.0)  # 100 / 100

    def test_different_paper_white(self):
        """Test with non-standard paper_white"""
        filter = RelativeLuminanceFilter(paper_white=203.0)

        pixels = np.array([[[203.0]]], dtype=np.float32)
        result = filter.apply(pixels)

        # 203 / 203 = 1.0
        assert np.isclose(result[0, 0, 0], 1.0)

    def test_metadata_removes_maxcll_maxfall(self):
        """Test that max_cll/max_fall are removed (only valid for display-referred)"""
        filter = RelativeLuminanceFilter(paper_white=100.0)

        pixels = np.array([[[100.0]]], dtype=np.float32)
        img_data = ImageData(pixels, metadata={
            'max_cll': 1000,
            'max_fall': 500
        })

        result = filter(img_data)

        # max_cll and max_fall should be removed
        assert 'max_cll' not in result.metadata
        assert 'max_fall' not in result.metadata

    def test_metadata_preserves_paper_white(self):
        """Test that paper_white is added if not present"""
        filter = RelativeLuminanceFilter(paper_white=100.0)

        pixels = np.array([[[50.0]]], dtype=np.float32)
        img_data = ImageData(pixels)

        result = filter(img_data)

        assert result.metadata['paper_white'] == 100.0

    def test_invalid_paper_white_raises(self):
        """Test that invalid paper_white raises error"""
        with pytest.raises(ValueError, match="must be positive"):
            RelativeLuminanceFilter(paper_white=0.0)


class TestRoundTrip:
    """Tests for absolute → relative → absolute round-trip"""

    def test_roundtrip_identity(self):
        """Test that relative(absolute(x)) ≈ x"""
        abs_filter = AbsoluteLuminanceFilter(paper_white=100.0)
        rel_filter = RelativeLuminanceFilter(paper_white=100.0)

        original = np.array([[[0.1, 0.5, 1.0, 2.0]]], dtype=np.float32)

        # Scene → Display → Scene
        display = abs_filter.apply(original)
        recovered = rel_filter.apply(display)

        assert np.allclose(recovered, original, rtol=1e-6)

    def test_roundtrip_with_random_data(self):
        """Test round-trip with random scene values"""
        np.random.seed(42)
        abs_filter = AbsoluteLuminanceFilter(paper_white=100.0)
        rel_filter = RelativeLuminanceFilter(paper_white=100.0)

        # Scene values: 0-2.0 range (typical HDR)
        original = np.random.rand(8, 8, 3).astype(np.float32) * 2.0

        display = abs_filter.apply(original)
        recovered = rel_filter.apply(display)

        assert np.allclose(recovered, original, rtol=1e-6)

    def test_roundtrip_preserves_zeros(self):
        """Test that zeros survive round-trip"""
        abs_filter = AbsoluteLuminanceFilter(paper_white=100.0)
        rel_filter = RelativeLuminanceFilter(paper_white=100.0)

        zeros = np.zeros((4, 4, 3), dtype=np.float32)

        display = abs_filter.apply(zeros)
        recovered = rel_filter.apply(display)

        assert np.allclose(recovered, 0.0, atol=1e-6)

    def test_roundtrip_different_paper_white(self):
        """Test round-trip with different paper_white values"""
        for paper_white in [80.0, 100.0, 203.0, 500.0]:
            abs_filter = AbsoluteLuminanceFilter(paper_white=paper_white)
            rel_filter = RelativeLuminanceFilter(paper_white=paper_white)

            original = np.array([[[0.5, 1.0, 1.5]]], dtype=np.float32)

            display = abs_filter.apply(original)
            recovered = rel_filter.apply(display)

            assert np.allclose(recovered, original, rtol=1e-6), \
                f"Round-trip failed for paper_white={paper_white}"


class TestEdgeCases:
    """Edge case tests"""

    def test_absolute_single_pixel(self):
        """Test with single pixel"""
        filter = AbsoluteLuminanceFilter(paper_white=100.0)
        pixel = np.array([[[0.5]]], dtype=np.float32)

        result = filter.apply(pixel)

        assert result.shape == (1, 1, 1)
        assert np.isclose(result[0, 0, 0], 50.0)

    def test_relative_single_pixel(self):
        """Test with single pixel"""
        filter = RelativeLuminanceFilter(paper_white=100.0)
        pixel = np.array([[[50.0]]], dtype=np.float32)

        result = filter.apply(pixel)

        assert result.shape == (1, 1, 1)
        assert np.isclose(result[0, 0, 0], 0.5)

    def test_absolute_uniform_values(self):
        """Test with uniform array"""
        filter = AbsoluteLuminanceFilter(paper_white=100.0)
        pixels = np.full((4, 4, 3), 1.0, dtype=np.float32)

        result = filter.apply(pixels)

        assert np.allclose(result, 100.0)
