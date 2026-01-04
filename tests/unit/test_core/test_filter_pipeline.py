"""
Tests for FilterPipeline

Tests cover:
- Initialization with and without filters
- Adding filters (add method with chaining)
- Removing filters (remove method)
- Clearing all filters
- Sequential filter application
- Verbose mode
- __len__ and __repr__
- Chain-of-responsibility pattern verification
"""

import numpy as np
import pytest
from unittest.mock import Mock, call
from io import StringIO
import sys

from image_pipeline.core.filter_pipeline import FilterPipeline
from image_pipeline.core.image_data import ImageData
from image_pipeline.filters.base import ImageFilter


# Mock filters for testing
class MockAddFilter(ImageFilter):
    """Mock filter that adds a constant value to all pixels"""

    def __init__(self, value: float):
        self.value = value
        super().__init__()

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        return pixels + self.value

    def __repr__(self):
        return f"MockAddFilter({self.value})"


class MockMultiplyFilter(ImageFilter):
    """Mock filter that multiplies all pixels by a constant"""

    def __init__(self, factor: float):
        self.factor = factor
        super().__init__()

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        return pixels * self.factor

    def __repr__(self):
        return f"MockMultiplyFilter({self.factor})"


class MockSetMetadataFilter(ImageFilter):
    """Mock filter that sets a custom metadata field"""

    def __init__(self, key: str, value: str):
        self.key = key
        self.value = value
        super().__init__()

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        return pixels

    def update_metadata(self, img_data: ImageData) -> None:
        super().update_metadata(img_data)
        img_data.metadata[self.key] = self.value

    def __repr__(self):
        return f"MockSetMetadataFilter({self.key}={self.value})"


class TestFilterPipelineInitialization:
    """Tests for FilterPipeline initialization"""

    def test_init_empty(self):
        """Test initialization without filters"""
        pipeline = FilterPipeline()

        assert len(pipeline.filters) == 0
        assert len(pipeline) == 0

    def test_init_with_filter_list(self):
        """Test initialization with list of filters"""
        filters = [MockAddFilter(1.0), MockMultiplyFilter(2.0)]
        pipeline = FilterPipeline(filters=filters)

        assert len(pipeline.filters) == 2
        assert len(pipeline) == 2
        assert isinstance(pipeline.filters[0], MockAddFilter)
        assert isinstance(pipeline.filters[1], MockMultiplyFilter)

    def test_init_with_none_creates_empty_list(self):
        """Test that passing None creates empty filter list"""
        pipeline = FilterPipeline(filters=None)

        assert len(pipeline.filters) == 0


class TestFilterPipelineAdd:
    """Tests for adding filters to pipeline"""

    def test_add_single_filter(self):
        """Test adding a single filter"""
        pipeline = FilterPipeline()
        filter_obj = MockAddFilter(5.0)

        pipeline.add(filter_obj)

        assert len(pipeline) == 1
        assert pipeline.filters[0] is filter_obj

    def test_add_multiple_filters(self):
        """Test adding multiple filters"""
        pipeline = FilterPipeline()

        pipeline.add(MockAddFilter(1.0))
        pipeline.add(MockMultiplyFilter(2.0))
        pipeline.add(MockAddFilter(3.0))

        assert len(pipeline) == 3

    def test_add_returns_self_for_chaining(self):
        """Test that add() returns self for method chaining"""
        pipeline = FilterPipeline()

        result = pipeline.add(MockAddFilter(1.0))

        assert result is pipeline

    def test_add_chaining(self):
        """Test fluent interface with chained add calls"""
        pipeline = FilterPipeline()

        pipeline.add(MockAddFilter(1.0)).add(MockMultiplyFilter(2.0)).add(MockAddFilter(3.0))

        assert len(pipeline) == 3
        assert isinstance(pipeline.filters[0], MockAddFilter)
        assert isinstance(pipeline.filters[1], MockMultiplyFilter)
        assert isinstance(pipeline.filters[2], MockAddFilter)


class TestFilterPipelineRemove:
    """Tests for removing filters from pipeline"""

    def test_remove_by_index(self):
        """Test removing filter by index"""
        pipeline = FilterPipeline([
            MockAddFilter(1.0),
            MockMultiplyFilter(2.0),
            MockAddFilter(3.0)
        ])

        pipeline.remove(1)  # Remove middle filter

        assert len(pipeline) == 2
        assert isinstance(pipeline.filters[0], MockAddFilter)
        assert isinstance(pipeline.filters[1], MockAddFilter)

    def test_remove_first_filter(self):
        """Test removing first filter"""
        pipeline = FilterPipeline([
            MockAddFilter(1.0),
            MockMultiplyFilter(2.0)
        ])

        pipeline.remove(0)

        assert len(pipeline) == 1
        assert isinstance(pipeline.filters[0], MockMultiplyFilter)

    def test_remove_last_filter(self):
        """Test removing last filter"""
        pipeline = FilterPipeline([
            MockAddFilter(1.0),
            MockMultiplyFilter(2.0)
        ])

        pipeline.remove(1)

        assert len(pipeline) == 1
        assert isinstance(pipeline.filters[0], MockAddFilter)

    def test_remove_invalid_index_ignored(self):
        """Test that removing with invalid index is ignored"""
        pipeline = FilterPipeline([MockAddFilter(1.0)])

        pipeline.remove(5)  # Out of bounds

        assert len(pipeline) == 1  # No change

    def test_remove_negative_index_ignored(self):
        """Test that negative index is ignored"""
        pipeline = FilterPipeline([MockAddFilter(1.0)])

        pipeline.remove(-1)

        assert len(pipeline) == 1  # No change

    def test_remove_returns_self_for_chaining(self):
        """Test that remove() returns self for chaining"""
        pipeline = FilterPipeline([MockAddFilter(1.0)])

        result = pipeline.remove(0)

        assert result is pipeline


class TestFilterPipelineClear:
    """Tests for clearing filters"""

    def test_clear_removes_all_filters(self):
        """Test that clear() removes all filters"""
        pipeline = FilterPipeline([
            MockAddFilter(1.0),
            MockMultiplyFilter(2.0),
            MockAddFilter(3.0)
        ])

        pipeline.clear()

        assert len(pipeline) == 0
        assert len(pipeline.filters) == 0

    def test_clear_on_empty_pipeline(self):
        """Test clear() on already empty pipeline"""
        pipeline = FilterPipeline()

        pipeline.clear()

        assert len(pipeline) == 0

    def test_clear_returns_self_for_chaining(self):
        """Test that clear() returns self for chaining"""
        pipeline = FilterPipeline([MockAddFilter(1.0)])

        result = pipeline.clear()

        assert result is pipeline


class TestFilterPipelineApply:
    """Tests for applying filters sequentially"""

    def test_apply_single_filter(self):
        """Test applying a single filter"""
        pipeline = FilterPipeline([MockAddFilter(5.0)])

        pixels = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        img_data = ImageData(pixels=pixels)

        result = pipeline.apply(img_data)

        # Should add 5.0 to all values
        expected = np.array([[[6.0, 7.0, 8.0]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.pixels, expected)

    def test_apply_multiple_filters_sequential(self):
        """Test that filters are applied sequentially"""
        # Add 1.0, then multiply by 2.0: (x + 1) * 2
        pipeline = FilterPipeline([
            MockAddFilter(1.0),
            MockMultiplyFilter(2.0)
        ])

        pixels = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        img_data = ImageData(pixels=pixels)

        result = pipeline.apply(img_data)

        # (1+1)*2=4, (2+1)*2=6, (3+1)*2=8
        expected = np.array([[[4.0, 6.0, 8.0]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.pixels, expected)

    def test_apply_order_matters(self):
        """Test that filter order matters"""
        # Multiply by 2.0, then add 1.0: (x * 2) + 1
        pipeline = FilterPipeline([
            MockMultiplyFilter(2.0),
            MockAddFilter(1.0)
        ])

        pixels = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        img_data = ImageData(pixels=pixels)

        result = pipeline.apply(img_data)

        # 1*2+1=3, 2*2+1=5, 3*2+1=7
        expected = np.array([[[3.0, 5.0, 7.0]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.pixels, expected)

    def test_apply_preserves_original_image_data(self):
        """Test that original ImageData is not modified"""
        pipeline = FilterPipeline([MockAddFilter(10.0)])

        pixels = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        img_data = ImageData(pixels=pixels)
        original_pixels = img_data.pixels.copy()

        result = pipeline.apply(img_data)

        # Original should be unchanged
        np.testing.assert_array_equal(img_data.pixels, original_pixels)
        # Result should be different
        assert not np.array_equal(result.pixels, original_pixels)

    def test_apply_empty_pipeline_returns_copy(self):
        """Test that empty pipeline returns a copy of input"""
        pipeline = FilterPipeline()

        pixels = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        img_data = ImageData(pixels=pixels)

        result = pipeline.apply(img_data)

        # Should be equal but different object
        np.testing.assert_array_equal(result.pixels, img_data.pixels)
        assert result is not img_data
        assert result.pixels is not img_data.pixels

    def test_apply_updates_metadata(self):
        """Test that filters can update metadata"""
        pipeline = FilterPipeline([
            MockSetMetadataFilter('processed', 'true'),
            MockSetMetadataFilter('version', '1.0')
        ])

        pixels = np.array([[[1.0]]], dtype=np.float32)
        img_data = ImageData(pixels=pixels)

        result = pipeline.apply(img_data)

        assert result.metadata['processed'] == 'true'
        assert result.metadata['version'] == '1.0'

    def test_apply_verbose_mode(self, capsys):
        """Test verbose mode prints filter information"""
        pipeline = FilterPipeline([
            MockAddFilter(1.0),
            MockMultiplyFilter(2.0)
        ])

        pixels = np.array([[[1.0]]], dtype=np.float32)
        img_data = ImageData(pixels=pixels)

        pipeline.apply(img_data, verbose=True)

        captured = capsys.readouterr()
        assert 'Step 1/2' in captured.out
        assert 'Step 2/2' in captured.out
        assert 'MockAddFilter' in captured.out
        assert 'MockMultiplyFilter' in captured.out

    def test_apply_verbose_false_no_output(self, capsys):
        """Test that verbose=False produces no output"""
        pipeline = FilterPipeline([MockAddFilter(1.0)])

        pixels = np.array([[[1.0]]], dtype=np.float32)
        img_data = ImageData(pixels=pixels)

        pipeline.apply(img_data, verbose=False)

        captured = capsys.readouterr()
        assert captured.out == ''


class TestFilterPipelineRepr:
    """Tests for string representation"""

    def test_repr_empty_pipeline(self):
        """Test __repr__ for empty pipeline"""
        pipeline = FilterPipeline()

        repr_str = repr(pipeline)

        assert 'FilterPipeline' in repr_str
        assert '[]' in repr_str

    def test_repr_with_filters(self):
        """Test __repr__ with filters"""
        pipeline = FilterPipeline([
            MockAddFilter(1.0),
            MockMultiplyFilter(2.0)
        ])

        repr_str = repr(pipeline)

        assert 'FilterPipeline' in repr_str
        assert 'MockAddFilter(1.0)' in repr_str
        assert 'MockMultiplyFilter(2.0)' in repr_str


class TestFilterPipelineLen:
    """Tests for __len__ method"""

    def test_len_empty(self):
        """Test len() on empty pipeline"""
        pipeline = FilterPipeline()
        assert len(pipeline) == 0

    def test_len_with_filters(self):
        """Test len() with filters"""
        pipeline = FilterPipeline([
            MockAddFilter(1.0),
            MockMultiplyFilter(2.0),
            MockAddFilter(3.0)
        ])
        assert len(pipeline) == 3

    def test_len_after_add(self):
        """Test len() updates after add"""
        pipeline = FilterPipeline()
        assert len(pipeline) == 0

        pipeline.add(MockAddFilter(1.0))
        assert len(pipeline) == 1

    def test_len_after_remove(self):
        """Test len() updates after remove"""
        pipeline = FilterPipeline([MockAddFilter(1.0)])
        assert len(pipeline) == 1

        pipeline.remove(0)
        assert len(pipeline) == 0

    def test_len_after_clear(self):
        """Test len() updates after clear"""
        pipeline = FilterPipeline([MockAddFilter(1.0), MockAddFilter(2.0)])
        assert len(pipeline) == 2

        pipeline.clear()
        assert len(pipeline) == 0
