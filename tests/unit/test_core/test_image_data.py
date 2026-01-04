"""
Tests for ImageData

Tests cover:
- Initialization with pixels and metadata
- Automatic metadata synchronization (shape, dtype, channels)
- Properties: width, height, channels, format, shape, dtype
- Deep copy functionality
- Edge cases: 2D vs 3D arrays, different dtypes
- bit_depth preservation (10-bit, 12-bit in uint16)
"""

import numpy as np
import pytest

from image_pipeline.core.image_data import ImageData


class TestImageDataInitialization:
    """Tests for ImageData initialization"""

    def test_init_with_rgb_image(self):
        """Test initialization with RGB image"""
        pixels = np.random.rand(10, 20, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        assert img_data.pixels.shape == (10, 20, 3)
        assert img_data.dtype == np.float32
        assert img_data.channels == 3

    def test_init_with_grayscale_2d(self):
        """Test initialization with 2D grayscale image"""
        pixels = np.random.rand(10, 20).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        assert img_data.pixels.shape == (10, 20)
        assert img_data.dtype == np.float32
        assert img_data.channels == 1

    def test_init_with_grayscale_3d(self):
        """Test initialization with 3D single-channel image"""
        pixels = np.random.rand(10, 20, 1).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        assert img_data.pixels.shape == (10, 20, 1)
        assert img_data.dtype == np.float32
        assert img_data.channels == 1

    def test_init_with_rgba_image(self):
        """Test initialization with RGBA image"""
        pixels = np.random.rand(10, 20, 4).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        assert img_data.pixels.shape == (10, 20, 4)
        assert img_data.channels == 4

    def test_init_with_empty_metadata(self):
        """Test initialization without metadata"""
        pixels = np.random.rand(10, 20, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        # Metadata should be auto-populated
        assert 'shape' in img_data.metadata
        assert 'dtype' in img_data.metadata
        assert 'channels' in img_data.metadata
        assert 'bit_depth' in img_data.metadata

    def test_init_with_custom_metadata(self):
        """Test initialization with custom metadata"""
        pixels = np.random.rand(10, 20, 3).astype(np.float32)
        metadata = {
            'format': 'png',
            'filename': 'test.png',
            'custom_field': 'value'
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Custom metadata should be preserved
        assert img_data.metadata['format'] == 'png'
        assert img_data.metadata['filename'] == 'test.png'
        assert img_data.metadata['custom_field'] == 'value'

        # Auto-sync metadata should also be present
        assert img_data.metadata['shape'] == (10, 20, 3)
        assert img_data.metadata['dtype'] == 'float32'
        assert img_data.metadata['channels'] == 3


class TestImageDataMetadataSync:
    """Tests for automatic metadata synchronization"""

    def test_metadata_sync_shape(self):
        """Test that shape is synced to metadata"""
        pixels = np.random.rand(15, 25, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        assert img_data.metadata['shape'] == (15, 25, 3)

    def test_metadata_sync_dtype(self):
        """Test that dtype is synced to metadata as string"""
        pixels = np.random.rand(10, 10, 3).astype(np.float64)
        img_data = ImageData(pixels=pixels)

        assert img_data.metadata['dtype'] == 'float64'

    def test_metadata_sync_channels_rgb(self):
        """Test that channels is synced for RGB"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        assert img_data.metadata['channels'] == 3

    def test_metadata_sync_channels_grayscale_2d(self):
        """Test that channels is synced for 2D grayscale"""
        pixels = np.random.rand(10, 10).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        assert img_data.metadata['channels'] == 1

    def test_bit_depth_auto_calculated_float32(self):
        """Test bit_depth auto-calculation for float32"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        # float32 = 4 bytes * 8 = 32 bits
        assert img_data.metadata['bit_depth'] == 32

    def test_bit_depth_auto_calculated_uint8(self):
        """Test bit_depth auto-calculation for uint8"""
        pixels = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        assert img_data.metadata['bit_depth'] == 8

    def test_bit_depth_auto_calculated_uint16(self):
        """Test bit_depth auto-calculation for uint16"""
        pixels = np.random.randint(0, 65536, (10, 10, 3), dtype=np.uint16)
        img_data = ImageData(pixels=pixels)

        # uint16 = 2 bytes * 8 = 16 bits (but could be 10 or 12-bit)
        assert img_data.metadata['bit_depth'] == 16

    def test_bit_depth_preserved_when_explicit(self):
        """Test that explicit bit_depth is preserved (e.g., 10-bit in uint16)"""
        pixels = np.random.randint(0, 1024, (10, 10, 3), dtype=np.uint16)
        metadata = {'bit_depth': 10}  # 10-bit data stored in uint16
        img_data = ImageData(pixels=pixels, metadata=metadata)

        # Explicit bit_depth should be preserved
        assert img_data.metadata['bit_depth'] == 10

    def test_bit_depth_preserved_12bit(self):
        """Test that 12-bit depth is preserved in uint16 array"""
        pixels = np.random.randint(0, 4096, (10, 10, 3), dtype=np.uint16)
        metadata = {'bit_depth': 12}
        img_data = ImageData(pixels=pixels, metadata=metadata)

        assert img_data.metadata['bit_depth'] == 12


class TestImageDataProperties:
    """Tests for ImageData properties"""

    def test_property_shape(self):
        """Test shape property"""
        pixels = np.random.rand(15, 25, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        assert img_data.shape == (15, 25, 3)

    def test_property_dtype(self):
        """Test dtype property"""
        pixels = np.random.rand(10, 10, 3).astype(np.float64)
        img_data = ImageData(pixels=pixels)

        assert img_data.dtype == np.float64

    def test_property_width(self):
        """Test width property (shape[1])"""
        pixels = np.random.rand(10, 20, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        assert img_data.width == 20

    def test_property_height(self):
        """Test height property (shape[0])"""
        pixels = np.random.rand(10, 20, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        assert img_data.height == 10

    def test_property_channels_rgb(self):
        """Test channels property for RGB"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        assert img_data.channels == 3

    def test_property_channels_rgba(self):
        """Test channels property for RGBA"""
        pixels = np.random.rand(10, 10, 4).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        assert img_data.channels == 4

    def test_property_channels_grayscale_2d(self):
        """Test channels property for 2D grayscale"""
        pixels = np.random.rand(10, 10).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        assert img_data.channels == 1

    def test_property_format_default(self):
        """Test format property when not set"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        assert img_data.format == 'unknown'

    def test_property_format_custom(self):
        """Test format property when set in metadata"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        metadata = {'format': 'png'}
        img_data = ImageData(pixels=pixels, metadata=metadata)

        assert img_data.format == 'png'


class TestImageDataCopy:
    """Tests for ImageData copy functionality"""

    def test_copy_creates_new_instance(self):
        """Test that copy creates a new ImageData instance"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        copied = img_data.copy()

        assert copied is not img_data
        assert isinstance(copied, ImageData)

    def test_copy_deep_copies_pixels(self):
        """Test that copy deep-copies the pixel array"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        copied = img_data.copy()

        # Arrays should be equal but not the same object
        np.testing.assert_array_equal(copied.pixels, img_data.pixels)
        assert copied.pixels is not img_data.pixels

        # Modifying copy should not affect original
        copied.pixels[0, 0, 0] = 999.0
        assert img_data.pixels[0, 0, 0] != 999.0

    def test_copy_deep_copies_metadata(self):
        """Test that copy deep-copies metadata"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        metadata = {'format': 'png', 'custom': 'value'}
        img_data = ImageData(pixels=pixels, metadata=metadata)

        copied = img_data.copy()

        # Metadata should be equal but not the same object
        assert copied.metadata == img_data.metadata
        assert copied.metadata is not img_data.metadata

        # Modifying copy metadata should not affect original
        copied.metadata['custom'] = 'changed'
        assert img_data.metadata['custom'] == 'value'

    def test_copy_preserves_all_properties(self):
        """Test that copy preserves all properties"""
        pixels = np.random.rand(15, 25, 3).astype(np.float64)
        metadata = {
            'format': 'tiff',
            'bit_depth': 16,
            'filename': 'test.tiff'
        }
        img_data = ImageData(pixels=pixels, metadata=metadata)

        copied = img_data.copy()

        assert copied.shape == img_data.shape
        assert copied.dtype == img_data.dtype
        assert copied.width == img_data.width
        assert copied.height == img_data.height
        assert copied.channels == img_data.channels
        assert copied.format == img_data.format
        assert copied.metadata['bit_depth'] == img_data.metadata['bit_depth']


class TestImageDataRepr:
    """Tests for ImageData string representation"""

    def test_repr_rgb(self):
        """Test __repr__ for RGB image"""
        pixels = np.random.rand(10, 20, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        repr_str = repr(img_data)

        assert 'ImageData' in repr_str
        assert 'shape=(10, 20, 3)' in repr_str
        assert 'dtype=float32' in repr_str
        assert 'channels=3' in repr_str

    def test_repr_grayscale(self):
        """Test __repr__ for grayscale image"""
        pixels = np.random.rand(15, 25).astype(np.float64)
        img_data = ImageData(pixels=pixels)

        repr_str = repr(img_data)

        assert 'ImageData' in repr_str
        assert 'shape=(15, 25)' in repr_str
        assert 'dtype=float64' in repr_str
        assert 'channels=1' in repr_str


class TestImageDataEdgeCases:
    """Tests for edge cases and special scenarios"""

    def test_single_pixel_image(self):
        """Test with 1x1 image"""
        pixels = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        img_data = ImageData(pixels=pixels)

        assert img_data.shape == (1, 1, 3)
        assert img_data.width == 1
        assert img_data.height == 1
        assert img_data.channels == 3

    def test_very_large_dimensions(self):
        """Test with large image dimensions"""
        # Don't actually allocate large array, just test shape handling
        pixels = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        assert img_data.width == 2000
        assert img_data.height == 1000

    def test_different_dtypes(self):
        """Test with various numpy dtypes"""
        dtypes = [np.uint8, np.uint16, np.float32, np.float64]

        for dtype in dtypes:
            pixels = np.zeros((10, 10, 3), dtype=dtype)
            img_data = ImageData(pixels=pixels)

            assert img_data.dtype == dtype
            assert img_data.metadata['dtype'] == str(np.dtype(dtype))

    def test_metadata_without_format(self):
        """Test that missing format returns 'unknown'"""
        pixels = np.random.rand(10, 10, 3).astype(np.float32)
        img_data = ImageData(pixels=pixels)

        # Format not set, should default to 'unknown'
        assert img_data.format == 'unknown'
        assert img_data.metadata.get('format', 'unknown') == 'unknown'
