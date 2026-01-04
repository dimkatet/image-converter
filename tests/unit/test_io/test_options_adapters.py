"""
Tests for format-specific SaveOptions adapters (PNG, TIFF, AVIF)

This module tests the validation and normalization logic for save options
across different image formats. Each adapter validates format-specific
parameters and provides sensible defaults.
"""
import warnings

import pytest

from image_pipeline.io.formats.png.options import PNGOptionsAdapter
from image_pipeline.io.formats.tiff.options import TiffOptionsAdapter
from image_pipeline.io.formats.avif.options import AVIFOptionsAdapter
from image_pipeline.types import SaveOptions


# ============================================================================
# PNG Options Adapter Tests
# ============================================================================

class TestPNGOptionsAdapter:
    """Tests for PNGOptionsAdapter validation"""

    def test_get_supported_options(self):
        """Test supported options list"""
        adapter = PNGOptionsAdapter()
        supported = adapter.get_supported_options()

        assert 'compression_level' in supported
        assert 'strategy' in supported
        assert 'icc_profile' in supported
        assert len(supported) == 3

    def test_default_options(self):
        """Test default PNG options when none provided"""
        adapter = PNGOptionsAdapter()
        validated = adapter.validate({})

        assert validated['compression_level'] == 6  # Default
        assert validated['strategy'] == 0  # Z_DEFAULT_STRATEGY
        assert 'icc_profile' not in validated

    def test_compression_level_valid_range(self):
        """Test valid compression_level values (0-9)"""
        adapter = PNGOptionsAdapter()

        for level in [0, 1, 5, 9]:
            options: SaveOptions = {'compression_level': level}
            validated = adapter.validate(options)
            assert validated['compression_level'] == level

    def test_compression_level_invalid_type(self):
        """Test compression_level with wrong type raises TypeError"""
        adapter = PNGOptionsAdapter()
        options: SaveOptions = {'compression_level': "5"}  # type: ignore

        with pytest.raises(TypeError, match="compression_level must be int"):
            adapter.validate(options)

    def test_compression_level_out_of_range_negative(self):
        """Test compression_level below 0 raises ValueError"""
        adapter = PNGOptionsAdapter()
        options: SaveOptions = {'compression_level': -1}

        with pytest.raises(ValueError, match="compression_level must be in range"):
            adapter.validate(options)

    def test_compression_level_out_of_range_high(self):
        """Test compression_level above 9 raises ValueError"""
        adapter = PNGOptionsAdapter()
        options: SaveOptions = {'compression_level': 10}

        with pytest.raises(ValueError, match="compression_level must be in range"):
            adapter.validate(options)

    def test_strategy_valid_range(self):
        """Test valid strategy values (0-4)"""
        adapter = PNGOptionsAdapter()

        for strategy in [0, 1, 2, 3, 4]:
            options: SaveOptions = {'strategy': strategy}
            validated = adapter.validate(options)
            assert validated['strategy'] == strategy

    def test_strategy_invalid_type(self):
        """Test strategy with wrong type raises TypeError"""
        adapter = PNGOptionsAdapter()
        options: SaveOptions = {'strategy': 2.5}  # type: ignore

        with pytest.raises(TypeError, match="strategy must be int"):
            adapter.validate(options)

    def test_strategy_out_of_range_negative(self):
        """Test strategy below 0 raises ValueError"""
        adapter = PNGOptionsAdapter()
        options: SaveOptions = {'strategy': -1}

        with pytest.raises(ValueError, match="strategy must be in range"):
            adapter.validate(options)

    def test_strategy_out_of_range_high(self):
        """Test strategy above 4 raises ValueError"""
        adapter = PNGOptionsAdapter()
        options: SaveOptions = {'strategy': 5}

        with pytest.raises(ValueError, match="strategy must be in range"):
            adapter.validate(options)

    def test_icc_profile_valid_path(self, tmp_path):
        """Test icc_profile with valid file path"""
        adapter = PNGOptionsAdapter()

        # Create dummy ICC profile file
        icc_file = tmp_path / "profile.icc"
        icc_file.write_bytes(b"ICC PROFILE DATA")

        options: SaveOptions = {'icc_profile': str(icc_file)}
        validated = adapter.validate(options)

        assert validated['icc_profile'] == str(icc_file)

    def test_icc_profile_invalid_type(self):
        """Test icc_profile with non-string type raises TypeError"""
        adapter = PNGOptionsAdapter()
        options: SaveOptions = {'icc_profile': 123}  # type: ignore

        with pytest.raises(TypeError, match="icc_profile must be str"):
            adapter.validate(options)

    def test_icc_profile_file_not_found(self):
        """Test icc_profile with non-existent file raises FileNotFoundError"""
        adapter = PNGOptionsAdapter()
        options: SaveOptions = {'icc_profile': '/nonexistent/profile.icc'}

        with pytest.raises(FileNotFoundError, match="ICC profile file not found"):
            adapter.validate(options)

    def test_icc_profile_path_is_directory(self, tmp_path):
        """Test icc_profile pointing to directory raises ValueError"""
        adapter = PNGOptionsAdapter()
        options: SaveOptions = {'icc_profile': str(tmp_path)}

        with pytest.raises(ValueError, match="ICC profile path is not a file"):
            adapter.validate(options)

    def test_combined_options(self, tmp_path):
        """Test multiple valid options together"""
        adapter = PNGOptionsAdapter()

        icc_file = tmp_path / "profile.icc"
        icc_file.write_bytes(b"ICC DATA")

        options: SaveOptions = {
            'compression_level': 9,
            'strategy': 3,
            'icc_profile': str(icc_file),
        }
        validated = adapter.validate(options)

        assert validated['compression_level'] == 9
        assert validated['strategy'] == 3
        assert validated['icc_profile'] == str(icc_file)

    def test_unsupported_options_warning(self):
        """Test that unsupported options trigger a warning"""
        adapter = PNGOptionsAdapter()
        options: SaveOptions = {
            'compression_level': 5,
            'quality': 95,  # Not supported by PNG adapter
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adapter.validate(options)

            # Should have warning about 'quality'
            assert len(w) >= 1
            assert 'quality' in str(w[0].message).lower()


# ============================================================================
# TIFF Options Adapter Tests
# ============================================================================

class TestTiffOptionsAdapter:
    """Tests for TiffOptionsAdapter validation"""

    def test_get_supported_options(self):
        """Test supported options list"""
        adapter = TiffOptionsAdapter()
        supported = adapter.get_supported_options()

        assert 'compression' in supported
        assert 'quality' in supported
        assert 'compression_level' in supported
        assert len(supported) == 3

    def test_default_options(self):
        """Test default TIFF options when none provided"""
        adapter = TiffOptionsAdapter()
        validated = adapter.validate({})

        assert validated['compression'] == 'none'
        assert 'quality' not in validated
        assert 'compression_level' not in validated

    def test_compression_valid_types(self):
        """Test all valid compression types"""
        adapter = TiffOptionsAdapter()
        valid_compressions = ['none', 'lzw', 'deflate', 'zstd', 'jpeg']

        for comp in valid_compressions:
            options: SaveOptions = {'compression': comp}
            validated = adapter.validate(options)
            assert validated['compression'] == comp

    def test_compression_none_value(self):
        """Test compression=None converts to 'none'"""
        adapter = TiffOptionsAdapter()
        options: SaveOptions = {'compression': None}  # type: ignore

        validated = adapter.validate(options)
        assert validated['compression'] == 'none'

    def test_compression_invalid_type(self):
        """Test compression with non-string type raises TypeError"""
        adapter = TiffOptionsAdapter()
        options: SaveOptions = {'compression': 123}  # type: ignore

        with pytest.raises(TypeError, match="compression must be str or None"):
            adapter.validate(options)

    def test_compression_invalid_value(self):
        """Test compression with invalid value raises ValueError"""
        adapter = TiffOptionsAdapter()
        options: SaveOptions = {'compression': 'gzip'}  # Not supported

        with pytest.raises(ValueError, match="compression must be one of"):
            adapter.validate(options)

    def test_quality_valid_range(self):
        """Test valid quality values (1-100)"""
        adapter = TiffOptionsAdapter()

        for quality in [1, 50, 100]:
            options: SaveOptions = {'compression': 'jpeg', 'quality': quality}
            validated = adapter.validate(options)
            assert validated['quality'] == quality

    def test_quality_invalid_type(self):
        """Test quality with wrong type raises TypeError"""
        adapter = TiffOptionsAdapter()
        options: SaveOptions = {'compression': 'jpeg', 'quality': "95"}  # type: ignore

        with pytest.raises(TypeError, match="quality must be int"):
            adapter.validate(options)

    def test_quality_out_of_range_low(self):
        """Test quality below 1 raises ValueError"""
        adapter = TiffOptionsAdapter()
        options: SaveOptions = {'compression': 'jpeg', 'quality': 0}

        with pytest.raises(ValueError, match="quality must be in range"):
            adapter.validate(options)

    def test_quality_out_of_range_high(self):
        """Test quality above 100 raises ValueError"""
        adapter = TiffOptionsAdapter()
        options: SaveOptions = {'compression': 'jpeg', 'quality': 101}

        with pytest.raises(ValueError, match="quality must be in range"):
            adapter.validate(options)

    def test_quality_with_non_jpeg_compression_warns(self):
        """Test quality with non-JPEG compression triggers warning"""
        adapter = TiffOptionsAdapter()
        options: SaveOptions = {'compression': 'lzw', 'quality': 95}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validated = adapter.validate(options)

            # Should warn about quality being ignored
            assert len(w) >= 1
            assert 'quality option is only used with compression' in str(w[0].message)
            assert validated['quality'] == 95  # Still stored

    def test_compression_level_valid_range(self):
        """Test valid compression_level values (0-9)"""
        adapter = TiffOptionsAdapter()

        for level in [0, 5, 9]:
            options: SaveOptions = {'compression': 'deflate', 'compression_level': level}
            validated = adapter.validate(options)
            assert validated['compression_level'] == level

    def test_compression_level_invalid_type(self):
        """Test compression_level with wrong type raises TypeError"""
        adapter = TiffOptionsAdapter()
        options: SaveOptions = {'compression': 'deflate', 'compression_level': 5.5}  # type: ignore

        with pytest.raises(TypeError, match="compression_level must be int"):
            adapter.validate(options)

    def test_compression_level_out_of_range_negative(self):
        """Test compression_level below 0 raises ValueError"""
        adapter = TiffOptionsAdapter()
        options: SaveOptions = {'compression': 'deflate', 'compression_level': -1}

        with pytest.raises(ValueError, match="compression_level must be in range"):
            adapter.validate(options)

    def test_compression_level_out_of_range_high(self):
        """Test compression_level above 9 raises ValueError"""
        adapter = TiffOptionsAdapter()
        options: SaveOptions = {'compression': 'deflate', 'compression_level': 10}

        with pytest.raises(ValueError, match="compression_level must be in range"):
            adapter.validate(options)

    def test_compression_level_with_wrong_compression_warns(self):
        """Test compression_level with non-deflate/zstd triggers warning"""
        adapter = TiffOptionsAdapter()
        options: SaveOptions = {'compression': 'lzw', 'compression_level': 5}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validated = adapter.validate(options)

            # Should warn about compression_level being ignored
            assert len(w) >= 1
            assert 'compression_level is only used with' in str(w[0].message)
            assert validated['compression_level'] == 5  # Still stored

    def test_combined_options_jpeg(self):
        """Test JPEG compression with quality"""
        adapter = TiffOptionsAdapter()
        options: SaveOptions = {
            'compression': 'jpeg',
            'quality': 90,
        }
        validated = adapter.validate(options)

        assert validated['compression'] == 'jpeg'
        assert validated['quality'] == 90

    def test_combined_options_deflate(self):
        """Test Deflate compression with compression_level"""
        adapter = TiffOptionsAdapter()
        options: SaveOptions = {
            'compression': 'deflate',
            'compression_level': 9,
        }
        validated = adapter.validate(options)

        assert validated['compression'] == 'deflate'
        assert validated['compression_level'] == 9

    def test_unsupported_options_warning(self):
        """Test that unsupported options trigger a warning"""
        adapter = TiffOptionsAdapter()
        options: SaveOptions = {
            'compression': 'lzw',
            'optimize': True,  # Not supported by TIFF
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adapter.validate(options)

            # Should have warning about 'optimize'
            assert len(w) >= 1
            assert 'optimize' in str(w[0].message).lower()


# ============================================================================
# AVIF Options Adapter Tests
# ============================================================================

class TestAVIFOptionsAdapter:
    """Tests for AVIFOptionsAdapter validation"""

    def test_get_supported_options(self):
        """Test supported options list"""
        adapter = AVIFOptionsAdapter()
        supported = adapter.get_supported_options()

        assert 'quality' in supported
        assert 'speed' in supported
        assert 'bit_depth' in supported
        assert 'numthreads' in supported
        assert len(supported) == 4

    def test_default_options(self):
        """Test default AVIF options when none provided"""
        adapter = AVIFOptionsAdapter()
        validated = adapter.validate({})

        assert validated['quality'] == 90  # Default
        assert validated['speed'] == 6  # Default (balanced)
        assert 'bitspersample' not in validated
        assert 'numthreads' not in validated

    def test_quality_valid_range(self):
        """Test valid quality values (0-100)"""
        adapter = AVIFOptionsAdapter()

        for quality in [0, 50, 90, 100]:
            options: SaveOptions = {'quality': quality}
            validated = adapter.validate(options)
            assert validated['quality'] == quality

    def test_quality_invalid_type(self):
        """Test quality with wrong type raises ValueError"""
        adapter = AVIFOptionsAdapter()
        options: SaveOptions = {'quality': "90"}  # type: ignore

        with pytest.raises(ValueError, match="quality must be 0-100"):
            adapter.validate(options)

    def test_quality_out_of_range_negative(self):
        """Test quality below 0 raises ValueError"""
        adapter = AVIFOptionsAdapter()
        options: SaveOptions = {'quality': -1}

        with pytest.raises(ValueError, match="quality must be 0-100"):
            adapter.validate(options)

    def test_quality_out_of_range_high(self):
        """Test quality above 100 raises ValueError"""
        adapter = AVIFOptionsAdapter()
        options: SaveOptions = {'quality': 101}

        with pytest.raises(ValueError, match="quality must be 0-100"):
            adapter.validate(options)

    def test_speed_valid_range(self):
        """Test valid speed values (0-10)"""
        adapter = AVIFOptionsAdapter()

        for speed in [0, 6, 10]:
            options: SaveOptions = {'speed': speed}
            validated = adapter.validate(options)
            assert validated['speed'] == speed

    def test_speed_invalid_type(self):
        """Test speed with wrong type raises ValueError"""
        adapter = AVIFOptionsAdapter()
        options: SaveOptions = {'speed': 5.5}  # type: ignore

        with pytest.raises(ValueError, match="speed must be 0-10"):
            adapter.validate(options)

    def test_speed_out_of_range_negative(self):
        """Test speed below 0 raises ValueError"""
        adapter = AVIFOptionsAdapter()
        options: SaveOptions = {'speed': -1}

        with pytest.raises(ValueError, match="speed must be 0-10"):
            adapter.validate(options)

    def test_speed_out_of_range_high(self):
        """Test speed above 10 raises ValueError"""
        adapter = AVIFOptionsAdapter()
        options: SaveOptions = {'speed': 11}

        with pytest.raises(ValueError, match="speed must be 0-10"):
            adapter.validate(options)

    def test_bit_depth_valid_values(self):
        """Test valid bit_depth values (8, 10, 12, 16)"""
        adapter = AVIFOptionsAdapter()

        for depth in [8, 10, 12, 16]:
            options: SaveOptions = {'bit_depth': depth}
            validated = adapter.validate(options)
            assert validated['bitspersample'] == depth

    def test_bit_depth_invalid_value(self):
        """Test bit_depth with invalid value raises ValueError"""
        adapter = AVIFOptionsAdapter()
        options: SaveOptions = {'bit_depth': 14}

        with pytest.raises(ValueError, match="bit_depth must be one of"):
            adapter.validate(options)

    def test_numthreads_valid_values(self):
        """Test valid numthreads values (>= 1)"""
        adapter = AVIFOptionsAdapter()

        for threads in [1, 4, 8, 16]:
            options: SaveOptions = {'numthreads': threads}
            validated = adapter.validate(options)
            assert validated['numthreads'] == threads

    def test_numthreads_invalid_type(self):
        """Test numthreads with wrong type raises ValueError"""
        adapter = AVIFOptionsAdapter()
        options: SaveOptions = {'numthreads': "4"}  # type: ignore

        with pytest.raises(ValueError, match="numthreads must be >= 1"):
            adapter.validate(options)

    def test_numthreads_zero(self):
        """Test numthreads=0 raises ValueError"""
        adapter = AVIFOptionsAdapter()
        options: SaveOptions = {'numthreads': 0}

        with pytest.raises(ValueError, match="numthreads must be >= 1"):
            adapter.validate(options)

    def test_numthreads_negative(self):
        """Test negative numthreads raises ValueError"""
        adapter = AVIFOptionsAdapter()
        options: SaveOptions = {'numthreads': -1}

        with pytest.raises(ValueError, match="numthreads must be >= 1"):
            adapter.validate(options)

    def test_combined_options(self):
        """Test multiple valid options together"""
        adapter = AVIFOptionsAdapter()
        options: SaveOptions = {
            'quality': 95,
            'speed': 4,
            'bit_depth': 12,
            'numthreads': 8,
        }
        validated = adapter.validate(options)

        assert validated['quality'] == 95
        assert validated['speed'] == 4
        assert validated['bitspersample'] == 12
        assert validated['numthreads'] == 8

    def test_unsupported_options_warning(self):
        """Test that unsupported options trigger a warning"""
        adapter = AVIFOptionsAdapter()
        options: SaveOptions = {
            'quality': 90,
            'compression': 'lzw',  # Not supported by AVIF
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adapter.validate(options)

            # Should have warning about 'compression'
            assert len(w) >= 1
            assert 'compression' in str(w[0].message).lower()
