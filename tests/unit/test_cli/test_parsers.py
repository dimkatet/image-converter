"""
Tests for CLI parser modules

Tests the filter parser and options parser used by the CLI to parse
command-line arguments into filter objects and save options.
"""
import pytest

from image_pipeline.cli.filter_parser import parse_value, parse_filter, parse_filters
from image_pipeline.cli.options_parser import parse_options, _parse_value
from image_pipeline.cli.filter_registry import get_available_filters
from image_pipeline.cli.color_space_registry import get_available_color_spaces, COLOR_SPACE_ALIASES
from image_pipeline.types import ColorSpace
from image_pipeline.filters import (
    BlurFilter,
    QuantizeFilter,
    NormalizeFilter,
    RemoveAlphaFilter,
    ColorConvertFilter,
    PQEncodeFilter,
    GrayscaleFilter,
)


# ============================================================================
# Filter Parser - parse_value Tests
# ============================================================================

class TestFilterParseValue:
    """Tests for filter_parser.parse_value (type auto-detection)"""

    def test_parse_integer(self):
        """Test parsing integer values"""
        assert parse_value('5') == 5
        assert parse_value('0') == 0
        assert parse_value('100') == 100
        assert parse_value('-10') == -10

    def test_parse_float(self):
        """Test parsing float values"""
        assert parse_value('2.5') == 2.5
        assert parse_value('0.0') == 0.0
        assert parse_value('1.0') == 1.0
        assert parse_value('3.14159') == 3.14159

    def test_parse_color_space_bt709(self):
        """Test parsing BT.709 color space aliases"""
        assert parse_value('bt709') == ColorSpace.BT709
        assert parse_value('BT.709') == ColorSpace.BT709
        assert parse_value('sRGB') == ColorSpace.BT709
        assert parse_value('rec709') == ColorSpace.BT709

    def test_parse_color_space_bt2020(self):
        """Test parsing BT.2020 color space aliases"""
        assert parse_value('bt2020') == ColorSpace.BT2020
        assert parse_value('BT.2020') == ColorSpace.BT2020
        assert parse_value('rec2020') == ColorSpace.BT2020

    def test_parse_color_space_displayp3(self):
        """Test parsing Display P3 color space aliases"""
        assert parse_value('p3') == ColorSpace.DISPLAY_P3
        assert parse_value('displayp3') == ColorSpace.DISPLAY_P3
        assert parse_value('Display-P3') == ColorSpace.DISPLAY_P3

    def test_parse_string(self):
        """Test parsing values that remain as strings"""
        assert parse_value('hello') == 'hello'
        assert parse_value('luminosity') == 'luminosity'
        assert parse_value('average') == 'average'

    def test_parse_empty_string(self):
        """Test parsing empty string"""
        assert parse_value('') == ''


# ============================================================================
# Filter Parser - parse_filter Tests
# ============================================================================

class TestFilterParseFilter:
    """Tests for filter_parser.parse_filter (single filter)"""

    def test_parse_filter_without_params(self):
        """Test parsing filter without parameters"""
        filter_obj = parse_filter('remove_alpha')
        assert isinstance(filter_obj, RemoveAlphaFilter)

    def test_parse_filter_with_single_param(self):
        """Test parsing filter with single parameter"""
        filter_obj = parse_filter('blur:sigma=2.5')
        assert isinstance(filter_obj, BlurFilter)
        assert filter_obj.sigma == 2.5

    def test_parse_filter_with_multiple_params(self):
        """Test parsing filter with multiple parameters"""
        filter_obj = parse_filter('normalize:min_val=0.0,max_val=1.0')
        assert isinstance(filter_obj, NormalizeFilter)
        assert filter_obj.min_val == 0.0
        assert filter_obj.max_val == 1.0

    def test_parse_filter_quantize_integer_param(self):
        """Test parsing quantize filter with integer parameter"""
        filter_obj = parse_filter('quantize:bit_depth=16')
        assert isinstance(filter_obj, QuantizeFilter)
        assert filter_obj.bit_depth == 16

    def test_parse_filter_color_convert_enum_params(self):
        """Test parsing color_convert filter with ColorSpace enums"""
        filter_obj = parse_filter('color_convert:source=bt709,target=bt2020')
        assert isinstance(filter_obj, ColorConvertFilter)
        assert filter_obj.source == ColorSpace.BT709
        assert filter_obj.target == ColorSpace.BT2020

    def test_parse_filter_pq_encode_with_param(self):
        """Test parsing pq_encode filter with reference_peak parameter"""
        filter_obj = parse_filter('pq_encode:reference_peak=10000')
        assert isinstance(filter_obj, PQEncodeFilter)
        assert filter_obj.reference_peak == 10000

    def test_parse_filter_grayscale_with_method(self):
        """Test parsing grayscale filter with method parameter"""
        filter_obj = parse_filter('grayscale:method=luminosity')
        assert isinstance(filter_obj, GrayscaleFilter)
        assert filter_obj.method == 'luminosity'

    def test_parse_filter_unknown_raises_error(self):
        """Test parsing unknown filter raises ValueError"""
        with pytest.raises(ValueError, match="Unknown filter: 'invalid_filter'"):
            parse_filter('invalid_filter')

    def test_parse_filter_unknown_shows_available(self):
        """Test error message shows available filters"""
        with pytest.raises(ValueError, match="Available filters:"):
            parse_filter('nonexistent')

    def test_parse_filter_invalid_param_format(self):
        """Test invalid parameter format raises ValueError"""
        with pytest.raises(ValueError, match="Invalid parameter format"):
            parse_filter('blur:invalid_param')

    def test_parse_filter_missing_equals(self):
        """Test parameter without equals sign raises ValueError"""
        with pytest.raises(ValueError, match="Expected 'key=value'"):
            parse_filter('blur:sigma')

    def test_parse_filter_wrong_param_name(self):
        """Test wrong parameter name raises ValueError"""
        with pytest.raises(ValueError, match="Error creating filter"):
            parse_filter('blur:invalid_param=5')

    def test_parse_filter_with_whitespace(self):
        """Test parsing filter with whitespace"""
        filter_obj = parse_filter('blur : sigma = 2.5')
        assert isinstance(filter_obj, BlurFilter)
        assert filter_obj.sigma == 2.5

    def test_parse_filter_with_extra_commas(self):
        """Test parsing filter handles whitespace around commas"""
        filter_obj = parse_filter('normalize: min_val=0.0 , max_val=1.0 ')
        assert isinstance(filter_obj, NormalizeFilter)
        assert filter_obj.min_val == 0.0
        assert filter_obj.max_val == 1.0


# ============================================================================
# Filter Parser - parse_filters Tests
# ============================================================================

class TestFilterParseFilters:
    """Tests for filter_parser.parse_filters (multiple filters)"""

    def test_parse_filters_empty_list(self):
        """Test parsing empty filter list"""
        filters = parse_filters([])
        assert filters == []

    def test_parse_filters_single(self):
        """Test parsing single filter in list"""
        filters = parse_filters(['blur:sigma=2.5'])
        assert len(filters) == 1
        assert isinstance(filters[0], BlurFilter)

    def test_parse_filters_multiple(self):
        """Test parsing multiple filters"""
        filter_strings = [
            'remove_alpha',
            'blur:sigma=2.5',
            'quantize:bit_depth=16'
        ]
        filters = parse_filters(filter_strings)

        assert len(filters) == 3
        assert isinstance(filters[0], RemoveAlphaFilter)
        assert isinstance(filters[1], BlurFilter)
        assert isinstance(filters[2], QuantizeFilter)

    def test_parse_filters_error_shows_position(self):
        """Test error message shows which filter failed"""
        filter_strings = [
            'blur:sigma=2.5',
            'invalid_filter',
            'quantize:bit_depth=16'
        ]

        with pytest.raises(ValueError, match="Error in filter #2"):
            parse_filters(filter_strings)

    def test_parse_filters_complex_pipeline(self):
        """Test parsing complex filter pipeline"""
        filter_strings = [
            'remove_alpha',
            'color_convert:source=bt709,target=bt2020',
            'normalize:min_val=0.0,max_val=1.0',
            'pq_encode:reference_peak=10000',
            'quantize:bit_depth=12'
        ]
        filters = parse_filters(filter_strings)

        assert len(filters) == 5
        assert isinstance(filters[0], RemoveAlphaFilter)
        assert isinstance(filters[1], ColorConvertFilter)
        assert isinstance(filters[2], NormalizeFilter)
        assert isinstance(filters[3], PQEncodeFilter)
        assert isinstance(filters[4], QuantizeFilter)


# ============================================================================
# Options Parser - _parse_value Tests
# ============================================================================

class TestOptionsParseValue:
    """Tests for options_parser._parse_value (type auto-detection)"""

    def test_parse_true_variants(self):
        """Test parsing boolean true values"""
        assert _parse_value('true') is True
        assert _parse_value('True') is True
        assert _parse_value('TRUE') is True
        assert _parse_value('yes') is True
        assert _parse_value('1') is True
        assert _parse_value('on') is True

    def test_parse_false_variants(self):
        """Test parsing boolean false values"""
        assert _parse_value('false') is False
        assert _parse_value('False') is False
        assert _parse_value('FALSE') is False
        assert _parse_value('no') is False
        assert _parse_value('0') is False
        assert _parse_value('off') is False

    def test_parse_integer(self):
        """Test parsing integer values"""
        assert _parse_value('5') == 5
        assert _parse_value('100') == 100
        assert _parse_value('-10') == -10

    def test_parse_float(self):
        """Test parsing float values"""
        assert _parse_value('2.5') == 2.5
        assert _parse_value('3.14') == 3.14
        assert _parse_value('0.0') == 0.0

    def test_parse_string(self):
        """Test parsing string values"""
        assert _parse_value('hello') == 'hello'
        assert _parse_value('lzw') == 'lzw'
        assert _parse_value('deflate') == 'deflate'


# ============================================================================
# Options Parser - parse_options Tests
# ============================================================================

class TestOptionsParser:
    """Tests for options_parser.parse_options"""

    def test_parse_options_empty_list(self):
        """Test parsing empty options list"""
        options = parse_options([])
        assert options == {}

    def test_parse_options_single_option(self):
        """Test parsing single option"""
        options = parse_options(['quality=90'])
        assert options == {'quality': 90}

    def test_parse_options_multiple_options(self):
        """Test parsing multiple options"""
        options = parse_options(['quality=90', 'lossless=true'])
        assert options == {'quality': 90, 'lossless': True}

    def test_parse_options_integer_values(self):
        """Test parsing integer option values"""
        options = parse_options(['compression_level=9', 'strategy=3'])
        assert options == {'compression_level': 9, 'strategy': 3}

    def test_parse_options_float_values(self):
        """Test parsing float option values"""
        options = parse_options(['speed=2.5'])
        assert options == {'speed': 2.5}

    def test_parse_options_boolean_values(self):
        """Test parsing boolean option values"""
        options = parse_options(['lossless=true', 'optimize=false'])
        assert options == {'lossless': True, 'optimize': False}

    def test_parse_options_string_values(self):
        """Test parsing string option values"""
        options = parse_options(['compression=lzw'])
        assert options == {'compression': 'lzw'}

    def test_parse_options_mixed_types(self):
        """Test parsing mixed option types"""
        options = parse_options([
            'quality=95',
            'lossless=false',
            'compression=deflate',
            'speed=2.5'
        ])
        assert options == {
            'quality': 95,
            'lossless': False,
            'compression': 'deflate',
            'speed': 2.5
        }

    def test_parse_options_missing_equals(self):
        """Test option without equals sign raises ValueError"""
        with pytest.raises(ValueError, match="Invalid option format"):
            parse_options(['quality90'])

    def test_parse_options_empty_key(self):
        """Test option with empty key raises ValueError"""
        with pytest.raises(ValueError, match="Empty key"):
            parse_options(['=90'])

    def test_parse_options_empty_value(self):
        """Test option with empty value raises ValueError"""
        with pytest.raises(ValueError, match="Empty value"):
            parse_options(['quality='])

    def test_parse_options_with_whitespace(self):
        """Test parsing options with whitespace"""
        options = parse_options(['quality = 90', ' lossless = true '])
        assert options == {'quality': 90, 'lossless': True}

    def test_parse_options_multiple_equals(self):
        """Test option with multiple equals signs (splits on first)"""
        options = parse_options(['key=value=extra'])
        assert options == {'key': 'value=extra'}


# ============================================================================
# Registry Tests
# ============================================================================

class TestFilterRegistry:
    """Tests for filter_registry"""

    def test_get_available_filters(self):
        """Test get_available_filters returns formatted string"""
        result = get_available_filters()

        assert 'Available filters:' in result
        assert 'blur' in result
        assert 'quantize' in result
        assert 'remove_alpha' in result
        assert 'BlurFilter' in result

    def test_get_available_filters_sorted(self):
        """Test available filters are sorted alphabetically"""
        result = get_available_filters()
        lines = result.split('\n')[1:]  # Skip header

        # Extract filter names
        filter_names = [line.split('•')[1].split('->')[0].strip() for line in lines if '•' in line]

        # Check sorted
        assert filter_names == sorted(filter_names)


class TestColorSpaceRegistry:
    """Tests for color_space_registry"""

    def test_get_available_color_spaces(self):
        """Test get_available_color_spaces returns formatted string"""
        result = get_available_color_spaces()

        assert 'Available color space aliases:' in result
        assert 'bt' in result.lower()
        assert 'p3' in result.lower()

    def test_color_space_aliases_lowercase(self):
        """Test all color space aliases are lowercase"""
        for alias in COLOR_SPACE_ALIASES.keys():
            assert alias == alias.lower(), f"Alias '{alias}' is not lowercase"

    def test_color_space_bt709_aliases(self):
        """Test BT.709 has multiple aliases"""
        bt709_aliases = [alias for alias, cs in COLOR_SPACE_ALIASES.items() if cs == ColorSpace.BT709]
        assert len(bt709_aliases) >= 3  # At least: bt709, srgb, rec709

    def test_color_space_bt2020_aliases(self):
        """Test BT.2020 has multiple aliases"""
        bt2020_aliases = [alias for alias, cs in COLOR_SPACE_ALIASES.items() if cs == ColorSpace.BT2020]
        assert len(bt2020_aliases) >= 2  # At least: bt2020, rec2020

    def test_color_space_displayp3_aliases(self):
        """Test Display P3 has multiple aliases"""
        p3_aliases = [alias for alias, cs in COLOR_SPACE_ALIASES.items() if cs == ColorSpace.DISPLAY_P3]
        assert len(p3_aliases) >= 3  # At least: p3, displayp3, display-p3
