# Tests

This directory contains the test suite for the Image Pipeline project.

## Running Tests

### Setup

First, install test dependencies:

```bash
source venv/bin/activate
pip install -e ".[dev]"
```

### Run all tests

```bash
# From project root
pytest

# Or with venv python directly
./venv/bin/pytest
```

### Run specific test files or categories

```bash
# Run only filter tests
pytest tests/unit/test_filters/

# Run specific test file
pytest tests/unit/test_filters/test_pq_encode_decode.py

# Run specific test class or function
pytest tests/unit/test_filters/test_pq_encode_decode.py::TestPQEncodeFilter
pytest tests/unit/test_filters/test_pq_encode_decode.py::TestPQRoundTrip::test_roundtrip_identity
```

### Useful options

```bash
# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Show test coverage (when available)
pytest --cov=src/image_pipeline --cov-report=html
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── unit/                    # Unit tests (isolated components)
│   ├── test_filters/        # Filter tests (PQ, quantize, etc.)
│   ├── test_core/           # Core components (ImageData, pipeline)
│   ├── test_io/             # I/O readers/writers
│   └── test_metadata/       # Metadata handling
└── integration/             # End-to-end workflow tests
```

## Writing Tests

### Test Organization

- Each test file should test a single module or related functionality
- Use test classes to group related tests
- Name tests descriptively: `test_<what>_<expected_behavior>`

### Example Test Structure

```python
class TestMyFilter:
    """Tests for MyFilter"""

    def test_basic_operation(self):
        """Test the main filter operation"""
        # Arrange
        filter = MyFilter(param=1.0)
        pixels = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)

        # Act
        result = filter.apply(pixels)

        # Assert
        assert result.shape == pixels.shape
        assert np.allclose(result, expected_value)

    def test_invalid_input_raises(self):
        """Test that invalid input raises appropriate error"""
        filter = MyFilter()

        with pytest.raises(ValueError, match="expected error message"):
            filter.apply(invalid_input)
```

### Using Fixtures

Shared test data is available via fixtures in `conftest.py`:

```python
def test_with_fixture(sample_linear_image):
    """Example using a fixture"""
    filter = MyFilter()
    result = filter.apply(sample_linear_image)
    assert result.shape == sample_linear_image.shape
```

### Testing Filters

All filter tests should cover:

1. **Basic operation** - correct output for known inputs
2. **Round-trip** - `decode(encode(x)) ≈ x` where applicable
3. **Parameter validation** - invalid parameters raise errors
4. **Edge cases** - boundary values, empty arrays, special shapes
5. **Metadata updates** - correct metadata changes

### Assertions

```python
# Floating point comparisons - use np.allclose with appropriate tolerance
assert np.allclose(result, expected, atol=0.01)  # Absolute tolerance
assert np.allclose(result, expected, rtol=0.001) # Relative tolerance

# Exceptions
with pytest.raises(ValueError, match="pattern in error message"):
    function_that_should_raise()

# Warnings
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    function_that_warns()
    assert len(w) == 1
    assert "warning text" in str(w[0].message)
```

## Test Principles

1. **Tests should be fast** - Use small synthetic arrays, not large real images
2. **Tests should be deterministic** - Use `np.random.seed()` for reproducibility
3. **Tests should be isolated** - No dependencies between tests
4. **One assertion per concept** - Split complex tests into multiple simple ones
5. **Test behavior, not implementation** - Focus on public API

## Current Coverage

Focus areas (Phase 1):
- ✅ PQ encode/decode filters
- 🚧 Quantize/Dequantize filters (TODO)
- 🚧 Color conversion filters (TODO)
- 🚧 Absolute/Relative luminance filters (TODO)
- 🚧 Core components (ImageData, FilterPipeline) (TODO)
