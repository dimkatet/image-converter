"""
Pytest configuration and shared fixtures

This file contains fixtures that are available to all test modules.
"""

import numpy as np
import pytest

from image_pipeline.core.image_data import ImageData
from image_pipeline.types import ImageMetadata, TransferFunction, ColorSpace


@pytest.fixture
def sample_linear_image():
    """
    8x8 RGB linear image with random values in [0, 1] range
    Typical for testing basic filter operations
    """
    np.random.seed(42)  # Reproducible
    return np.random.rand(8, 8, 3).astype(np.float32)


@pytest.fixture
def sample_hdr_image():
    """
    8x8 RGB HDR image with values in [0, 5000] nits range
    Suitable for testing HDR operations (PQ, tone mapping, etc.)
    """
    np.random.seed(42)
    return (np.random.rand(8, 8, 3).astype(np.float32) * 5000.0)


@pytest.fixture
def sample_grayscale_image():
    """
    8x8 single-channel grayscale image
    For testing filters that should work with grayscale
    """
    np.random.seed(42)
    return np.random.rand(8, 8, 1).astype(np.float32)


@pytest.fixture
def sample_metadata():
    """
    Typical ImageMetadata dictionary with common fields
    """
    metadata: ImageMetadata = {
        'shape': (8, 8, 3),
        'dtype': 'float32',
        'channels': 3,
        'bit_depth': 32,
        'format': 'TIFF',
        'transfer_function': TransferFunction.LINEAR,
        'color_space': ColorSpace.BT709,
        'paper_white': 100.0,
    }
    return metadata


@pytest.fixture
def sample_image_data(sample_linear_image, sample_metadata):
    """
    Complete ImageData object with pixels and metadata
    """
    return ImageData(sample_linear_image, metadata=sample_metadata.copy())


@pytest.fixture
def known_values_pq():
    """
    Known input/output pairs for PQ encoding (from ST.2084 specification)
    Useful for testing mathematical correctness

    Returns:
        List of (input_nits, expected_pq_value) tuples
    """
    # Exact reference value from ST.2084 / ITU-R BT.2100:
    # 100 nits = 0.5081 (SDR white on PQ curve)
    # Tolerance of ~0.001 should be used when comparing
    return [
        (0.0, 0.0),           # Black
        (100.0, 0.5081),      # SDR white (exact from spec)
        (10000.0, 1.0),       # Reference peak
    ]


@pytest.fixture
def small_test_image():
    """
    Minimal 2x2x3 RGB image for quick tests
    Contains known values for easy verification
    """
    return np.array([
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ], dtype=np.float32)


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Temporary directory for test outputs (images, files)
    Automatically cleaned up after tests
    """
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir
