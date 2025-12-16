"""
Constants for image pipeline
"""
from .types import ColorSpace, TransferFunction


# Standard color primaries (CIE 1931 xy coordinates)
# Format: {'red': (x, y), 'green': (x, y), 'blue': (x, y), 'white': (x, y)}
STANDARD_COLOR_PRIMARIES = {
    ColorSpace.BT709: {
        'red': (0.64, 0.33),
        'green': (0.30, 0.60),
        'blue': (0.15, 0.06),
        'white': (0.3127, 0.3290)  # D65
    },
    ColorSpace.BT2020: {
        'red': (0.708, 0.292),
        'green': (0.170, 0.797),
        'blue': (0.131, 0.046),
        'white': (0.3127, 0.3290)  # D65
    },
    ColorSpace.DISPLAY_P3: {
        'red': (0.680, 0.320),
        'green': (0.265, 0.690),
        'blue': (0.150, 0.060),
        'white': (0.3127, 0.3290)  # D65
    }
}


# Default peak luminance values for different transfer functions (nits)
DEFAULT_PEAK_LUMINANCE = {
    TransferFunction.PQ: 10000.0,
    TransferFunction.HLG: 1000.0,
    TransferFunction.SRGB: 100.0,
    TransferFunction.LINEAR: None  # Not applicable
}


# Default minimum luminance (nits)
DEFAULT_MIN_LUMINANCE = 0.0001


# Mapping for cICP chunk generation
# Transfer function → cICP transfer_characteristics code
TRANSFER_TO_CICP = {
    TransferFunction.SRGB: 13,
    TransferFunction.PQ: 16,
    TransferFunction.HLG: 18,
    TransferFunction.LINEAR: 8,  # Linear transfer
}

# Color space → cICP color_primaries code
COLORSPACE_TO_CICP = {
    ColorSpace.BT709: 1,
    ColorSpace.BT2020: 9,
    ColorSpace.DISPLAY_P3: 12,
}

# Color space → cICP matrix_coefficients code
COLORSPACE_TO_MATRIX = {
    ColorSpace.BT709: 1,      # BT.709 matrix
    ColorSpace.BT2020: 9,     # BT.2020 non-constant luminance
    ColorSpace.DISPLAY_P3: 1, # Use BT.709 matrix for Display P3
}