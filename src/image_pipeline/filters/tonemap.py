from image_pipeline.core.image_data import ImageData

import numpy as np
from .base import ImageFilter


class ToneMappingFilter(ImageFilter):
    """
    Tone mapping filter: compress display-referred HDR to display-referred SDR.

    Converts high dynamic range linear pixel values (in nits) to a lower
    dynamic range suitable for standard displays, preserving detail and
    preventing clipping.

    Supported methods:
    - 'reinhard': Simple Reinhard operator (L_out = L / (1 + L))
    - 'reinhard_extended': Reinhard with white point control
    - 'aces': ACES Filmic curve (cinematic look)

    Important:
    - Input must be display-referred linear HDR (absolute nits)
    - Output is display-referred linear SDR (typically 0-100 nits)
    - Apply before quantization or gamma encoding

    Example workflow:
        absolute_luminance (scene → display) → tonemap (HDR → SDR) → quantize
    """

    VALID_METHODS = ['reinhard', 'reinhard_extended', 'aces']

    def __init__(
        self,
        method: str = 'reinhard',
        exposure: float = 1.0,
        white_point: float = 1.0,
        target_peak: float = 100.0,
        key_value: float = 0.18,
        min_exp: float = 0.1,
        max_exp: float = 10.0,
    ):
        """
        Args:
            method: Tone mapping operator to use
            exposure: Exposure adjustment multiplier (applied before tone mapping)
            white_point: White point for reinhard_extended (values above this map to white)
            target_peak: Target peak luminance in nits (for metadata only)
            key_value: Key value for log-average luminance scaling (default: 0.18)
                      Controls overall brightness: lower = darker, higher = brighter
                      Typical range: 0.09 (low key) to 0.36 (high key)
            min_exp: Minimum exposure multiplier to prevent over-darkening (default: 0.1)
            max_exp: Maximum exposure multiplier to prevent over-brightening (default: 10.0)
        """
        self.method = method
        self.exposure = exposure
        self.white_point = white_point
        self.target_peak = target_peak
        self.key_value = key_value
        self.min_exp = min_exp
        self.max_exp = max_exp
        super().__init__()

    def validate_params(self) -> None:
        if not isinstance(self.method, str):
            raise TypeError(
                f"method must be str, got {type(self.method).__name__}"
            )

        if self.method not in self.VALID_METHODS:
            raise ValueError(
                f"method must be one of {self.VALID_METHODS}, got '{self.method}'"
            )

        # Validate numeric parameters
        if not isinstance(self.exposure, (int, float)):
            raise TypeError(
                f"exposure must be numeric, got {type(self.exposure).__name__}"
            )
        if self.exposure <= 0:
            raise ValueError(f"exposure must be positive, got {self.exposure}")

        if not isinstance(self.white_point, (int, float)):
            raise TypeError(
                f"white_point must be numeric, got {type(self.white_point).__name__}"
            )
        if self.white_point <= 0:
            raise ValueError(f"white_point must be positive, got {self.white_point}")

        if not isinstance(self.target_peak, (int, float)):
            raise TypeError(
                f"target_peak must be numeric, got {type(self.target_peak).__name__}"
            )
        if self.target_peak <= 0:
            raise ValueError(f"target_peak must be positive, got {self.target_peak}")

        if not isinstance(self.key_value, (int, float)):
            raise TypeError(
                f"key_value must be numeric, got {type(self.key_value).__name__}"
            )
        if self.key_value <= 0:
            raise ValueError(f"key_value must be positive, got {self.key_value}")

        if not isinstance(self.min_exp, (int, float)):
            raise TypeError(
                f"min_exp must be numeric, got {type(self.min_exp).__name__}"
            )
        if self.min_exp <= 0:
            raise ValueError(f"min_exp must be positive, got {self.min_exp}")

        if not isinstance(self.max_exp, (int, float)):
            raise TypeError(
                f"max_exp must be numeric, got {type(self.max_exp).__name__}"
            )
        if self.max_exp <= 0:
            raise ValueError(f"max_exp must be positive, got {self.max_exp}")

        if self.min_exp >= self.max_exp:
            raise ValueError(
                f"min_exp must be less than max_exp, got min_exp={self.min_exp}, max_exp={self.max_exp}"
            )

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)

        # Validate dtype - should be float for HDR data
        self._check_dtype(pixels, [np.float16, np.float32, np.float64])

        # Clip negative values (can occur from floating point errors or data artifacts)
        hdr = np.maximum(pixels, 0.0)

        # Apply exposure adjustment
        hdr = hdr * self.exposure

        # Apply tone mapping operator
        # All operators output [0, 1] range
        if self.method == 'reinhard':
            sdr = self._reinhard(hdr)
        elif self.method == 'reinhard_extended':
            sdr = self._reinhard_extended(hdr)
        elif self.method == 'aces':
            sdr = self._aces_filmic(hdr)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Note: target_peak is only used for metadata, not pixel scaling
        # Output is in [0, 1] range, suitable for quantize filter

        return sdr.astype(np.float32)

    def _reinhard(self, hdr: np.ndarray) -> np.ndarray:
        """
        Reinhard Global operator with luminance-based tone mapping.

        Applies tone mapping to luminance channel only, then reapplies color
        to preserve chromaticity. This prevents hue shifts and color distortion.

        Steps:
        1. Compute luminance (Y) from RGB
        2. Calculate log-average luminance for exposure scaling
        3. Scale luminance by key_value
        4. Compute auto white point from max scaled luminance
        5. Apply Reinhard formula to luminance: L_out = (L * (1 + L/L_white²)) / (1 + L)
        6. Reconstruct RGB preserving original chromaticity

        Args:
            hdr: Linear RGB HDR image (already exposure-adjusted)

        Returns:
            Linear RGB SDR image in [0, 1] range
        """
        delta = 1e-6  # Small constant to avoid log(0) and division by zero

        # Compute luminance (Rec.709 coefficients for linear RGB)
        if hdr.ndim == 3 and hdr.shape[-1] == 3:
            # Y = 0.2126*R + 0.7152*G + 0.0722*B 
            # TODO: Support other color spaces later
            luminance_weights = np.array([0.2126, 0.7152, 0.0722], dtype=hdr.dtype)
            L = np.dot(hdr, luminance_weights)
        else:
            # Grayscale or single channel
            L = hdr

        # Ensure non-negative luminance
        L = np.maximum(L, 0.0)

        # Compute log-average luminance (geometric mean)
        # This gives perceptually correct average brightness
        log_avg_luminance = np.exp(np.mean(np.log(delta + L)))

        # Scale luminance by key value
        # This maps the log-average luminance to key_value (typically 0.18 = middle gray)
        # Clip exposure to prevent extreme scaling from very dark/bright images
        exposure = self.key_value / log_avg_luminance
        exposure = np.clip(exposure, self.min_exp, self.max_exp)
        L_scaled = exposure * L

        # Compute automatic white point from ORIGINAL luminance (before scaling)
        # This represents the input dynamic range, not the exposure-adjusted range
        L_white = np.percentile(L, 99.9)
        L_white = max(L_white, 1.0)  # Ensure at least 1.0 to avoid issues

        # Apply Reinhard formula with white point to luminance
        # Formula: L_out = (L * (1 + L/L_white²)) / (1 + L)
        L_white_sq = L_white * L_white
        L_mapped = (L_scaled * (1.0 + L_scaled / L_white_sq)) / (1.0 + L_scaled)

        # Reconstruct RGB preserving chromaticity (color ratios)
        # Scale each channel by the ratio of mapped to original luminance
        if hdr.ndim == 3 and hdr.shape[-1] == 3:
            # Avoid division by zero
            scale_factor = L_mapped / (L + delta)
            # Expand scale_factor to match RGB shape: (H, W) -> (H, W, 1)
            scale_factor = scale_factor[..., np.newaxis]
            rgb_out = hdr * scale_factor
        else:
            rgb_out = L_mapped

        return rgb_out

    def _reinhard_extended(self, hdr: np.ndarray) -> np.ndarray:
        """
        Extended Reinhard operator with manual white point control.

        Similar to standard Reinhard but uses user-specified white point
        instead of auto-computed one.

        Args:
            hdr: Linear RGB HDR image (already exposure-adjusted)

        Returns:
            Linear RGB SDR image in [0, 1] range
        """
        delta = 1e-6

        # Compute luminance
        if hdr.ndim == 3 and hdr.shape[-1] == 3:
            luminance_weights = np.array([0.2126, 0.7152, 0.0722], dtype=hdr.dtype)
            L = np.dot(hdr, luminance_weights)
        else:
            L = hdr

        L = np.maximum(L, 0.0)

        # Compute log-average luminance and scale with exposure clipping
        log_avg_luminance = np.exp(np.mean(np.log(delta + L)))
        exposure = self.key_value / log_avg_luminance
        exposure = np.clip(exposure, self.min_exp, self.max_exp)
        L_scaled = exposure * L

        # Use user-specified white point
        L_white = self.white_point
        L_white_sq = L_white * L_white

        # Apply Reinhard formula to luminance
        L_mapped = (L_scaled * (1.0 + L_scaled / L_white_sq)) / (1.0 + L_scaled)

        # Reconstruct RGB preserving chromaticity
        if hdr.ndim == 3 and hdr.shape[-1] == 3:
            scale_factor = L_mapped / (L + delta)
            scale_factor = scale_factor[..., np.newaxis]
            rgb_out = hdr * scale_factor
        else:
            rgb_out = L_mapped

        return rgb_out

    def _aces_filmic(self, hdr: np.ndarray) -> np.ndarray:
        """
        ACES Filmic tone mapping curve (approximation).

        Provides a cinematic S-curve with nice shoulder and toe rolloff.
        Based on Narkowicz 2015 ACES approximation.

        ACES applies the curve to RGB channels directly (not luminance-based)
        but we still use key_value for exposure scaling via luminance.

        Args:
            hdr: Linear RGB HDR image (already exposure-adjusted)

        Returns:
            Linear RGB SDR image in [0, 1] range
        """
        delta = 1e-6

        # Compute luminance for exposure scaling
        if hdr.ndim == 3 and hdr.shape[-1] == 3:
            luminance_weights = np.array([0.2126, 0.7152, 0.0722], dtype=hdr.dtype)
            L = np.dot(hdr, luminance_weights)
        else:
            L = hdr

        L = np.maximum(L, 0.0)

        # Compute log-average luminance and scale with exposure clipping
        log_avg_luminance = np.exp(np.mean(np.log(delta + L)))
        exposure_scale = self.key_value / log_avg_luminance
        exposure_scale = np.clip(exposure_scale, self.min_exp, self.max_exp)

        # Apply exposure scaling to RGB
        hdr_scaled = hdr * exposure_scale

        # ACES coefficients (Narkowicz 2015 approximation)
        a = 2.51
        b = 0.03
        c = 2.43
        d = 2.43
        e = 0.59

        # Apply ACES curve to each RGB channel
        # Formula: (x*(a*x+b))/(x*(c*x+d)+e)
        numerator = hdr_scaled * (a * hdr_scaled + b)
        denominator = hdr_scaled * (c * hdr_scaled + d) + e

        result = numerator / denominator
        return np.clip(result, 0.0, 1.0)

    def update_metadata(self, img_data: ImageData) -> None:
        """Update metadata after tone mapping."""
        super().update_metadata(img_data)

        # Pixels are now in [0, 1] range, scale by target_peak for metadata
        # MaxCLL and MaxFALL should reflect the target display luminance
        if img_data.pixels.size > 0:
            max_cll_value = float(np.max(img_data.pixels)) * self.target_peak
            max_fall_value = float(np.mean(img_data.pixels)) * self.target_peak
            img_data.metadata['max_cll'] = int(round(max_cll_value))
            img_data.metadata['max_fall'] = int(round(max_fall_value))

        # Remove paper_white (no longer scene-referred)
        if 'paper_white' in img_data.metadata:
            del img_data.metadata['paper_white']

    def __repr__(self) -> str:
        return (
            f"ToneMappingFilter(method='{self.method}', "
            f"exposure={self.exposure}, key_value={self.key_value})"
        )
