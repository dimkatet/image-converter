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
        """
        self.method = method
        self.exposure = exposure
        self.white_point = white_point
        self.target_peak = target_peak
        self.key_value = key_value
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

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)

        # Validate dtype - should be float for HDR data
        self._check_dtype(pixels, [np.float32, np.float64])

        # Clip negative values (can occur from floating point errors or data artifacts)
        hdr = np.maximum(pixels, 0.0)

        # Apply exposure adjustment
        hdr = hdr * self.exposure

        # Compute log-average luminance (Reinhard Global approach)
        # This gives perceptually correct average that isn't dominated by bright pixels
        delta = 0.0001  # Small constant to avoid log(0)
        log_avg_luminance = np.exp(np.mean(np.log(delta + hdr)))

        # Scale by key value to map to perceptually appropriate range
        # key_value / log_avg_luminance gives the scaling factor
        # This makes log_avg_luminance map to key_value (typically 0.18 = middle gray)
        hdr_scaled = (self.key_value / log_avg_luminance) * hdr

        # Apply tone mapping operator
        # All operators output [0, 1] range
        if self.method == 'reinhard':
            sdr = self._reinhard(hdr_scaled)
        elif self.method == 'reinhard_extended':
            sdr = self._reinhard_extended(hdr_scaled)
        elif self.method == 'aces':
            sdr = self._aces_filmic(hdr_scaled)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Note: target_peak is only used for metadata, not pixel scaling
        # Output is in [0, 1] range, suitable for quantize filter

        return sdr.astype(np.float32)

    def _reinhard(self, x: np.ndarray) -> np.ndarray:
        """
        Reinhard Global operator with automatic white point.

        Uses the max luminance as white point for better handling of bright highlights.
        Formula: L_out = (L * (1 + L/L_white²)) / (1 + L)

        Input is expected to be scaled by log-average luminance.
        """
        # Use max luminance as white point (burns to white)
        # This handles bright highlights better than simple Reinhard
        L_white = np.max(x)
        if L_white < 1.0:
            L_white = 1.0  # Avoid division issues

        # Reinhard with white point
        numerator = x * (1.0 + x / (L_white * L_white))
        denominator = 1.0 + x
        return numerator / denominator

    def _reinhard_extended(self, x: np.ndarray) -> np.ndarray:
        """
        Extended Reinhard operator with white point:
        L_out = L_in * (1 + L_in/L_white²) / (1 + L_in)

        Values at L_white and above are mapped closer to 1.0 (white).
        Input is expected to be normalized HDR values.
        """
        L_white = self.white_point

        # Extended Reinhard formula
        numerator = x * (1.0 + x / (L_white * L_white))
        denominator = 1.0 + x
        return numerator / denominator

    def _aces_filmic(self, x: np.ndarray) -> np.ndarray:
        """
        ACES Filmic tone mapping curve (approximation).

        Provides a cinematic S-curve with nice shoulder and toe rolloff.
        Based on Narkowicz 2015 ACES approximation.
        Input is expected to be normalized HDR values.
        """
        # ACES coefficients (Narkowicz 2015 approximation)
        # Simplified Hill function form
        a = 2.51
        b = 0.03
        c = 2.43

        # Apply ACES curve: (x*(a*x+b))/(x*(a*x+b)+c)
        numerator = x * (a * x + b)
        denominator = x * (a * x + b) + c
        return np.clip(numerator / denominator, 0.0, 1.0)

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
