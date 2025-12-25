"""
JPEG Ultra HDR format writer using imagecodecs.libultrahdr
"""

import numpy as np
import imagecodecs

from image_pipeline.core.image_data import ImageData
from image_pipeline.types import TransferFunction, ColorSpace, SaveOptions
from image_pipeline.io.formats.base import FormatWriter


class UltraHDRWriter(FormatWriter):
    """
    Writer for JPEG Ultra HDR format

    Input requirements:
        - dtype: float32 or float16
        - channels: 3 (RGB) or 4 (RGBA)
        - transfer_function: LINEAR (REQUIRED - libultrahdr does PQ encoding internally)
        - color_space: BT709, BT2020, or DISPLAY_P3 (metadata required)
        - Data: Scene-referred (relative values 0.0-1.0+), NOT display-referred

    The writer will:
        1. Validate metadata (color_space, transfer_function must be LINEAR)
        2. Convert float32 → float16 if needed
        3. Add alpha channel if missing
        4. Encode using imagecodecs.ultrahdr_encode()

    IMPORTANT:
        - Do NOT use AbsoluteLuminanceFilter before this writer
        - Data should be scene-referred (relative values, where 1.0 = reference white)
        - libultrahdr makes internal assumptions about reference white (~100-203 nits SDR)
        - Using AbsoluteLuminanceFilter may conflict with library assumptions → incorrect brightness

    Technical note:
        - PQ encoding requires absolute nits, but libultrahdr API has no 'nits' parameter
        - Library assumes 1.0 = SDR white and applies tone mapping + PQ internally
        - Passing pre-scaled display-referred data causes mismatch with library's reference
        - Better to let libultrahdr apply its own assumptions consistently
    """

    def validate(self, img_data: ImageData) -> None:
        """
        Validate that image data is compatible with Ultra HDR encoding

        Args:
            img_data: ImageData to validate

        Raises:
            ValueError: If data is not compatible
        """
        pixels = img_data.pixels
        metadata = img_data.metadata

        # Check dtype
        if pixels.dtype not in (np.float16, np.float32):
            raise ValueError(
                f"Ultra HDR requires float16 or float32 data, got {pixels.dtype}. "
                f"Use DequantizeFilter to convert integer data to float."
            )

        # Check dimensions
        if pixels.ndim != 3:
            raise ValueError(
                f"Ultra HDR requires 3D array (H, W, C), got shape {pixels.shape}"
            )

        # Check channels
        channels = pixels.shape[2]
        if channels not in (3, 4):
            raise ValueError(
                f"Ultra HDR requires 3 (RGB) or 4 (RGBA) channels, got {channels}"
            )

        # Validate required metadata: color_space
        if 'color_space' not in metadata:
            raise ValueError(
                "Ultra HDR requires 'color_space' in metadata. "
                "Use ColorConvertFilter to set color space (BT.709, BT.2020, Display-P3)."
            )

        color_space = metadata['color_space']
        if not isinstance(color_space, ColorSpace):
            raise TypeError(
                f"color_space must be ColorSpace enum, got {type(color_space).__name__}"
            )

        # Validate supported color spaces
        if color_space not in (ColorSpace.BT709, ColorSpace.BT2020, ColorSpace.DISPLAY_P3):
            raise ValueError(
                f"Ultra HDR supports BT.709, BT.2020, Display-P3. Got {color_space.value}"
            )

        # Validate required metadata: transfer_function (must be LINEAR)
        if 'transfer_function' not in metadata:
            raise ValueError(
                "Ultra HDR requires 'transfer_function' in metadata. "
                "Data must be LINEAR (do NOT use PQEncodeFilter - libultrahdr handles encoding internally)."
            )

        transfer_function = metadata['transfer_function']
        if not isinstance(transfer_function, TransferFunction):
            raise TypeError(
                f"transfer_function must be TransferFunction enum, got {type(transfer_function).__name__}"
            )

        # STRICT: Only LINEAR is supported for float16
        if transfer_function != TransferFunction.LINEAR:
            raise ValueError(
                f"Ultra HDR requires LINEAR transfer function for float16 data. Got {transfer_function.value}. "
                f"Remove PQEncodeFilter from your pipeline - libultrahdr does PQ encoding internally."
            )

    def write(self, img_data: ImageData, options: SaveOptions) -> None:
        """
        Write image as JPEG Ultra HDR

        Args:
            img_data: ImageData with HDR pixels and metadata
            options: Already adapted save options (quality, gainmap_scale)

        Raises:
            ValueError: If image data is invalid
            RuntimeError: If encoding fails
        """
        # Validate
        self.validate(img_data)

        # Options are already adapted by JPEGWriter facade
        adapted_options = options

        # Prepare pixels
        pixels = self._prepare_pixels(img_data)

        # Map metadata to Ultra HDR parameters
        gamut = self._map_color_space(img_data.metadata.get('color_space') or ColorSpace.BT709)
        transfer = imagecodecs.ULTRAHDR.CT.LINEAR  # Always LINEAR for float16

        # Encode
        try:
            encoded_bytes = imagecodecs.ultrahdr_encode(
                pixels,
                level=adapted_options.get('quality'),
                scale=adapted_options.get('gainmap_scale'),
                gamut=gamut,
                transfer=transfer,
                crange=imagecodecs.ULTRAHDR.CR.FULL_RANGE,
            )
        except Exception as e:
            raise RuntimeError(f"Ultra HDR encoding failed: {e}") from e

        # Write to file
        self.ensure_directory()
        with open(self.filepath, 'wb') as f:
            f.write(encoded_bytes)

    def _prepare_pixels(self, img_data: ImageData) -> np.ndarray:
        """
        Prepare pixels for Ultra HDR encoding

        - Convert float32 → float16 if needed
        - Add alpha channel if missing
        - Ensure C-contiguous array

        Args:
            img_data: Input ImageData (must be LINEAR)

        Returns:
            RGBA float16 array ready for encoding
        """
        pixels = img_data.pixels

        # Add alpha channel if missing
        if pixels.shape[2] == 3:
            height, width = pixels.shape[:2]
            rgba = np.ones((height, width, 4), dtype=pixels.dtype)
            rgba[:, :, :3] = pixels
            pixels = rgba

        # Convert to float16 if needed
        if pixels.dtype == np.float32:
            pixels = pixels.astype(np.float16)

        # Ensure C-contiguous
        if not pixels.flags.c_contiguous:
            pixels = np.ascontiguousarray(pixels)

        return pixels

    def _map_color_space(self, color_space: ColorSpace) -> int:
        """
        Map ColorSpace enum to imagecodecs.ULTRAHDR.CG constant

        Args:
            color_space: ColorSpace enum

        Returns:
            ULTRAHDR.CG constant
        """
        mapping = {
            ColorSpace.BT709: imagecodecs.ULTRAHDR.CG.BT_709,
            ColorSpace.BT2020: imagecodecs.ULTRAHDR.CG.BT_2100,  # BT.2100 uses BT.2020 primaries
            ColorSpace.DISPLAY_P3: imagecodecs.ULTRAHDR.CG.DISPLAY_P3,
        }
        return mapping[color_space]
