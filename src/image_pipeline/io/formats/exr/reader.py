"""OpenEXR format reader"""

import os
import numpy as np
import OpenEXR
import Imath

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatReader
from image_pipeline.types import ImageMetadata, TransferFunction
from .metadata import EXRMetadataAdapter


class EXRFormatReader(FormatReader):
    """Reader for OpenEXR images"""

    def read(self) -> ImageData:
        """
        Read OpenEXR image

        Returns:
            ImageData with float32 pixels and metadata

        Note:
            - Reads RGB or RGBA channels
            - Converts HALF to float32, FLOAT remains float32
            - All pixels returned in (H, W, C) format
            - Assumes scanline images (not tiled/deep/multi-part)
        """
        try:
            # Open EXR file
            exr_file = OpenEXR.InputFile(str(self.filepath))
            header = exr_file.header()

            # Get image dimensions
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            # Determine channels
            channels_dict = header['channels']
            has_alpha = 'A' in channels_dict

            if has_alpha:
                channel_names = ['R', 'G', 'B', 'A']
                num_channels = 4
            else:
                channel_names = ['R', 'G', 'B']
                num_channels = 3

            # Check that all required channels exist
            for ch in channel_names:
                if ch not in channels_dict:
                    raise ValueError(
                        f"EXR file missing required channel '{ch}'. "
                        f"Available channels: {list(channels_dict.keys())}"
                    )

            # Determine pixel type (use R channel as reference)
            pixel_type = channels_dict['R'].type

            # Read channel data
            channel_data = {}
            for ch_name in channel_names:
                # Read raw bytes
                raw_bytes = exr_file.channel(ch_name)

                # Convert to numpy array based on pixel type (compare using .v attribute)
                if pixel_type.v == Imath.PixelType.HALF:
                    # HALF (16-bit float)
                    dt = np.dtype(np.float16)
                    channel_array = np.frombuffer(raw_bytes, dtype=dt)
                elif pixel_type.v == Imath.PixelType.FLOAT:
                    # FLOAT (32-bit float)
                    dt = np.dtype(np.float32)
                    channel_array = np.frombuffer(raw_bytes, dtype=dt)
                elif pixel_type.v == Imath.PixelType.UINT:
                    # UINT (32-bit unsigned int)
                    dt = np.dtype(np.uint32)
                    channel_array = np.frombuffer(raw_bytes, dtype=dt)
                else:
                    raise ValueError(f"Unsupported EXR pixel type: {pixel_type}")

                # Reshape to (height, width)
                channel_array = channel_array.reshape((height, width))

                # Convert to float32 for consistency
                channel_data[ch_name] = channel_array.astype(np.float32)

            # Stack channels into (H, W, C) array
            if has_alpha:
                pixels = np.stack([
                    channel_data['R'],
                    channel_data['G'],
                    channel_data['B'],
                    channel_data['A']
                ], axis=-1)
            else:
                pixels = np.stack([
                    channel_data['R'],
                    channel_data['G'],
                    channel_data['B']
                ], axis=-1)

            # Extract metadata
            metadata = EXRMetadataAdapter.from_exr_header(header)

            # Add file info
            metadata['filename'] = self.filepath.name
            metadata['file_size'] = os.path.getsize(self.filepath)

            # EXR files are always scene-linear
            metadata['transfer_function'] = TransferFunction.LINEAR

            return ImageData(pixels, metadata)

        except Exception as e:
            raise IOError(f"Error reading OpenEXR file: {e}")
