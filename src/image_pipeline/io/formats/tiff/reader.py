"""TIFF format reader"""
import os
from typing import Optional, Dict, Tuple

import tifffile

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatReader
from image_pipeline.types import ColorSpace, ImageMetadata, TransferFunction
from image_pipeline.constants import STANDARD_COLOR_PRIMARIES


class TiffFormatReader(FormatReader):
    """Reader for TIFF images"""
    
    def read(self) -> ImageData:
        """
        Read TIFF image
        
        Returns:
            ImageData with pixels and metadata
        """
        try:
            pixels = tifffile.imread(self.filepath)
            with tifffile.TiffFile(self.filepath) as tif:
                page: tifffile.TiffPage = tif.pages[0] # type: ignore

                metadata: ImageMetadata = {
                    'format': 'TIFF',
                    'filename': self.filepath.name,
                    'file_size': os.path.getsize(self.filepath),
                }

                # Read chromaticity coordinates from TIFF tags
                primaries = self._read_chromaticities(page)

                if primaries:
                    # Try to match to a standard color space
                    color_space = self._match_color_space(primaries)

                    if color_space:
                        # Matched a standard color space
                        metadata['color_space'] = color_space
                    else:
                        # Custom primaries - store as-is
                        metadata['color_primaries'] = primaries

                # Detect transfer function based on sample format
                # TIFF with float samples are typically linear
                if page.sampleformat == 3:  # SAMPLEFORMAT.IEEEFP = 3
                    metadata['transfer_function'] = TransferFunction.LINEAR

                # TODO: Auto-detect paper_white from TIFF tags for scene-referred HDR
                # Scene-referred TIFF files encode linear values relative to a paper white
                # (typically 100 nits). Consider reading relevant EXIF/TIFF tags:
                # - Custom tags for luminance reference
                # Then set metadata['paper_white'] = detected_value

            return ImageData(pixels, metadata)

        except Exception as e:
            raise IOError(f"Error reading TIFF file: {e}")

    @staticmethod
    def _read_chromaticities(page: tifffile.TiffPage) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Read chromaticity coordinates from TIFF tags

        Args:
            page: TiffFile page

        Returns:
            Dictionary with 'white', 'red', 'green', 'blue' chromaticity coordinates
            or None if tags not found
        """
        # WhitePoint tag (318) - format: (x_num, x_den, y_num, y_den)
        white_point_tag = page.tags.get(318)
        if not white_point_tag:
            return None

        wp_values = white_point_tag.value
        if len(wp_values) < 4:
            return None

        white_x = wp_values[0] / wp_values[1]
        white_y = wp_values[2] / wp_values[3]

        # PrimaryChromaticities tag (319) - format: (R_x_num, R_x_den, R_y_num, R_y_den, ...)
        primaries_tag = page.tags.get(319)
        if not primaries_tag:
            return None

        prim_values = primaries_tag.value
        if len(prim_values) < 12:
            return None

        red_x = prim_values[0] / prim_values[1]
        red_y = prim_values[2] / prim_values[3]
        green_x = prim_values[4] / prim_values[5]
        green_y = prim_values[6] / prim_values[7]
        blue_x = prim_values[8] / prim_values[9]
        blue_y = prim_values[10] / prim_values[11]

        return {
            'white': (white_x, white_y),
            'red': (red_x, red_y),
            'green': (green_x, green_y),
            'blue': (blue_x, blue_y)
        }

    @staticmethod
    def _match_color_space(primaries: Dict[str, Tuple[float, float]],
                          tolerance: float = 0.001) -> Optional[ColorSpace]:
        """
        Match chromaticity primaries to a standard color space

        Args:
            primaries: Dictionary with chromaticity coordinates
            tolerance: Maximum allowed difference for matching (default: 0.001)

        Returns:
            ColorSpace enum if matched, None otherwise
        """
        for color_space, std_primaries in STANDARD_COLOR_PRIMARIES.items():
            # Check if all primaries match within tolerance
            matches = all(
                abs(primaries[color][0] - std_primaries[color][0]) < tolerance and
                abs(primaries[color][1] - std_primaries[color][1]) < tolerance
                for color in ['red', 'green', 'blue', 'white']
            )

            if matches:
                return color_space

        return None