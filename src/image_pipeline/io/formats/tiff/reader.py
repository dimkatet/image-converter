"""TIFF format reader"""
import os
from typing import Optional, Dict, Tuple

import tifffile

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatReader
from image_pipeline.types import ImageMetadata, TransferFunction
from image_pipeline.color import match_color_space


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
                    color_space = match_color_space(primaries)

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

                # Read custom metadata from ImageDescription tag (270)
                description_tag = page.tags.get(270)
                if description_tag:
                    description = description_tag.value
                    if isinstance(description, str):
                        # Parse key=value pairs from ImageDescription
                        self._parse_description_metadata(description, metadata)

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
    def _parse_description_metadata(description: str, metadata: ImageMetadata) -> None:
        """
        Parse custom metadata from ImageDescription tag

        Format: "key1=value1; key2=value2; ..."

        Args:
            description: ImageDescription string
            metadata: Metadata dict to update
        """
        # Split by semicolon and parse key=value pairs
        for pair in description.split(';'):
            pair = pair.strip()
            if '=' in pair:
                key, value = pair.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Parse known metadata fields
                try:
                    if key == 'bit_depth':
                        metadata['bit_depth'] = int(value)
                    elif key == 'paper_white':
                        metadata['paper_white'] = float(value)
                    elif key == 'mastering_display_max_luminance':
                        metadata['mastering_display_max_luminance'] = float(value)
                    elif key == 'mastering_display_min_luminance':
                        metadata['mastering_display_min_luminance'] = float(value)
                except (ValueError, TypeError):
                    # Skip invalid values
                    pass