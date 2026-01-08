"""OpenEXR metadata adapter - converts ImageMetadata to/from EXR header attributes"""

from typing import Any

from image_pipeline.types import ImageMetadata
from image_pipeline.color import (
    get_primaries_from_metadata,
    match_color_space,
)


class EXRMetadataAdapter:
    """Converts between ImageMetadata and OpenEXR header attributes"""

    @staticmethod
    def to_exr_header(metadata: ImageMetadata, channels: int) -> dict[str, Any]:
        """
        Convert ImageMetadata to OpenEXR header attributes

        Args:
            metadata: Generic image metadata
            channels: Number of channels (3 for RGB, 4 for RGBA)

        Returns:
            Dictionary of EXR header attributes

        Standard EXR attributes:
            - chromaticities: Color primaries (red, green, blue, white xy coordinates)
            - whiteLuminance: Reference white luminance in cd/m² (from paper_white)

        Notes:
            - EXR data is always scene-linear (no transfer function stored)
            - Only standard EXR attributes are written (no custom attributes)
        """
        header = {}

        # Chromaticities - standard EXR attribute
        primaries = get_primaries_from_metadata(
            metadata.get('color_space'),
            metadata.get('color_primaries')
        )

        if primaries:
            # OpenEXR.File API expects chromaticities as tuple of 8 floats:
            # (red_x, red_y, green_x, green_y, blue_x, blue_y, white_x, white_y)
            header['chromaticities'] = (
                float(primaries['red'][0]), float(primaries['red'][1]),
                float(primaries['green'][0]), float(primaries['green'][1]),
                float(primaries['blue'][0]), float(primaries['blue'][1]),
                float(primaries['white'][0]), float(primaries['white'][1])
            )

        # whiteLuminance - standard EXR attribute for reference white (in cd/m²)
        # Maps to our paper_white metadata
        paper_white = metadata.get('paper_white')
        if paper_white is not None:
            header['whiteLuminance'] = float(paper_white)

        return header

    @staticmethod
    def from_exr_header(header: dict[str, Any]) -> ImageMetadata:
        """
        Extract ImageMetadata from OpenEXR header attributes

        Args:
            header: Dictionary of EXR header attributes (from OpenEXR.File.header())

        Returns:
            ImageMetadata with extracted values

        Reads:
            - chromaticities → color_space (matched) or color_primaries (custom)
            - whiteLuminance → paper_white

        Defaults:
            - color_space defaults to BT709 if no chromaticities
            - transfer_function always set to LINEAR (EXR is scene-linear)
        """
        metadata: ImageMetadata = {
            'format': 'OpenEXR',
        }

        # Chromaticities
        if 'chromaticities' in header:
            chroma = header['chromaticities']
            # OpenEXR Chromaticities object has red, green, blue, white attributes (Imath.V2f)
            primaries = {
                'red': (chroma.red.x, chroma.red.y),
                'green': (chroma.green.x, chroma.green.y),
                'blue': (chroma.blue.x, chroma.blue.y),
                'white': (chroma.white.x, chroma.white.y),
            }

            # Try to match to standard color space
            color_space = match_color_space(primaries)
            if color_space:
                metadata['color_space'] = color_space
            else:
                # Store custom primaries
                metadata['color_primaries'] = primaries
        else:
            # Default: assume sRGB/BT.709 if no chromaticities specified
            from image_pipeline.types import ColorSpace
            metadata['color_space'] = ColorSpace.BT709

        # whiteLuminance - standard EXR attribute for reference white
        if 'whiteLuminance' in header:
            metadata['paper_white'] = float(header['whiteLuminance'])

        return metadata
