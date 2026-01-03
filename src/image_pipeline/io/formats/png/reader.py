"""PNG format reader using imagecodecs"""
import os
from imagecodecs import png_decode

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.formats.base import FormatReader
from image_pipeline.types import ImageMetadata, TransferFunction, ColorSpace


class PNGFormatReader(FormatReader):
    """Reader for PNG images using imagecodecs"""

    def read(self) -> ImageData:
        """
        Read PNG image with metadata

        Returns:
            ImageData with pixels and metadata
        """
        try:
            # Step 1: Read PNG bytes
            with open(self.filepath, 'rb') as f:
                png_bytes = f.read()

            # Step 2: Decode pixels
            pixels = png_decode(png_bytes)

            # Step 3: Read metadata from PNG chunks
            metadata = self._read_metadata()

            # ImageData will automatically fill: shape, dtype, channels, bit_depth
            return ImageData(pixels, metadata)

        except Exception as e:
            raise IOError(f"Error reading PNG: {e}")

    def _read_metadata(self) -> ImageMetadata:
        """
        Extract metadata from PNG chunks

        Reads HDR metadata (cICP, mDCv, cLLi) using PNGMetadataCodec.

        Returns:
            ImageMetadata dictionary
        """
        from image_pipeline.metadata.png.codec import PNGMetadataCodec

        # Start with basic file metadata
        metadata: ImageMetadata = {
            'format': 'PNG',
            'filename': self.filepath.name,
            'file_size': os.path.getsize(self.filepath),
        }

        try:
            # Read PNG chunks
            codec = PNGMetadataCodec(str(self.filepath))
            codec.read_chunks()
            chunk_metadata = codec.get_metadata()

            # Extract cICP (color info)
            if 'cicp' in chunk_metadata:
                cicp = chunk_metadata['cicp']

                # Map CICP values to our metadata format
                # Transfer characteristics
                transfer_map = {
                    1: TransferFunction.SRGB,  # BT.709 uses same curve as sRGB
                    13: TransferFunction.SRGB,
                    16: TransferFunction.PQ,
                    18: TransferFunction.HLG
                }
                if cicp['transfer_characteristics'] in transfer_map:
                    metadata['transfer_function'] = transfer_map[cicp['transfer_characteristics']]

                # Color space (primaries)
                space_map = {
                    1: ColorSpace.BT709,
                    9: ColorSpace.BT2020,
                    12: ColorSpace.DISPLAY_P3
                }
                if cicp['color_primaries'] in space_map:
                    metadata['color_space'] = space_map[cicp['color_primaries']]

            # Extract cLLi (content light level)
            if 'clli' in chunk_metadata:
                clli = chunk_metadata['clli']
                metadata['max_cll'] = clli['max_content_light_level']
                metadata['max_fall'] = clli['max_frame_average_light_level']

            # Extract mDCv (mastering display)
            if 'mdcv' in chunk_metadata:
                mdcv = chunk_metadata['mdcv']
                metadata['mastering_display_max_luminance'] = mdcv['max_luminance_nits']
                metadata['mastering_display_min_luminance'] = mdcv['min_luminance_nits']

            # Extract ICC profile
            if 'icc_profile' in chunk_metadata:
                icc_profile = chunk_metadata['icc_profile']
                if icc_profile:  # Only add if not None
                    metadata['icc_profile'] = icc_profile

        except Exception:
            # If metadata reading fails, continue with basic metadata
            # Don't crash the whole read operation
            pass

        return metadata