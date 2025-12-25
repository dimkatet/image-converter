"""
JPEG format writer with automatic delegation to standard/Ultra HDR implementations
"""

from image_pipeline.core.image_data import ImageData
from image_pipeline.types import SaveOptions
from image_pipeline.io.formats.base import FormatWriter
from .options import JPEGSaveOptionsAdapter


class JPEGWriter(FormatWriter):
    """
    Facade for JPEG writing with automatic delegation

    Delegates to:
        - UltraHDRWriter: if options['ultra_hdr'] == True
        - StandardJPEGWriter: if options['ultra_hdr'] == False (TODO: not implemented yet)

    This facade handles:
        - Option validation via JPEGSaveOptionsAdapter
        - Delegation based on ultra_hdr flag
        - Consistent interface for both JPEG variants
    """

    def validate(self, img_data: ImageData) -> None:
        """
        Validate image data for JPEG encoding

        Actual validation is delegated to the specific implementation
        (Ultra HDR or standard JPEG)

        Args:
            img_data: ImageData to validate
        """
        # Validation happens in the delegated writer
        pass

    def write(self, img_data: ImageData, options: SaveOptions) -> None:
        """
        Write image as JPEG (standard or Ultra HDR)

        Args:
            img_data: ImageData with pixels and metadata
            options: Save options (quality, ultra_hdr, etc.)

        Raises:
            ValueError: If image data is invalid
            RuntimeError: If encoding fails
            NotImplementedError: If standard JPEG is requested (not implemented yet)
        """
        # Validate and adapt options
        adapted_options = JPEGSaveOptionsAdapter.adapt(options)

        # Delegate based on ultra_hdr flag
        if adapted_options['ultra_hdr']:
            # Use Ultra HDR implementation
            from .ultrahdr import UltraHDRWriter
            writer = UltraHDRWriter(str(self.filepath))
            writer.validate(img_data)
            writer.write(img_data, adapted_options)
        else:
            # Standard JPEG not implemented yet
            raise NotImplementedError(
                "Standard JPEG encoding is not implemented yet. "
                "Use --ultra-hdr flag for JPEG Ultra HDR encoding, "
                "or use a different format (PNG, AVIF, WebP, TIFF)."
            )
