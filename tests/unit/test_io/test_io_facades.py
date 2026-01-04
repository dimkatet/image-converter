"""
Tests for ImageReader and ImageWriter facade classes

These facades provide automatic format detection and delegation to
format-specific readers/writers based on file extension.
"""
import numpy as np
import pytest
from pathlib import Path

from image_pipeline.core.image_data import ImageData
from image_pipeline.io.reader import ImageReader
from image_pipeline.io.writer import ImageWriter
from image_pipeline.types import SaveOptions


# ============================================================================
# ImageReader Facade Tests
# ============================================================================

class TestImageReaderFacade:
    """Tests for ImageReader facade"""

    def test_create_reader_png(self):
        """Test ImageReader creates PNG reader for .png extension"""
        # Create dummy PNG file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            png_path = f.name
            # Write minimal valid PNG
            f.write(b'\x89PNG\r\n\x1a\n')  # PNG signature
            f.write(b'\x00\x00\x00\rIHDR')  # IHDR chunk
            f.write(b'\x00\x00\x00\x01' * 2)  # 1x1 image
            f.write(b'\x08\x02\x00\x00\x00')  # 8-bit RGB
            f.write(b'\x90wS\xde')  # CRC
            f.write(b'\x00\x00\x00\x00IEND\xaeB`\x82')  # IEND chunk

        try:
            reader = ImageReader(png_path)
            # Check that reader was created (won't be None)
            assert reader._reader is not None
        finally:
            Path(png_path).unlink()

    def test_create_reader_tiff(self, tmp_path):
        """Test ImageReader creates TIFF reader for .tiff extension"""
        tiff_path = tmp_path / "test.tiff"

        # Create minimal TIFF using tifffile
        from tifffile import imwrite
        pixels = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        imwrite(tiff_path, pixels)

        reader = ImageReader(str(tiff_path))
        assert reader._reader is not None

    def test_create_reader_tif_extension(self, tmp_path):
        """Test ImageReader handles .tif extension (alias for .tiff)"""
        tif_path = tmp_path / "test.tif"

        from tifffile import imwrite
        pixels = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        imwrite(tif_path, pixels)

        reader = ImageReader(str(tif_path))
        assert reader._reader is not None

    def test_create_reader_webp(self, tmp_path):
        """Test ImageReader creates WebP reader for .webp extension"""
        webp_path = tmp_path / "test.webp"

        # Create minimal WebP using imagecodecs
        from imagecodecs import webp_encode
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        webp_bytes = webp_encode(pixels, level=80)
        webp_path.write_bytes(webp_bytes)

        reader = ImageReader(str(webp_path))
        assert reader._reader is not None

    def test_create_reader_avif(self, tmp_path):
        """Test ImageReader creates AVIF reader for .avif extension"""
        avif_path = tmp_path / "test.avif"

        # Create minimal AVIF using imagecodecs
        from imagecodecs import avif_encode
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        avif_bytes = avif_encode(pixels, level=90)
        avif_path.write_bytes(avif_bytes)

        reader = ImageReader(str(avif_path))
        assert reader._reader is not None

    def test_create_reader_jpeg(self, tmp_path):
        """Test ImageReader creates JPEG reader for .jpg extension"""
        jpeg_path = tmp_path / "test.jpg"

        # Create minimal JPEG using Pillow
        from PIL import Image
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img = Image.fromarray(pixels, mode='RGB')
        img.save(jpeg_path, quality=95)

        reader = ImageReader(str(jpeg_path))
        assert reader._reader is not None

    def test_create_reader_jpeg_extension(self, tmp_path):
        """Test ImageReader handles .jpeg extension"""
        jpeg_path = tmp_path / "test.jpeg"

        from PIL import Image
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img = Image.fromarray(pixels, mode='RGB')
        img.save(jpeg_path, quality=95)

        reader = ImageReader(str(jpeg_path))
        assert reader._reader is not None

    def test_unsupported_format_raises_error(self):
        """Test ImageReader raises ValueError for unsupported format"""
        with pytest.raises(ValueError, match="Unsupported format: .bmp"):
            ImageReader("/path/to/image.bmp")

    def test_unsupported_format_shows_supported_list(self):
        """Test error message includes list of supported formats"""
        with pytest.raises(ValueError, match="Supported formats:"):
            ImageReader("/path/to/image.gif")

    def test_case_insensitive_extension(self, tmp_path):
        """Test extension matching is case-insensitive"""
        png_path = tmp_path / "test.PNG"

        # Create minimal PNG
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_png = f.name
            f.write(b'\x89PNG\r\n\x1a\n')
            f.write(b'\x00\x00\x00\rIHDR')
            f.write(b'\x00\x00\x00\x01' * 2)
            f.write(b'\x08\x02\x00\x00\x00')
            f.write(b'\x90wS\xde')
            f.write(b'\x00\x00\x00\x00IEND\xaeB`\x82')

        # Move to uppercase extension path
        Path(temp_png).rename(png_path)

        try:
            reader = ImageReader(str(png_path))
            assert reader._reader is not None
        finally:
            png_path.unlink()

    def test_read_delegates_to_format_reader(self, tmp_path):
        """Test ImageReader.read() delegates to format-specific reader"""
        # Create a simple PNG
        from PIL import Image
        png_path = tmp_path / "test.png"
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img = Image.fromarray(pixels, mode='RGB')
        img.save(png_path)

        reader = ImageReader(str(png_path))
        result = reader.read()

        # Should return ImageData
        assert isinstance(result, ImageData)
        assert result.pixels is not None


# ============================================================================
# ImageWriter Facade Tests
# ============================================================================

class TestImageWriterFacade:
    """Tests for ImageWriter facade"""

    def test_write_png(self, tmp_path):
        """Test ImageWriter writes PNG for .png extension"""
        png_path = tmp_path / "output.png"
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = ImageWriter(str(png_path))
        writer.write(img_data, {})

        assert png_path.exists()

    def test_write_webp(self, tmp_path):
        """Test ImageWriter writes WebP for .webp extension"""
        webp_path = tmp_path / "output.webp"
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = ImageWriter(str(webp_path))
        writer.write(img_data, {'quality': 90})

        assert webp_path.exists()

    def test_write_avif(self, tmp_path):
        """Test ImageWriter writes AVIF for .avif extension"""
        avif_path = tmp_path / "output.avif"
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = ImageWriter(str(avif_path))
        writer.write(img_data, {'quality': 90})

        assert avif_path.exists()

    @pytest.mark.skip(reason="Standard JPEG writer not implemented, only Ultra HDR")
    def test_write_jpeg(self, tmp_path):
        """Test ImageWriter writes JPEG for .jpg extension"""
        jpeg_path = tmp_path / "output.jpg"
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = ImageWriter(str(jpeg_path))
        writer.write(img_data, {'quality': 95})

        assert jpeg_path.exists()

    @pytest.mark.skip(reason="Standard JPEG writer not implemented, only Ultra HDR")
    def test_write_jpeg_extension(self, tmp_path):
        """Test ImageWriter handles .jpeg extension"""
        jpeg_path = tmp_path / "output.jpeg"
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = ImageWriter(str(jpeg_path))
        writer.write(img_data, {'quality': 95})

        assert jpeg_path.exists()

    def test_unsupported_format_raises_error(self):
        """Test ImageWriter raises ValueError for unsupported format"""
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = ImageWriter("/path/to/output.bmp")

        with pytest.raises(ValueError, match="Unsupported format: .bmp"):
            writer.write(img_data, {})

    def test_unsupported_format_shows_supported_list(self):
        """Test error message includes list of supported formats"""
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = ImageWriter("/path/to/output.gif")

        with pytest.raises(ValueError, match="Supported formats:"):
            writer.write(img_data, {})

    def test_case_insensitive_extension(self, tmp_path):
        """Test extension matching is case-insensitive"""
        png_path = tmp_path / "output.PNG"
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = ImageWriter(str(png_path))
        writer.write(img_data, {})

        assert png_path.exists()

    def test_writer_creates_directory(self, tmp_path):
        """Test ImageWriter creates parent directory if it doesn't exist"""
        nested_path = tmp_path / "subdir" / "nested" / "output.png"
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        writer = ImageWriter(str(nested_path))
        writer.write(img_data, {})

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_writer_validates_data(self, tmp_path):
        """Test ImageWriter validates data before writing"""
        png_path = tmp_path / "output.png"

        # Create invalid ImageData (empty pixels)
        empty_pixels = np.array([], dtype=np.uint8)
        img_data = ImageData(pixels=empty_pixels.reshape(0, 0, 3))

        writer = ImageWriter(str(png_path))

        # Should raise error during validation
        with pytest.raises((ValueError, IndexError)):
            writer.write(img_data, {})

    def test_write_with_options(self, tmp_path):
        """Test ImageWriter passes options to format writer"""
        png_path = tmp_path / "output.png"
        pixels = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img_data = ImageData(pixels=pixels)

        options: SaveOptions = {
            'compression_level': 9,
        }

        writer = ImageWriter(str(png_path))
        writer.write(img_data, options)

        assert png_path.exists()


# ============================================================================
# Format Registration Tests
# ============================================================================

class TestFormatRegistration:
    """Tests for format registration system"""

    def test_reader_registered_formats(self):
        """Test ImageReader has all expected formats registered"""
        registered = ImageReader._READERS.keys()

        assert '.png' in registered
        assert '.tiff' in registered
        assert '.tif' in registered
        assert '.avif' in registered
        assert '.webp' in registered
        assert '.jpg' in registered
        assert '.jpeg' in registered

    def test_writer_registered_formats(self):
        """Test ImageWriter has all expected formats registered"""
        registered = ImageWriter._WRITERS.keys()

        assert '.png' in registered
        assert '.avif' in registered
        assert '.webp' in registered
        assert '.jpg' in registered
        assert '.jpeg' in registered

    def test_reader_format_count(self):
        """Test ImageReader has correct number of registered extensions"""
        # Should have at least: PNG, TIFF (2 exts), AVIF, WebP, JPEG (2 exts)
        assert len(ImageReader._READERS) >= 7

    def test_writer_format_count(self):
        """Test ImageWriter has correct number of registered extensions"""
        # Should have at least: PNG, AVIF, WebP, JPEG (2 exts)
        assert len(ImageWriter._WRITERS) >= 5
