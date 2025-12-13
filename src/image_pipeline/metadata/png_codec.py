"""
PNG Metadata Codec - for reading and writing PNG chunks
Supports text chunks (tEXt, zTXt, iTXt) and other metadata
"""
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ChunkType(Enum):
    """PNG chunk types for metadata"""
    TEXT = b'tEXt'      # Plain text (Latin-1)
    ZTXT = b'zTXt'      # Compressed text
    ITXT = b'iTXt'      # International text (UTF-8)
    TIME = b'tIME'      # Last modification time
    PHYS = b'pHYs'      # Physical pixel dimensions
    GAMA = b'gAMA'      # Gamma
    CHRM = b'cHRM'      # Chromaticity coordinates
    SRGB = b'sRGB'      # sRGB color space
    ICCP = b'iCCP'      # ICC profile


@dataclass
class PNGChunk:
    """PNG chunk representation"""
    chunk_type: bytes
    data: bytes
    
    def __repr__(self):
        type_str = self.chunk_type.decode('latin-1', errors='ignore')
        return f"PNGChunk(type={type_str}, size={len(self.data)})"


class PNGMetadataCodec:
    """Codec for working with PNG metadata via chunks"""
    
    PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
    
    # Critical chunks that must not be removed
    CRITICAL_CHUNKS = {b'IHDR', b'PLTE', b'IDAT', b'IEND'}
    
    def __init__(self, filepath: Optional[str] = None):
        """
        Args:
            filepath: Path to PNG file (optional)
        """
        self.filepath = Path(filepath) if filepath else None
        self._chunks: List[PNGChunk] = []
    
    def read_chunks(self, filepath: Optional[str] = None) -> List[PNGChunk]:
        """
        Read all chunks from a PNG file
        
        Args:
            filepath: Path to file (or uses self.filepath)
            
        Returns:
            List of PNGChunk objects
        """
        path = Path(filepath) if filepath else self.filepath
        if not path:
            raise ValueError("Filepath not specified")
        
        self._chunks = []
        
        with open(path, 'rb') as f:
            # Check PNG signature
            signature = f.read(8)
            if signature != self.PNG_SIGNATURE:
                raise ValueError(f"Not a valid PNG file: {path}")
            
            # Read chunks
            while True:
                chunk = self._read_chunk(f)
                if chunk is None:
                    break
                self._chunks.append(chunk)
                
                # IEND - end of file
                if chunk.chunk_type == b'IEND':
                    break
        
        return self._chunks
    
    def _read_chunk(self, f) -> Optional[PNGChunk]:
        """Read a single chunk from file"""
        # Data length (4 bytes)
        length_bytes = f.read(4)
        if len(length_bytes) < 4:
            return None
        
        length = struct.unpack('>I', length_bytes)[0]
        
        # Chunk type (4 bytes)
        chunk_type = f.read(4)
        if len(chunk_type) < 4:
            return None
        
        # Chunk data
        data = f.read(length)
        if len(data) < length:
            return None
        
        # CRC (4 bytes) - skip, but check
        crc = f.read(4)
        if len(crc) < 4:
            return None
        
        # Check CRC
        expected_crc = struct.unpack('>I', crc)[0]
        calculated_crc = zlib.crc32(chunk_type + data) & 0xffffffff
        if expected_crc != calculated_crc:
            print(f"Warning: CRC mismatch for chunk {chunk_type}")
        
        return PNGChunk(chunk_type=chunk_type, data=data)
    
    def write_chunks(self, filepath: str, chunks: Optional[List[PNGChunk]] = None) -> None:
        """
        Write chunks to a PNG file
        
        Args:
            filepath: Path to save
            chunks: List of chunks (or uses self._chunks)
        """
        chunks_to_write = chunks if chunks is not None else self._chunks
        
        if not chunks_to_write:
            raise ValueError("No chunks to write")
        
        with open(filepath, 'wb') as f:
            # PNG signature
            f.write(self.PNG_SIGNATURE)
            
            # Write all chunks
            for chunk in chunks_to_write:
                self._write_chunk(f, chunk)
    
    def _write_chunk(self, f, chunk: PNGChunk) -> None:
        """Write a single chunk to file"""
        # Data length
        f.write(struct.pack('>I', len(chunk.data)))
        
        # Chunk type
        f.write(chunk.chunk_type)
        
        # Data
        f.write(chunk.data)
        
        # CRC
        crc = zlib.crc32(chunk.chunk_type + chunk.data) & 0xffffffff
        f.write(struct.pack('>I', crc))
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from chunks in a convenient format
        
        Returns:
            Dictionary with metadata
        """
        metadata = {}
        
        for chunk in self._chunks:
            if chunk.chunk_type == ChunkType.TEXT.value:
                key, value = self._parse_text_chunk(chunk.data)
                metadata[key] = value
            
            elif chunk.chunk_type == ChunkType.ZTXT.value:
                key, value = self._parse_ztxt_chunk(chunk.data)
                metadata[key] = value
            
            elif chunk.chunk_type == ChunkType.ITXT.value:
                key, value = self._parse_itxt_chunk(chunk.data)
                metadata[key] = value
            
            elif chunk.chunk_type == ChunkType.TIME.value:
                metadata['modification_time'] = self._parse_time_chunk(chunk.data)
            
            elif chunk.chunk_type == ChunkType.PHYS.value:
                metadata['physical_dimensions'] = self._parse_phys_chunk(chunk.data)
            
            elif chunk.chunk_type == ChunkType.GAMA.value:
                metadata['gamma'] = self._parse_gama_chunk(chunk.data)
        
        return metadata
    
    def set_metadata(self, metadata: Dict[str, str], compress: bool = False) -> None:
        """
        Set text metadata (add tEXt/zTXt chunks)
        
        Args:
            metadata: Dictionary of key-value metadata
            compress: Use compression (zTXt instead of tEXt)
        """
        # Remove existing text chunks
        self._chunks = [
            c for c in self._chunks 
            if c.chunk_type not in {ChunkType.TEXT.value, ChunkType.ZTXT.value, ChunkType.ITXT.value}
        ]
        
        # Find position to insert (after IHDR, before IDAT)
        insert_pos = self._find_metadata_insert_position()
        
        # Create new chunks
        for key, value in metadata.items():
            if compress:
                chunk = self._create_ztxt_chunk(key, value)
            else:
                chunk = self._create_text_chunk(key, value)
            
            self._chunks.insert(insert_pos, chunk)
            insert_pos += 1
    
    def _find_metadata_insert_position(self) -> int:
        """Finds position to insert metadata (after IHDR)"""
        for i, chunk in enumerate(self._chunks):
            if chunk.chunk_type == b'IHDR':
                return i + 1
        return 0
    
    def _parse_text_chunk(self, data: bytes) -> Tuple[str, str]:
        """Parse tEXt chunk"""
        null_pos = data.find(b'\x00')
        if null_pos == -1:
            return "", ""
        
        keyword = data[:null_pos].decode('latin-1')
        text = data[null_pos + 1:].decode('latin-1', errors='replace')
        return keyword, text
    
    def _parse_ztxt_chunk(self, data: bytes) -> Tuple[str, str]:
        """Parse zTXt chunk (compressed text)"""
        null_pos = data.find(b'\x00')
        if null_pos == -1:
            return "", ""
        
        keyword = data[:null_pos].decode('latin-1')
        compression_method = data[null_pos + 1]  # Should be 0 (deflate)
        
        if compression_method != 0:
            return keyword, ""
        
        compressed_data = data[null_pos + 2:]
        try:
            decompressed = zlib.decompress(compressed_data)
            text = decompressed.decode('latin-1', errors='replace')
            return keyword, text
        except:
            return keyword, ""
    
    def _parse_itxt_chunk(self, data: bytes) -> Tuple[str, str]:
        """Parse iTXt chunk (international text, UTF-8)"""
        null_pos = data.find(b'\x00')
        if null_pos == -1:
            return "", ""
        
        keyword = data[:null_pos].decode('latin-1')
        
        # Skip compression flag and method
        compression_flag = data[null_pos + 1]
        compression_method = data[null_pos + 2]
        
        # Find language tag and translated keyword (both null-terminated)
        rest = data[null_pos + 3:]
        lang_null = rest.find(b'\x00')
        if lang_null == -1:
            return keyword, ""
        
        rest = rest[lang_null + 1:]
        trans_null = rest.find(b'\x00')
        if trans_null == -1:
            return keyword, ""
        
        text_data = rest[trans_null + 1:]
        
        # Decompress if needed
        if compression_flag == 1:
            try:
                text_data = zlib.decompress(text_data)
            except:
                return keyword, ""
        
        text = text_data.decode('utf-8', errors='replace')
        return keyword, text
    
    def _parse_time_chunk(self, data: bytes) -> Dict[str, int]:
        """Parse tIME chunk"""
        if len(data) < 7:
            return {}
        
        year = struct.unpack('>H', data[0:2])[0]
        month, day, hour, minute, second = struct.unpack('5B', data[2:7])
        
        return {
            'year': year,
            'month': month,
            'day': day,
            'hour': hour,
            'minute': minute,
            'second': second
        }
    
    def _parse_phys_chunk(self, data: bytes) -> Dict[str, int]:
        """Parse pHYs chunk"""
        if len(data) < 9:
            return {}
        
        pixels_per_unit_x = struct.unpack('>I', data[0:4])[0]
        pixels_per_unit_y = struct.unpack('>I', data[4:8])[0]
        unit = data[8]  # 0 = unknown, 1 = meter
        
        return {
            'pixels_per_unit_x': pixels_per_unit_x,
            'pixels_per_unit_y': pixels_per_unit_y,
            'unit': 'meter' if unit == 1 else 'unknown'
        }
    
    def _parse_gama_chunk(self, data: bytes) -> float:
        """Parse gAMA chunk"""
        if len(data) < 4:
            return 0.0
        
        gamma_int = struct.unpack('>I', data)[0]
        return gamma_int / 100000.0
    
    def _create_text_chunk(self, keyword: str, text: str) -> PNGChunk:
        """Create tEXt chunk"""
        keyword_bytes = keyword.encode('latin-1')[:79]  # Max 79 bytes
        text_bytes = text.encode('latin-1', errors='replace')
        
        data = keyword_bytes + b'\x00' + text_bytes
        return PNGChunk(chunk_type=ChunkType.TEXT.value, data=data)
    
    def _create_ztxt_chunk(self, keyword: str, text: str) -> PNGChunk:
        """Create zTXt chunk (compressed)"""
        keyword_bytes = keyword.encode('latin-1')[:79]
        text_bytes = text.encode('latin-1', errors='replace')
        compressed = zlib.compress(text_bytes, level=9)
        
        data = keyword_bytes + b'\x00' + b'\x00' + compressed
        return PNGChunk(chunk_type=ChunkType.ZTXT.value, data=data)
    
    def copy_image_with_metadata(self, source: str, destination: str, 
                                 metadata: Dict[str, str]) -> None:
        """
        Copy PNG with adding/updating metadata
        
        Args:
            source: Source PNG file
            destination: Destination file
            metadata: Metadata to add
        """
        # Read chunks from source file
        self.read_chunks(source)
        
        # Set new metadata
        self.set_metadata(metadata)
        
        # Save
        self.write_chunks(destination)
    
    def list_chunks(self) -> None:
        """Print list of all chunks for debugging"""
        print(f"Total chunks: {len(self._chunks)}")
        for i, chunk in enumerate(self._chunks):
            type_str = chunk.chunk_type.decode('latin-1', errors='ignore')
            critical = "critical" if chunk.chunk_type in self.CRITICAL_CHUNKS else "ancillary"
            print(f"{i:3d}. {type_str:4s} - {len(chunk.data):8d} bytes ({critical})")