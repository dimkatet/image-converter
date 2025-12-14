"""
PNG Metadata Codec - for reading and writing PNG chunks
Supports text chunks (tEXt, zTXt, iTXt) and HDR metadata chunks (cICP, mDCv, cLLi, gMAP, gDAT)
"""
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ChunkType(Enum):
    """PNG chunk types for metadata"""
    # Text chunks
    TEXT = b'tEXt'      # Plain text (Latin-1)
    ZTXT = b'zTXt'      # Compressed text
    ITXT = b'iTXt'      # International text (UTF-8)
    
    # Standard metadata
    TIME = b'tIME'      # Last modification time
    PHYS = b'pHYs'      # Physical pixel dimensions
    GAMA = b'gAMA'      # Gamma
    CHRM = b'cHRM'      # Chromaticity coordinates
    SRGB = b'sRGB'      # sRGB color space
    ICCP = b'iCCP'      # ICC profile
    
    # HDR metadata chunks (W3C June 2025)
    CICP = b'cICP'      # Coding-independent code points
    MDCV = b'mDCv'      # Mastering Display Color Volume
    CLLI = b'cLLi'      # Content Light Level Information
    GMAP = b'gMAP'      # Gain Map
    GDAT = b'gDAT'      # Gain Map Data


@dataclass
class PNGChunk:
    """PNG chunk representation"""
    chunk_type: bytes
    data: bytes
    
    def __repr__(self):
        type_str = self.chunk_type.decode('latin-1', errors='ignore')
        return f"PNGChunk(type={type_str}, size={len(self.data)})"


@dataclass
class CICPData:
    """cICP chunk data - Coding-independent code points"""
    color_primaries: int        # 0-255
    transfer_characteristics: int  # 0-255
    matrix_coefficients: int    # 0-255
    video_full_range_flag: int  # 0 or 1
    
    # Common values:
    # color_primaries: 1=BT.709, 9=BT.2020
    # transfer_characteristics: 1=BT.709, 13=sRGB, 16=PQ (ST.2084), 18=HLG
    # matrix_coefficients: 0=Identity (RGB), 1=BT.709, 9=BT.2020


@dataclass
class MDCVData:
    """mDCv chunk data - Mastering Display Color Volume (SMPTE ST 2086)"""
    # Display primaries in 0.00002 units (x, y for R, G, B)
    display_primaries_x: Tuple[int, int, int]  # (R_x, G_x, B_x)
    display_primaries_y: Tuple[int, int, int]  # (R_y, G_y, B_y)
    
    # White point in 0.00002 units
    white_point_x: int
    white_point_y: int
    
    # Luminance in 0.0001 nits
    max_display_mastering_luminance: int  # Max luminance
    min_display_mastering_luminance: int  # Min luminance


@dataclass
class CLLIData:
    """cLLi chunk data - Content Light Level Information"""
    max_content_light_level: int      # MaxCLL in nits
    max_frame_average_light_level: int  # MaxFALL in nits


@dataclass
class GMAPData:
    """gMAP chunk data - Gain Map parameters"""
    version: int                    # Version (currently 0)
    gain_map_min: Tuple[int, int, int]  # Min gain values (R, G, B) in 1/256 units
    gain_map_max: Tuple[int, int, int]  # Max gain values (R, G, B) in 1/256 units
    gamma: Tuple[int, int, int]     # Gamma values (R, G, B) in 1/256 units
    base_offset: Tuple[int, int, int]  # Base offset (R, G, B) in 1/256 units
    alternate_offset: Tuple[int, int, int]  # Alternate offset (R, G, B) in 1/256 units


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
            
            # HDR chunks
            elif chunk.chunk_type == ChunkType.CICP.value:
                metadata['cicp'] = self._parse_cicp_chunk(chunk.data)
            
            elif chunk.chunk_type == ChunkType.MDCV.value:
                metadata['mdcv'] = self._parse_mdcv_chunk(chunk.data)
            
            elif chunk.chunk_type == ChunkType.CLLI.value:
                metadata['clli'] = self._parse_clli_chunk(chunk.data)
            
            elif chunk.chunk_type == ChunkType.GMAP.value:
                metadata['gmap'] = self._parse_gmap_chunk(chunk.data)
            
            elif chunk.chunk_type == ChunkType.GDAT.value:
                metadata['gdat'] = chunk.data  # Raw gain map data
        
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
    
    def set_hdr_metadata(self, 
                        cicp: Optional[CICPData] = None,
                        mdcv: Optional[MDCVData] = None,
                        clli: Optional[CLLIData] = None,
                        gmap: Optional[GMAPData] = None,
                        gdat: Optional[bytes] = None) -> None:
        """
        Set HDR metadata chunks
        
        Args:
            cicp: Coding-independent code points
            mdcv: Mastering Display Color Volume
            clli: Content Light Level Information
            gmap: Gain Map parameters
            gdat: Gain Map data (raw bytes)
        """
        # Remove existing HDR chunks
        self._chunks = [
            c for c in self._chunks 
            if c.chunk_type not in {
                ChunkType.CICP.value, ChunkType.MDCV.value, 
                ChunkType.CLLI.value, ChunkType.GMAP.value, ChunkType.GDAT.value
            }
        ]
        
        # Find position to insert (after IHDR, before IDAT)
        insert_pos = self._find_metadata_insert_position()
        
        # Create and insert chunks
        if cicp:
            chunk = self._create_cicp_chunk(cicp)
            self._chunks.insert(insert_pos, chunk)
            insert_pos += 1
        
        if mdcv:
            chunk = self._create_mdcv_chunk(mdcv)
            self._chunks.insert(insert_pos, chunk)
            insert_pos += 1
        
        if clli:
            chunk = self._create_clli_chunk(clli)
            self._chunks.insert(insert_pos, chunk)
            insert_pos += 1
        
        if gmap:
            chunk = self._create_gmap_chunk(gmap)
            self._chunks.insert(insert_pos, chunk)
            insert_pos += 1
        
        if gdat:
            chunk = PNGChunk(chunk_type=ChunkType.GDAT.value, data=gdat)
            self._chunks.insert(insert_pos, chunk)
            insert_pos += 1
    
    def _find_metadata_insert_position(self) -> int:
        """Finds position to insert metadata (after IHDR)"""
        for i, chunk in enumerate(self._chunks):
            if chunk.chunk_type == b'IHDR':
                return i + 1
        return 0
    
    # === Text chunk parsers ===
    
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
        compression_method = data[null_pos + 1]
        
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
        
        compression_flag = data[null_pos + 1]
        compression_method = data[null_pos + 2]
        
        rest = data[null_pos + 3:]
        lang_null = rest.find(b'\x00')
        if lang_null == -1:
            return keyword, ""
        
        rest = rest[lang_null + 1:]
        trans_null = rest.find(b'\x00')
        if trans_null == -1:
            return keyword, ""
        
        text_data = rest[trans_null + 1:]
        
        if compression_flag == 1:
            try:
                text_data = zlib.decompress(text_data)
            except:
                return keyword, ""
        
        text = text_data.decode('utf-8', errors='replace')
        return keyword, text
    
    # === Standard metadata parsers ===
    
    def _parse_time_chunk(self, data: bytes) -> Dict[str, int]:
        """Parse tIME chunk"""
        if len(data) < 7:
            return {}
        
        year = struct.unpack('>H', data[0:2])[0]
        month, day, hour, minute, second = struct.unpack('5B', data[2:7])
        
        return {
            'year': year, 'month': month, 'day': day,
            'hour': hour, 'minute': minute, 'second': second
        }
    
    def _parse_phys_chunk(self, data: bytes) -> Dict[str, str]:
        """Parse pHYs chunk"""
        if len(data) < 9:
            return {}
        
        pixels_per_unit_x = struct.unpack('>I', data[0:4])[0]
        pixels_per_unit_y = struct.unpack('>I', data[4:8])[0]
        unit = data[8]
        
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
    
    # === HDR chunk parsers ===
    
    def _parse_cicp_chunk(self, data: bytes) -> Dict[str, Any]:
        """Parse cICP chunk - Coding-independent code points"""
        if len(data) < 4:
            return {}
        
        color_primaries, transfer_char, matrix_coef, video_range = struct.unpack('4B', data[:4])
        
        # Human-readable names
        primaries_names = {
            1: 'BT.709', 9: 'BT.2020', 12: 'Display P3'
        }
        transfer_names = {
            1: 'BT.709', 13: 'sRGB', 16: 'PQ (ST.2084)', 18: 'HLG (BT.2100)'
        }
        matrix_names = {
            0: 'Identity (RGB)', 1: 'BT.709', 9: 'BT.2020'
        }
        
        return {
            'color_primaries': color_primaries,
            'color_primaries_name': primaries_names.get(color_primaries, f'Unknown ({color_primaries})'),
            'transfer_characteristics': transfer_char,
            'transfer_name': transfer_names.get(transfer_char, f'Unknown ({transfer_char})'),
            'matrix_coefficients': matrix_coef,
            'matrix_name': matrix_names.get(matrix_coef, f'Unknown ({matrix_coef})'),
            'video_full_range': bool(video_range)
        }
    
    def _parse_mdcv_chunk(self, data: bytes) -> Dict[str, Any]:
        """Parse mDCv chunk - Mastering Display Color Volume"""
        if len(data) < 24:
            return {}
        
        # All values are big-endian 16-bit unsigned integers
        values = struct.unpack('>12H', data[:24])
        
        # Display primaries (x, y for R, G, B) in 0.00002 units
        display_primaries = {
            'red': (values[0] * 0.00002, values[1] * 0.00002),
            'green': (values[2] * 0.00002, values[3] * 0.00002),
            'blue': (values[4] * 0.00002, values[5] * 0.00002)
        }
        
        # White point in 0.00002 units
        white_point = (values[6] * 0.00002, values[7] * 0.00002)
        
        # Luminance values in 0.0001 nits
        max_luminance = values[8] * 0.0001
        min_luminance = values[9] * 0.0001
        
        return {
            'display_primaries': display_primaries,
            'white_point': white_point,
            'max_luminance_nits': max_luminance,
            'min_luminance_nits': min_luminance
        }
    
    def _parse_clli_chunk(self, data: bytes) -> Dict[str, int]:
        """Parse cLLi chunk - Content Light Level Information"""
        if len(data) < 4:
            return {}
        
        max_cll, max_fall = struct.unpack('>2H', data[:4])
        
        return {
            'max_content_light_level': max_cll,
            'max_frame_average_light_level': max_fall
        }
    
    def _parse_gmap_chunk(self, data: bytes) -> Dict[str, Any]:
        """Parse gMAP chunk - Gain Map parameters"""
        if len(data) < 37:  # 1 + 3*3*4 bytes
            return {}
        
        version = data[0]
        
        # Parse 9 int32 values: min(RGB), max(RGB), gamma(RGB)
        values = struct.unpack('>9i', data[1:37])
        
        # Parse 6 more int32 values if present: base_offset(RGB), alt_offset(RGB)
        if len(data) >= 61:
            offsets = struct.unpack('>6i', data[37:61])
        else:
            offsets = (0, 0, 0, 0, 0, 0)
        
        # Convert to float (stored as 1/256 units)
        def to_float(val):
            return val / 256.0
        
        return {
            'version': version,
            'gain_map_min': tuple(to_float(v) for v in values[0:3]),
            'gain_map_max': tuple(to_float(v) for v in values[3:6]),
            'gamma': tuple(to_float(v) for v in values[6:9]),
            'base_offset': tuple(to_float(v) for v in offsets[0:3]),
            'alternate_offset': tuple(to_float(v) for v in offsets[3:6])
        }
    
    # === Text chunk creators ===
    
    def _create_text_chunk(self, keyword: str, text: str) -> PNGChunk:
        """Create tEXt chunk"""
        keyword_bytes = keyword.encode('latin-1')[:79]
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
    
    # === HDR chunk creators ===
    
    def _create_cicp_chunk(self, cicp: CICPData) -> PNGChunk:
        """Create cICP chunk"""
        data = struct.pack('4B',
            cicp.color_primaries,
            cicp.transfer_characteristics,
            cicp.matrix_coefficients,
            cicp.video_full_range_flag
        )
        return PNGChunk(chunk_type=ChunkType.CICP.value, data=data)
    
    def _create_mdcv_chunk(self, mdcv: MDCVData) -> PNGChunk:
        """Create mDCv chunk"""
        data = struct.pack('>12H',
            mdcv.display_primaries_x[0], mdcv.display_primaries_y[0],  # Red
            mdcv.display_primaries_x[1], mdcv.display_primaries_y[1],  # Green
            mdcv.display_primaries_x[2], mdcv.display_primaries_y[2],  # Blue
            mdcv.white_point_x, mdcv.white_point_y,
            mdcv.max_display_mastering_luminance,
            mdcv.min_display_mastering_luminance,
            0, 0  # Reserved
        )
        return PNGChunk(chunk_type=ChunkType.MDCV.value, data=data)
    
    def _create_clli_chunk(self, clli: CLLIData) -> PNGChunk:
        """Create cLLi chunk"""
        data = struct.pack('>2H',
            clli.max_content_light_level,
            clli.max_frame_average_light_level
        )
        return PNGChunk(chunk_type=ChunkType.CLLI.value, data=data)
    
    def _create_gmap_chunk(self, gmap: GMAPData) -> PNGChunk:
        """Create gMAP chunk"""
        # Convert float to int (stored as 1/256 units)
        def to_int(val):
            return int(val * 256)
        
        data = struct.pack('>B9i6i',
            gmap.version,
            to_int(gmap.gain_map_min[0]), to_int(gmap.gain_map_min[1]), to_int(gmap.gain_map_min[2]),
            to_int(gmap.gain_map_max[0]), to_int(gmap.gain_map_max[1]), to_int(gmap.gain_map_max[2]),
            to_int(gmap.gamma[0]), to_int(gmap.gamma[1]), to_int(gmap.gamma[2]),
            to_int(gmap.base_offset[0]), to_int(gmap.base_offset[1]), to_int(gmap.base_offset[2]),
            to_int(gmap.alternate_offset[0]), to_int(gmap.alternate_offset[1]), to_int(gmap.alternate_offset[2])
        )
        return PNGChunk(chunk_type=ChunkType.GMAP.value, data=data)
    
    # === Utility methods ===
    
    def copy_image_with_metadata(self, source: str, destination: str, 
                                 metadata: Dict[str, str]) -> None:
        """
        Copy PNG with adding/updating metadata
        
        Args:
            source: Source PNG file
            destination: Destination file
            metadata: Metadata to add
        """
        self.read_chunks(source)
        self.set_metadata(metadata)
        self.write_chunks(destination)
    
    def list_chunks(self) -> None:
        """Print list of all chunks for debugging"""
        print(f"Total chunks: {len(self._chunks)}")
        for i, chunk in enumerate(self._chunks):
            type_str = chunk.chunk_type.decode('latin-1', errors='ignore')
            critical = "critical" if chunk.chunk_type in self.CRITICAL_CHUNKS else "ancillary"
            print(f"{i:3d}. {type_str:4s} - {len(chunk.data):8d} bytes ({critical})")