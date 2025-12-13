"""
PNG Metadata Codec - для чтения и записи PNG chunks
Поддерживает текстовые chunks (tEXt, zTXt, iTXt) и другие метаданные
"""
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ChunkType(Enum):
    """Типы PNG chunks для метаданных"""
    TEXT = b'tEXt'      # Простой текст (Latin-1)
    ZTXT = b'zTXt'      # Сжатый текст
    ITXT = b'iTXt'      # Международный текст (UTF-8)
    TIME = b'tIME'      # Время последнего изменения
    PHYS = b'pHYs'      # Физические размеры пикселя
    GAMA = b'gAMA'      # Гамма
    CHRM = b'cHRM'      # Хроматические координаты
    SRGB = b'sRGB'      # sRGB цветовое пространство
    ICCP = b'iCCP'      # ICC профиль


@dataclass
class PNGChunk:
    """Представление PNG chunk"""
    chunk_type: bytes
    data: bytes
    
    def __repr__(self):
        type_str = self.chunk_type.decode('latin-1', errors='ignore')
        return f"PNGChunk(type={type_str}, size={len(self.data)})"


class PNGMetadataCodec:
    """Кодек для работы с PNG метаданными через chunks"""
    
    PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
    
    # Критичные chunks, которые нельзя удалять
    CRITICAL_CHUNKS = {b'IHDR', b'PLTE', b'IDAT', b'IEND'}
    
    def __init__(self, filepath: Optional[str] = None):
        """
        Args:
            filepath: Путь к PNG файлу (опционально)
        """
        self.filepath = Path(filepath) if filepath else None
        self._chunks: List[PNGChunk] = []
    
    def read_chunks(self, filepath: Optional[str] = None) -> List[PNGChunk]:
        """
        Чтение всех chunks из PNG файла
        
        Args:
            filepath: Путь к файлу (или используется self.filepath)
            
        Returns:
            Список PNGChunk объектов
        """
        path = Path(filepath) if filepath else self.filepath
        if not path:
            raise ValueError("Filepath not specified")
        
        self._chunks = []
        
        with open(path, 'rb') as f:
            # Проверка PNG сигнатуры
            signature = f.read(8)
            if signature != self.PNG_SIGNATURE:
                raise ValueError(f"Not a valid PNG file: {path}")
            
            # Чтение chunks
            while True:
                chunk = self._read_chunk(f)
                if chunk is None:
                    break
                self._chunks.append(chunk)
                
                # IEND - конец файла
                if chunk.chunk_type == b'IEND':
                    break
        
        return self._chunks
    
    def _read_chunk(self, f) -> Optional[PNGChunk]:
        """Чтение одного chunk из файла"""
        # Длина данных (4 байта)
        length_bytes = f.read(4)
        if len(length_bytes) < 4:
            return None
        
        length = struct.unpack('>I', length_bytes)[0]
        
        # Тип chunk (4 байта)
        chunk_type = f.read(4)
        if len(chunk_type) < 4:
            return None
        
        # Данные chunk
        data = f.read(length)
        if len(data) < length:
            return None
        
        # CRC (4 байта) - пропускаем, но проверяем
        crc = f.read(4)
        if len(crc) < 4:
            return None
        
        # Проверка CRC
        expected_crc = struct.unpack('>I', crc)[0]
        calculated_crc = zlib.crc32(chunk_type + data) & 0xffffffff
        if expected_crc != calculated_crc:
            print(f"Warning: CRC mismatch for chunk {chunk_type}")
        
        return PNGChunk(chunk_type=chunk_type, data=data)
    
    def write_chunks(self, filepath: str, chunks: Optional[List[PNGChunk]] = None) -> None:
        """
        Запись chunks в PNG файл
        
        Args:
            filepath: Путь для сохранения
            chunks: Список chunks (или используется self._chunks)
        """
        chunks_to_write = chunks if chunks is not None else self._chunks
        
        if not chunks_to_write:
            raise ValueError("No chunks to write")
        
        with open(filepath, 'wb') as f:
            # PNG сигнатура
            f.write(self.PNG_SIGNATURE)
            
            # Запись всех chunks
            for chunk in chunks_to_write:
                self._write_chunk(f, chunk)
    
    def _write_chunk(self, f, chunk: PNGChunk) -> None:
        """Запись одного chunk в файл"""
        # Длина данных
        f.write(struct.pack('>I', len(chunk.data)))
        
        # Тип chunk
        f.write(chunk.chunk_type)
        
        # Данные
        f.write(chunk.data)
        
        # CRC
        crc = zlib.crc32(chunk.chunk_type + chunk.data) & 0xffffffff
        f.write(struct.pack('>I', crc))
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Извлечение метаданных из chunks в удобный формат
        
        Returns:
            Словарь с метаданными
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
        Установка текстовых метаданных (добавление tEXt/zTXt chunks)
        
        Args:
            metadata: Словарь ключ-значение для метаданных
            compress: Использовать сжатие (zTXt вместо tEXt)
        """
        # Удаляем существующие текстовые chunks
        self._chunks = [
            c for c in self._chunks 
            if c.chunk_type not in {ChunkType.TEXT.value, ChunkType.ZTXT.value, ChunkType.ITXT.value}
        ]
        
        # Находим позицию для вставки (после IHDR, до IDAT)
        insert_pos = self._find_metadata_insert_position()
        
        # Создаём новые chunks
        for key, value in metadata.items():
            if compress:
                chunk = self._create_ztxt_chunk(key, value)
            else:
                chunk = self._create_text_chunk(key, value)
            
            self._chunks.insert(insert_pos, chunk)
            insert_pos += 1
    
    def _find_metadata_insert_position(self) -> int:
        """Находит позицию для вставки метаданных (после IHDR)"""
        for i, chunk in enumerate(self._chunks):
            if chunk.chunk_type == b'IHDR':
                return i + 1
        return 0
    
    def _parse_text_chunk(self, data: bytes) -> Tuple[str, str]:
        """Парсинг tEXt chunk"""
        null_pos = data.find(b'\x00')
        if null_pos == -1:
            return "", ""
        
        keyword = data[:null_pos].decode('latin-1')
        text = data[null_pos + 1:].decode('latin-1', errors='replace')
        return keyword, text
    
    def _parse_ztxt_chunk(self, data: bytes) -> Tuple[str, str]:
        """Парсинг zTXt chunk (сжатый текст)"""
        null_pos = data.find(b'\x00')
        if null_pos == -1:
            return "", ""
        
        keyword = data[:null_pos].decode('latin-1')
        compression_method = data[null_pos + 1]  # Должен быть 0 (deflate)
        
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
        """Парсинг iTXt chunk (международный текст, UTF-8)"""
        null_pos = data.find(b'\x00')
        if null_pos == -1:
            return "", ""
        
        keyword = data[:null_pos].decode('latin-1')
        
        # Пропускаем compression flag и method
        compression_flag = data[null_pos + 1]
        compression_method = data[null_pos + 2]
        
        # Находим language tag и translated keyword (оба null-terminated)
        rest = data[null_pos + 3:]
        lang_null = rest.find(b'\x00')
        if lang_null == -1:
            return keyword, ""
        
        rest = rest[lang_null + 1:]
        trans_null = rest.find(b'\x00')
        if trans_null == -1:
            return keyword, ""
        
        text_data = rest[trans_null + 1:]
        
        # Декомпрессия если нужно
        if compression_flag == 1:
            try:
                text_data = zlib.decompress(text_data)
            except:
                return keyword, ""
        
        text = text_data.decode('utf-8', errors='replace')
        return keyword, text
    
    def _parse_time_chunk(self, data: bytes) -> Dict[str, int]:
        """Парсинг tIME chunk"""
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
        """Парсинг pHYs chunk"""
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
        """Парсинг gAMA chunk"""
        if len(data) < 4:
            return 0.0
        
        gamma_int = struct.unpack('>I', data)[0]
        return gamma_int / 100000.0
    
    def _create_text_chunk(self, keyword: str, text: str) -> PNGChunk:
        """Создание tEXt chunk"""
        keyword_bytes = keyword.encode('latin-1')[:79]  # Max 79 bytes
        text_bytes = text.encode('latin-1', errors='replace')
        
        data = keyword_bytes + b'\x00' + text_bytes
        return PNGChunk(chunk_type=ChunkType.TEXT.value, data=data)
    
    def _create_ztxt_chunk(self, keyword: str, text: str) -> PNGChunk:
        """Создание zTXt chunk (сжатый)"""
        keyword_bytes = keyword.encode('latin-1')[:79]
        text_bytes = text.encode('latin-1', errors='replace')
        compressed = zlib.compress(text_bytes, level=9)
        
        data = keyword_bytes + b'\x00' + b'\x00' + compressed
        return PNGChunk(chunk_type=ChunkType.ZTXT.value, data=data)
    
    def copy_image_with_metadata(self, source: str, destination: str, 
                                 metadata: Dict[str, str]) -> None:
        """
        Копирование PNG с добавлением/изменением метаданных
        
        Args:
            source: Исходный PNG файл
            destination: Файл назначения
            metadata: Метаданные для добавления
        """
        # Читаем chunks из исходного файла
        self.read_chunks(source)
        
        # Устанавливаем новые метаданные
        self.set_metadata(metadata)
        
        # Сохраняем
        self.write_chunks(destination)
    
    def list_chunks(self) -> None:
        """Вывод списка всех chunks для отладки"""
        print(f"Total chunks: {len(self._chunks)}")
        for i, chunk in enumerate(self._chunks):
            type_str = chunk.chunk_type.decode('latin-1', errors='ignore')
            critical = "critical" if chunk.chunk_type in self.CRITICAL_CHUNKS else "ancillary"
            print(f"{i:3d}. {type_str:4s} - {len(chunk.data):8d} bytes ({critical})")