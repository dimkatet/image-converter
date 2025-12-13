"""
Базовый класс для метаданных и примеры использования
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class MetadataCodec(ABC):
    """Базовый класс для кодеков метаданных"""
    
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = Path(filepath) if filepath else None
    
    @abstractmethod
    def read_metadata(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Чтение метаданных из файла
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            Словарь с метаданными
        """
        pass
    
    @abstractmethod
    def write_metadata(self, metadata: Dict[str, Any], 
                      source: str, destination: str) -> None:
        """
        Запись метаданных в файл
        
        Args:
            metadata: Метаданные для записи
            source: Исходный файл
            destination: Файл назначения
        """
        pass
    
    @abstractmethod
    def update_metadata(self, filepath: str, 
                       metadata: Dict[str, Any]) -> None:
        """
        Обновление метаданных в существующем файле
        
        Args:
            filepath: Путь к файлу
            metadata: Метаданные для обновления
        """
        pass
