import numpy as np
from .base import ImageFilter


class PQEncodeFilter(ImageFilter):
    """
    Фильтр для применения PQ (Perceptual Quantizer) гамма-кривой по стандарту ST.2084
    Преобразует линейные HDR значения в PQ-кодированные значения
    """
    
    # Константы PQ по стандарту ITU-R BT.2100
    M1 = 2610.0 / 16384.0  # 0.1593017578125
    M2 = 2523.0 / 4096.0 * 128.0  # 78.84375
    C1 = 3424.0 / 4096.0  # 0.8359375
    C2 = 2413.0 / 4096.0 * 32.0  # 18.8515625
    C3 = 2392.0 / 4096.0 * 32.0  # 18.6875
    
    def __init__(self, peak_luminance: float = 10000.0):
        """
        Args:
            peak_luminance: Пиковая яркость в нитах (cd/m²)
                           Обычно 10000 для ST.2084, но можно настроить
        """
        super().__init__()
        self.peak_luminance = peak_luminance
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)
        
        # Нормализуем к диапазону [0, 1] относительно пиковой яркости
        # Предполагаем, что входные данные в линейном пространстве
        normalized = pixels / self.peak_luminance
        
        # Обрезаем отрицательные значения
        normalized = np.maximum(normalized, 0.0)
        
        # Применяем PQ EOTF
        # Y = ((c1 + c2 * L^m1) / (1 + c3 * L^m1))^m2
        L_m1 = np.power(normalized, self.M1)
        
        numerator = self.C1 + self.C2 * L_m1
        denominator = 1.0 + self.C3 * L_m1
        
        # Избегаем деления на ноль
        denominator = np.maximum(denominator, 1e-10)
        
        pq_encoded = np.power(numerator / denominator, self.M2)
        
        return pq_encoded.astype(np.float32)
    
    def __repr__(self) -> str:
        return f"PQEncodeFilter(peak_luminance={self.peak_luminance})"
