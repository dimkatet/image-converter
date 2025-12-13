import numpy as np
from .base import ImageFilter


class RemoveAlphaFilter(ImageFilter):
    """
    Фильтр, который удаляет альфа-канал из изображения.
    Если альфа-канала нет — возвращает изображение без изменений.
    """

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)

        # Если меньше 4 каналов — нечего удалять
        if pixels.ndim < 3 or pixels.shape[-1] < 4:
            return pixels

        # Отрезаем последний канал (A)
        rgb = pixels[..., :3]

        return rgb.astype(pixels.dtype)

    def __repr__(self) -> str:
        return "RemoveAlphaFilter()"
