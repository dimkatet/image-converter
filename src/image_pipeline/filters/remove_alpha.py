import numpy as np

from .base import ImageFilter


class RemoveAlphaFilter(ImageFilter):
    """
    Filter that removes the alpha channel from an image.
    If there is no alpha channel, returns the image unchanged.
    """

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        self.validate(pixels)

        # If less than 4 channels, nothing to remove
        if pixels.ndim < 3 or pixels.shape[-1] < 4:
            return pixels

        # Remove the last channel (A)
        rgb = pixels[..., :3]

        return rgb.astype(pixels.dtype)
    
    #  todo
    # def update_metadata(self, img_data: ImageData) -> None:
    #     super().update_metadata(img_data)
    #     img_data.metadata['has_alpha'] = False
    #     img_data.metadata['color_mode'] = 'RGB'

    def __repr__(self) -> str:
        return "RemoveAlphaFilter()"
