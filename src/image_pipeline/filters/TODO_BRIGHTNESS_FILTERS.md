# TODO: Brightness/Contrast Filters

When implementing brightness, contrast, or other filters that modify pixel luminance values, ensure they update MaxCLL/MaxFALL metadata if the data is display-referred (absolute nits).

## Required filters to implement:

1. **BrightnessFilter** - Adjust brightness by scaling pixel values
2. **ContrastFilter** - Adjust contrast around midpoint
3. **ExposureFilter** - Adjust exposure (log-space brightness)
4. **GammaFilter** - Apply gamma correction

## Example implementation pattern:

```python
class BrightnessFilter(ImageFilter):
    def __init__(self, factor: float = 1.0):
        self.factor = factor
        super().__init__()

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        return pixels * self.factor

    def update_metadata(self, img_data: ImageData) -> None:
        super().update_metadata(img_data)

        # Update MaxCLL/MaxFALL if data is display-referred (has these fields)
        if 'max_cll' in img_data.metadata:
            img_data.metadata['max_cll'] = int(round(img_data.pixels.max()))
        if 'max_fall' in img_data.metadata:
            img_data.metadata['max_fall'] = int(round(img_data.pixels.mean()))
```

## Important notes:

- Only update MaxCLL/MaxFALL if they already exist in metadata (indicates display-referred data)
- MaxCLL = max luminance in nits (int)
- MaxFALL = average frame luminance in nits (int)
- These are HDR10 signaling metadata, only valid for absolute luminance values
