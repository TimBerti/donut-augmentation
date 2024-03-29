import random
import torch
from torchvision.transforms.functional import get_dimensions


class DonutShift():
    """Shift the image with periodic boundary conditions.
    
    Args:
        max_shift_h (int, optional): Maximum shift in the height direction. Defaults to None.
        max_shift_w (int, optional): Maximum shift in the width direction. Defaults to None.

    Returns:
        torch.Tensor: Shifted image.

    Example:
        >>> img = torch.rand(3, 32, 32)
        >>> shift = DonutShift(max_shift_h=4, max_shift_w=4)
        >>> img_shifted = shift(img)
    """
    def __init__(self, max_shift_h=None, max_shift_w=None):
        self.max_shift_h = max_shift_h
        self.max_shift_w = max_shift_w

    def __call__(self, img):
        _, h,w = get_dimensions(img)
        max_shift_h = h if self.max_shift_h is None else self.max_shift_h
        max_shift_w = w if self.max_shift_w is None else self.max_shift_w

        shift_h = random.randint(0, max_shift_h)
        shift_w = random.randint(0, max_shift_w)

        return torch.roll(img, shifts=(shift_h, shift_w), dims=(1, 2))
    
class PaddedShift():
    """Shift the image with zero padding.
    
    Args:
        max_shift_h (int, optional): Maximum shift in the height direction. Defaults to None.
        max_shift_w (int, optional): Maximum shift in the width direction. Defaults to None.

    Returns:
        torch.Tensor: Shifted image.

    Example:
        >>> img = torch.rand(3, 32, 32)
        >>> shift = PaddedShift(max_shift_h=4, max_shift_w=4)
        >>> img_shifted = shift(img)
    """
    def __init__(self, max_shift_h=None, max_shift_w=None):
        self.max_shift_h = max_shift_h
        self.max_shift_w = max_shift_w

    def __call__(self, img):
        _, h, w = get_dimensions(img)
        max_shift_h = h if self.max_shift_h is None else self.max_shift_h
        max_shift_w = w if self.max_shift_w is None else self.max_shift_w

        shift_h = random.randint(-max_shift_h, max_shift_h)
        shift_w = random.randint(-max_shift_w, max_shift_w)

        output = torch.zeros_like(img)

        start_h = max(shift_h, 0)
        end_h = h + min(shift_h, 0)
        start_w = max(shift_w, 0)
        end_w = w + min(shift_w, 0)

        orig_start_h = -min(shift_h, 0)
        orig_end_h = h - max(shift_h, 0)
        orig_start_w = -min(shift_w, 0)
        orig_end_w = w - max(shift_w, 0)

        output[:, start_h:end_h, start_w:end_w] = img[:, orig_start_h:orig_end_h, orig_start_w:orig_end_w]

        return output

