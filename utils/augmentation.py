import random
import torch
from torchvision.transforms.functional import get_dimensions


class DonutShift(torch.nn.Module):
    """Shift the image with periodic boundary conditions."""
    def __init__(self, max_shift_h=None, max_shift_w=None):
        self.max_shift_x = max_shift_h
        self.max_shift_y = max_shift_w

    def __call__(self, img):
        _, h,w = get_dimensions(img)
        max_shift_h = h if self.max_shift_x is None else self.max_shift_x
        max_shift_w = w if self.max_shift_y is None else self.max_shift_y

        shift_x = random.randint(0, max_shift_h)
        shift_y = random.randint(0, max_shift_w)

        return torch.roll(img, shifts=(shift_x, shift_y), dims=(1, 2))