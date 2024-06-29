from functools import wraps
from typing import Callable

import numpy as np
import xarray as xr
from typing_extensions import Any, Concatenate

from ..types import RT, ImageType, P
from .wrapper import GenericWrapper


def is_image_type(X: Any) -> bool:
    # Feature array images must have exactly 3 dimensions: (y, x, band) or (band, y, x)
    if isinstance(X, (np.ndarray, xr.DataArray)):
        return X.ndim == 3

    # Feature Dataset images must have exactly 2 dimensions: (x, y)
    if isinstance(X, xr.Dataset):
        return len(X.dims) == 2

    return False


def image_or_fallback(
    func: Callable[Concatenate[GenericWrapper, ImageType, P], RT],
) -> Callable[Concatenate[GenericWrapper, ImageType, P], RT]:
    """Decorator that calls the wrapped method for non-image X arrays."""

    @wraps(func)
    def wrapper(self: GenericWrapper, X_image: ImageType, *args, **kwargs):
        if not is_image_type(X_image):
            return getattr(self._wrapped, func.__name__)(X_image, *args, **kwargs)

        return func(self, X_image, *args, **kwargs)

    return wrapper
