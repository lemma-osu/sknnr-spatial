from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import xarray as xr

from ._base import ImagePreprocessor
from ._dask_backed import DaskBackedWrapper

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..types import NoDataType


class DataArrayPreprocessor(ImagePreprocessor):
    """Pre-processor for multi-band xr.DataArrays."""

    _backend = da
    band_dim = 0

    @property
    def band_names(self) -> NDArray:
        band_dim_name = self.image.dims[self.band_dim]
        return self.image[band_dim_name].values

    def _validate_nodata_vals(self, nodata_vals: NoDataType) -> NDArray | None:
        """
        Get an array of NoData values in the shape (bands,) based on user input and
        DataArray metadata.
        """
        # Defer to user-provided NoData values over stored attributes
        if nodata_vals is not None:
            return super()._validate_nodata_vals(nodata_vals)

        # If present, broadcast the _FillValue attribute to all bands
        fill_val = self.image.attrs.get("_FillValue")
        if fill_val is not None:
            return np.full((self.n_bands,), fill_val)

        return None

    def _flatten(self, image: xr.DataArray) -> xr.DataArray:
        """Flatten the dataarray from (bands, y, x) to (pixels, bands)."""
        # Dask can't reshape multiple dimensions at once, so transpose to swap axes
        return image.data.reshape(self.n_bands, -1).T

    def unflatten(
        self,
        flat_image: xr.DataArray,
        *,
        apply_mask=True,
        var_names=None,
    ) -> xr.DataArray:
        if apply_mask:
            flat_image = self._fill_nodata(flat_image, np.nan)

        n_outputs = flat_image.shape[self.flat_band_dim]
        # Default the variable coordinate to sequential numbers if not provided
        var_names = var_names if var_names is not None else range(n_outputs)

        # Replace the original variable coordinates and dimensions
        band_dim_name = self.image.dims[self.band_dim]
        dims = {**self.image.sizes, band_dim_name: n_outputs}
        coords = {**self.image.coords, band_dim_name: var_names}
        shape = list(dims.values())

        return xr.DataArray(
            # Transpose the flat image from (pixels, bands) to (bands, pixels) prior
            # to reshaping to match the expected output.
            flat_image.T.reshape(shape),
            coords=coords,
            dims=dims,
        )


class DataArrayWrapper(DaskBackedWrapper[xr.DataArray]):
    """A wrapper around a DataArray that provides sklearn methods."""

    preprocessor_cls = DataArrayPreprocessor
