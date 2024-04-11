from __future__ import annotations

import dask.array as da
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor

from ._base import ImagePreprocessor, kneighbors, predict
from ._dask_backed import (
    _kneighbors_from_dask_backed_array,
    _predict_from_dask_backed_array,
)


class DataArrayPreprocessor(ImagePreprocessor):
    """
    Pre-processor for multi-band xr.DataArrays.
    """

    _backend = da
    band_dim = 0

    def _validate_nodata_vals(
        self, nodata_vals: float | tuple[float] | NDArray | None
    ) -> NDArray | None:
        """
        Get an array of NoData values in the shape (bands,) based on user input and
        DataArray metadata.
        """
        # Defer to user-provided nodata values over stored attributes
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
        dims = {**self.image.sizes, "variable": n_outputs}
        coords = {**self.image.coords, "variable": var_names}
        shape = list(dims.values())

        return xr.DataArray(
            # Transpose the flat image from (pixels, bands) to (bands, pixels) prior
            # to reshaping to match the expected output.
            flat_image.T.reshape(shape),
            coords=coords,
            dims=dims,
        )


@predict.register(xr.DataArray)
def _predict_from_dataarray(
    X_image: xr.DataArray, *, estimator: BaseEstimator, y, nodata_vals=None
) -> xr.DataArray:
    return _predict_from_dask_backed_array(
        X_image,
        estimator=estimator,
        y=y,
        nodata_vals=nodata_vals,
        preprocessor_cls=DataArrayPreprocessor,
    )


@kneighbors.register(xr.DataArray)
def _kneighbors_from_dataarray(
    X_image: xr.Dataset,
    *,
    estimator: KNeighborsRegressor,
    nodata_vals=None,
    **kneighbors_kwargs,
) -> xr.Dataset:
    return _kneighbors_from_dask_backed_array(
        X_image,
        estimator=estimator,
        nodata_vals=nodata_vals,
        preprocessor_cls=DataArrayPreprocessor,
        **kneighbors_kwargs,
    )