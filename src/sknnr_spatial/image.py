from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sized
from typing import Any, Callable, Generic

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from typing_extensions import Concatenate

from .types import ImageType, NoDataType, P


class _ImageChunk:
    """A chunk of an NDArray in shape (y, x, band)."""

    def __init__(
        self, array: NDArray, nodata_vals: list[float] | None = None, nan_fill=0.0
    ):
        self.array = array
        self.flat_array = self._preprocess(array, nan_fill=nan_fill)
        self.nodata_vals = nodata_vals

    def _mask_nodata(self, flat_image: NDArray) -> NDArray:
        """
        Set NaNs in the flat image where NoData values are present.
        """
        # Skip allocating a mask if the image is float and NoData wasn't given
        if (
            not (is_float := self.flat_array.dtype.kind == "f")
            and self.nodata_vals is None
        ):
            return flat_image

        mask = np.zeros(self.flat_array.shape, dtype=bool)
        flat_image = flat_image.astype(np.float64)

        # If it's floating point, always mask NaNs
        if is_float:
            mask |= np.isnan(self.flat_array)

        # If NoData was specified, mask those values
        if self.nodata_vals is not None:
            mask |= self.flat_array == self.nodata_vals

        # Set the mask where any band contains NoData
        flat_image[mask.max(axis=-1)] = np.nan

        return flat_image

    def _preprocess(self, array: NDArray, nan_fill: float = 0.0) -> NDArray:
        """Preprocess the chunk by flattening to (pixels, bands) and filling NaNs."""
        flat = array.reshape(-1, array.shape[-1])
        if nan_fill is not None:
            flat[np.isnan(flat)] = nan_fill

        return flat

    def _postprocess(self, array: NDArray, mask_nodata: bool = True) -> NDArray:
        """Postprocess the chunk by unflattening to (y, x, band) and masking NoData."""
        output_shape = [*self.array.shape[:2], -1]
        if mask_nodata:
            array = self._mask_nodata(array)

        return array.reshape(output_shape)

    def apply(
        self, func, returns_tuple=False, mask_nodata=True, **kwargs
    ) -> NDArray | tuple[NDArray]:
        """
        Apply a function to the flattened, processed chunk.

        The function should accept and return one or more NDArrays in shape
        (pixels, bands). The output will be reshaped back to the original chunk shape.
        """
        flat_result = func(self.flat_array, **kwargs)

        if returns_tuple:
            return tuple(
                self._postprocess(result, mask_nodata=mask_nodata)
                for result in flat_result
            )

        return self._postprocess(flat_result, mask_nodata=mask_nodata)


class Image(Generic[ImageType], ABC):
    """A wrapper around a multi-band image"""

    band_dim_name: str
    band_dim: int
    band_names: NDArray

    def __init__(self, image: ImageType, nodata_vals: NoDataType = None):
        self.image = image
        self.n_bands = self.image.shape[self.band_dim]
        self.nodata_vals = self._validate_nodata_vals(nodata_vals)

    def _validate_nodata_vals(self, nodata_vals: NoDataType) -> NDArray | None:
        """
        Get an array of NoData values in the shape (bands,) based on user input.

        Scalars are broadcast to all bands while sequences are checked against the
        number of bands and cast to ndarrays. There is no need to specify np.nan as a
        NoData value because it will be masked automatically for floating point images.
        """
        if nodata_vals is None:
            return None

        # If it's a numeric scalar, broadcast it to all bands
        if isinstance(nodata_vals, (float, int)) and not isinstance(nodata_vals, bool):
            return np.full((self.n_bands,), nodata_vals)

        # If it's not a scalar, it must be an iterable
        if not isinstance(nodata_vals, Sized) or isinstance(nodata_vals, (str, dict)):
            raise TypeError(
                f"Invalid type `{type(nodata_vals).__name__}` for `nodata_vals`. "
                "Provide a single number to apply to all bands, a sequence of numbers, "
                "or None."
            )

        # If it's an iterable, it must contain one element per band
        if len(nodata_vals) != self.n_bands:
            raise ValueError(
                f"Expected {self.n_bands} NoData values but got {len(nodata_vals)}. "
                f"The length of `nodata_vals` must match the number of bands."
            )

        return np.asarray(nodata_vals, dtype=float)

    @abstractmethod
    def apply_ufunc_across_bands(
        self,
        func: Callable[Concatenate[NDArray, P], NDArray],
        *,
        output_dims: list[list[str]] | None = None,
        output_dtypes: list[np.dtype] | None = None,
        output_sizes: dict[str, int] | None = None,
        output_coords: dict[str, list[str | int]] | None = None,
        nan_fill: float = 0.0,
        mask_nodata: bool = True,
        **ufunc_kwargs,
    ) -> ImageType | tuple[ImageType]:
        """
        Apply a universal function to all bands of the image.

        If the image is backed by a Dask array, the computation will be parallelized
        across spatial chunks.
        """

    @staticmethod
    def from_image(image: Any, nodata_vals: NoDataType = None) -> Image:
        """Create an Image object from a supported image type."""
        if isinstance(image, np.ndarray):
            return NDArrayImage(image, nodata_vals=nodata_vals)

        if isinstance(image, xr.DataArray):
            return DataArrayImage(image, nodata_vals=nodata_vals)

        if isinstance(image, xr.Dataset):
            return DatasetImage(image, nodata_vals=nodata_vals)

        raise TypeError(f"Unsupported image type `{type(image).__name__}`.")


class NDArrayImage(Image):
    band_dim = -1
    band_names = np.array([])

    def __init__(self, image: NDArray, nodata_vals: NoDataType = None):
        super().__init__(image, nodata_vals=nodata_vals)

    def apply_ufunc_across_bands(
        self,
        func: Callable[Concatenate[NDArray, P], NDArray],
        *,
        output_dims: list[list[str]] | None = None,
        output_dtypes: list[np.dtype] | None = None,
        output_sizes: dict[str, int] | None = None,
        output_coords: dict[str, list[str | int]] | None = None,
        nan_fill: float = 0.0,
        mask_nodata: bool = True,
        **ufunc_kwargs,
    ) -> NDArray | tuple[NDArray]:
        n_outputs = len(output_dims) if output_dims is not None else 1

        return _ImageChunk(
            self.image, nodata_vals=self.nodata_vals, nan_fill=nan_fill
        ).apply(
            func,
            returns_tuple=n_outputs > 1,
            mask_nodata=mask_nodata,
            **ufunc_kwargs,
        )


class DataArrayImage(Image):
    band_dim = 0

    def __init__(self, image: xr.DataArray, nodata_vals: NoDataType = None):
        super().__init__(image, nodata_vals=nodata_vals)
        self.band_dim_name = image.dims[self.band_dim]

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

    def _postprocess(
        self,
        result: xr.DataArray,
        output_coords: dict[str, list[str | int]],
    ) -> xr.DataArray:
        """Process the output of an applied ufunc"""
        if output_coords is not None:
            result = result.assign_coords(output_coords)
        var_dim = list(output_coords.keys())[0]

        # apply_gufunc swaps dimension order, so we need to restore it back to
        # (band, y, x).
        return result.transpose(var_dim, ...)

    def apply_ufunc_across_bands(
        self,
        func: Callable[Concatenate[NDArray, P], NDArray],
        *,
        output_dims: list[list[str]] | None = None,
        output_dtypes: list[np.dtype] | None = None,
        output_sizes: dict[str, int] | None = None,
        output_coords: dict[str, list[str | int]] | None = None,
        nan_fill: float = 0.0,
        mask_nodata: bool = True,
        **ufunc_kwargs,
    ) -> xr.DataArray | tuple[xr.DataArray]:
        """
        Apply a universal function to all bands of the image.

        If the image is backed by a Dask array, the computation will be parallelized
        across spatial chunks.
        """
        image = self.image

        output_dims = output_dims or [["variable"]]
        n_outputs = len(output_dims)
        # Fall back to float output if unknown
        output_dtypes = output_dtypes or [np.float32] * n_outputs
        # If output sizes are not provided, assume a single output coordinate
        output_sizes = output_sizes or {"variable": 1}
        # Default to sequential coordinates for each output dimension, if not provided
        output_coords = output_coords or {
            k: list(range(s)) for k, s in output_sizes.items()
        }

        def ufunc(x):
            return _ImageChunk(
                x, nodata_vals=self.nodata_vals, nan_fill=nan_fill
            ).apply(
                func,
                returns_tuple=n_outputs > 1,
                mask_nodata=mask_nodata,
                **ufunc_kwargs,
            )

        result = xr.apply_ufunc(
            ufunc,
            image,
            dask="parallelized",
            input_core_dims=[[self.band_dim_name]],
            exclude_dims=set((self.band_dim_name,)),
            output_core_dims=output_dims,
            output_dtypes=output_dtypes,
            dask_gufunc_kwargs=dict(
                output_sizes=output_sizes,
                allow_rechunk=True,
            ),
        )

        if n_outputs > 1:
            result = tuple(
                self._postprocess(x, output_coords=output_coords) for x in result
            )
        else:
            result = self._postprocess(result, output_coords=output_coords)

        return result


class DatasetImage(DataArrayImage):
    def __init__(self, image: xr.Dataset, nodata_vals: NoDataType = None):
        # The image itself will be stored as a DataArray, but keep the Dataset for
        # metadata like _FillValues.
        self.dataset = image
        super().__init__(image.to_dataarray(), nodata_vals=nodata_vals)

    @property
    def band_names(self) -> NDArray:
        return np.array(list(self.dataset.data_vars))

    def _validate_nodata_vals(self, nodata_vals: NoDataType) -> NDArray | None:
        """
        Get an array of NoData values in the shape (bands,) based on user input and
        Dataset metadata.
        """
        fill_vals = [
            self.dataset[var].attrs.get("_FillValue") for var in self.dataset.data_vars
        ]

        # Defer to provided NoData vals first. Next, try using per-variable fill values.
        # If at least one variable specifies a NoData value, use them all. Variables
        # that didn't specify a fill value will be assigned None.
        if nodata_vals is None and not all(v is None for v in fill_vals):
            return np.array(fill_vals)

        # Fall back to the DataArray logic for handling NoData
        return super()._validate_nodata_vals(nodata_vals)

    def _postprocess(
        self,
        result: xr.DataArray,
        output_coords: dict[str, list[str | int]],
    ) -> xr.Dataset:
        """Process the output of an applied ufunc"""
        result = super()._postprocess(result, output_coords=output_coords)

        var_dim = result.dims[self.band_dim]
        return result.to_dataset(dim=var_dim)
