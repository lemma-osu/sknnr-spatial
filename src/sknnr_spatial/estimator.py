from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from warnings import warn

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import _get_feature_names, check_is_fitted
from typing_extensions import Literal, overload

from .image import Image
from .types import EstimatorType
from .utils.estimator import is_fitted, suppress_feature_name_warnings
from .utils.image import image_or_fallback
from .utils.wrapper import AttrWrapper, check_wrapper_implements

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

    from .types import ImageType, NoDataType

ESTIMATOR_OUTPUT_DTYPES: dict[str, np.dtype] = {
    "classifier": np.int32,
    "clusterer": np.int32,
    "regressor": np.float64,
}


@dataclass
class FittedMetadata:
    """Metadata from a fitted estimator."""

    n_targets: int
    target_names: tuple[str | int, ...]
    feature_names: NDArray


class ImageEstimator(AttrWrapper[EstimatorType]):
    """
    An sklearn-compatible estimator wrapper with overriden methods for image data.

    Parameters
    ----------
    wrapped : BaseEstimator
        An sklearn-compatible estimator to wrap with image methods. Fitted estimators
        will be reset when wrapped and must be re-fit after wrapping.
    """

    _wrapped: EstimatorType
    _wrapped_meta: FittedMetadata

    def __init__(self, wrapped: EstimatorType):
        super().__init__(self._reset_estimator(wrapped))

    @staticmethod
    def _reset_estimator(estimator: EstimatorType) -> EstimatorType:
        """Take an estimator and reset and warn if it was previously fitted."""
        if is_fitted(estimator):
            warn(
                "Wrapping estimator that has already been fit. The estimator must be "
                "fit again after wrapping.",
                stacklevel=2,
            )
            return clone(estimator)

        return estimator

    def _get_n_targets(self, y: NDArray | pd.DataFrame | pd.Series | None) -> int:
        """Get the number of targets used to fit the estimator."""
        # Unsupervised and single-output estimators should both return a single target
        if y is None or y.ndim == 1:
            return 1

        return y.shape[-1]

    def _get_target_names(
        self, y: NDArray | pd.DataFrame | pd.Series
    ) -> tuple[str | int, ...]:
        """Get the target names used to fit the estimator, if available."""
        # Dataframe
        if hasattr(y, "columns"):
            return tuple(y.columns)

        # Series
        if hasattr(y, "name"):
            return tuple([y.name])

        # Default to sequential identifiers
        return tuple(range(self._get_n_targets(y)))

    @check_wrapper_implements
    def fit(self, X, y=None, **kwargs) -> ImageEstimator[EstimatorType]:
        """
        Fit an estimator from a training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression). Single-output targets of shape (n_samples, 1) will be squeezed
            to shape (n_samples,) to allow consistent prediction across all estimators.
        **kwargs : dict
            Additional keyword arguments passed to the estimator's fit method, e.g.
            `sample_weight`.

        Returns
        -------
        self : ImageEstimator
            The wrapper around the fitted estimator.
        """
        if y is not None:
            # Squeeze extra y dimensions. This will convert from shape (n_samples, 1)
            # which causes inconsistent output shapes with different sklearn estimators,
            # to (n_samples,), which has a consistent output shape.
            y = y.squeeze()

        self._wrapped = self._wrapped.fit(X, y, **kwargs)
        fitted_feature_names = _get_feature_names(X)

        self._wrapped_meta = FittedMetadata(
            n_targets=self._get_n_targets(y),
            target_names=self._get_target_names(y),
            feature_names=fitted_feature_names
            if fitted_feature_names is not None
            else np.array([]),
        )

        return self

    @check_wrapper_implements
    @image_or_fallback
    def predict(
        self, X_image: ImageType, *, nodata_vals: NoDataType = None, **predict_kwargs
    ) -> ImageType:
        """
        Predict target(s) for X_image.

        Notes
        -----
        If X_image is not an image, the estimator's unmodified predict method will be
        called instead.

        Parameters
        ----------
        X_image : Numpy or Xarray image with 3 dimensions (y, x, band)
            The input image. Features in the band dimension should correspond with the
            features used to fit the estimator.
        nodata_vals : float or sequence of floats, optional
            NoData values to mask in the output image. A single value will be broadcast
            to all bands while sequences of values will be assigned band-wise. If None,
            values will be inferred if possible based on image metadata.
        **predict_kwargs
            Additional arguments passed to the estimator's predict method.

        Returns
        -------
        y_image : Numpy or Xarray image with 3 dimensions (y, x, targets)
            The predicted values.
        """
        output_dim_name = "variable"
        image = Image.from_image(X_image, nodata_vals=nodata_vals)

        # TODO: Re-implement once Image can parse band names
        # self._check_feature_names(wrapper.preprocessor.band_names)

        # Any estimator with an undefined type should fall back to floating
        # point for safety.
        estimator_type = getattr(self._wrapped, "_estimator_type", "")
        output_dtype = ESTIMATOR_OUTPUT_DTYPES.get(estimator_type, np.float64)

        return image.apply_ufunc_across_bands(
            suppress_feature_name_warnings(self._wrapped.predict),
            output_dims=[[output_dim_name]],
            output_dtypes=[output_dtype],
            output_sizes={output_dim_name: self._wrapped_meta.n_targets},
            output_coords={output_dim_name: list(self._wrapped_meta.target_names)},
            **predict_kwargs,
        )

    @check_wrapper_implements
    @image_or_fallback
    @overload
    def kneighbors(
        self,
        X_image: ImageType,
        *,
        n_neighbors: int | None = None,
        return_distance: Literal[False] = False,
        nodata_vals: NoDataType = None,
        **kneighbors_kwargs,
    ) -> ImageType: ...

    @check_wrapper_implements
    @image_or_fallback
    @overload
    def kneighbors(
        self,
        X_image: ImageType,
        *,
        n_neighbors: int | None = None,
        return_distance: Literal[True] = True,
        nodata_vals: NoDataType = None,
        **kneighbors_kwargs,
    ) -> tuple[ImageType, ImageType]: ...

    @check_wrapper_implements
    @image_or_fallback
    def kneighbors(
        self,
        X_image: ImageType,
        *,
        n_neighbors: int | None = None,
        return_distance: bool = True,
        nodata_vals: NoDataType = None,
        **kneighbors_kwargs,
    ) -> ImageType | tuple[ImageType, ImageType]:
        """
        Find the K-neighbors of each pixel in an image.

        Returns indices of and distances to the neighbors for each pixel.

        Notes
        -----
        If X_image is not an image, the estimator's unmodified kneighbors method will be
        called instead.

        Parameters
        ----------
        X_image : Numpy or Xarray image with 3 dimensions (y, x, band)
            The input image. Features in the band dimension should correspond with the
            features used to fit the estimator.
        n_neighbors : int, optional
            Number of neighbors required for each sample. The default is the value
            passed to the wrapped estimator's constructor.
        return_distance : bool, default=True
            If True, return distances to the neighbors of each sample. If False, return
            indices only.
        nodata_vals : float or sequence of floats, optional
            NoData values to mask in the output image. A single value will be broadcast
            to all bands while sequences of values will be assigned band-wise. If None,
            values will be inferred if possible based on image metadata.
        **kneighbors_kwargs
            Additional arguments passed to the estimator's kneighbors method.

        Returns
        -------
        neigh_dist : Numpy or Xarray image with 3 dimensions (y, x, neighbor)
            Array representing the lengths to points, only present if
            return_distance=True.
        neigh_ind : Numpy or Xarray image with 3 dimensions (y, x, neighbor)
            Indices of the nearest points in the population matrix.
        """
        image = Image.from_image(X_image, nodata_vals=nodata_vals)
        k = n_neighbors or cast(int, getattr(self._wrapped, "n_neighbors", 5))

        # TODO: Re-implement
        # self._check_feature_names(wrapper.preprocessor.band_names)

        return image.apply_ufunc_across_bands(
            suppress_feature_name_warnings(self._wrapped.kneighbors),
            output_dims=[["k"], ["k"]] if return_distance else [["k"]],
            output_dtypes=[float, int] if return_distance else [int],
            output_sizes={"k": k},
            output_coords={"k": list(range(1, k + 1))},
            n_neighbors=k,
            return_distance=return_distance,
            **kneighbors_kwargs,
        )

    def _check_feature_names(self, image_feature_names: NDArray) -> None:
        """Check that image feature names match feature names seen during fitting."""
        check_is_fitted(self._wrapped)
        fitted_feature_names = self._wrapped_meta.feature_names

        no_fitted_names = len(fitted_feature_names) == 0
        no_image_names = len(image_feature_names) == 0

        if no_fitted_names and no_image_names:
            return

        if no_fitted_names:
            warn(
                f"X_image has feature names, but {self._wrapped.__class__.__name__} was"
                " fitted without feature names",
                stacklevel=2,
            )
            return

        if no_image_names:
            warn(
                "X_image does not have feature names, but"
                f" {self._wrapped.__class__.__name__} was fitted with feature names",
                stacklevel=2,
            )
            return

        if len(fitted_feature_names) != len(image_feature_names) or np.any(
            fitted_feature_names != image_feature_names
        ):
            msg = "Image band names should match those that were passed during fit.\n"
            fitted_feature_names_set = set(fitted_feature_names)
            image_feature_names_set = set(image_feature_names)

            unexpected_names = sorted(
                image_feature_names_set - fitted_feature_names_set
            )
            missing_names = sorted(fitted_feature_names_set - image_feature_names_set)

            def add_names(names):
                max_n_names = 5
                if len(names) > max_n_names:
                    names = [*names[: max_n_names + 1], "..."]

                return "".join([f"- {name}\n" for name in names])

            if unexpected_names:
                msg += "Band names unseen at fit time:\n"
                msg += add_names(unexpected_names)

            if missing_names:
                msg += "Band names seen at fit time, yet now missing:\n"
                msg += add_names(missing_names)

            if not missing_names and not unexpected_names:
                msg += "Band names must be in the same order as they were in fit.\n"

            raise ValueError(msg)


def wrap(estimator: EstimatorType) -> ImageEstimator[EstimatorType]:
    """
    Wrap an sklearn-compatible estimator with overriden methods for image data.

    Parameters
    ----------
    estimator : BaseEstimator
        An sklearn-compatible estimator to wrap with image methods. Fitted estimators
        will be reset when wrapped and must be re-fit after wrapping.

    Returns
    -------
    ImageEstimator
        An estimator with relevant methods overriden to work with image data, e.g.
        `predict` and `kneighbors`. Methods will continue to work with non-image data
        and non-overriden methods and attributes will be unchanged.

    Examples
    --------
    Instantiate an estimator, wrap it, then fit as usual:

    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> from sknnr_spatial.datasets import load_swo_ecoplot
    >>> X_img, X, y = load_swo_ecoplot(as_dataset=True)
    >>> est = wrap(KNeighborsRegressor(n_neighbors=3)).fit(X, y)

    Use a wrapped estimator to predict from image data stored in Numpy or Xarray arrays:

    >>> pred = est.predict(X_img)
    >>> pred.PSME_COV.shape
    (128, 128)
    """
    return ImageEstimator(estimator)
