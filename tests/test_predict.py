import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sknnr_spatial import kneighbors, predict

from .image_utils import parametrize_image_types, unwrap, wrap


@pytest.fixture()
def dummy_model_data():
    n_features = 5
    n_rows = 10

    X_image = np.random.rand(8, 16, n_features)
    X = np.random.rand(n_rows, n_features)
    y = np.random.rand(n_rows, 3)

    return X_image, X, y


@parametrize_image_types
@pytest.mark.parametrize("estimator", [KNeighborsRegressor(), RandomForestRegressor()])
def test_predict(dummy_model_data, image_type, estimator):
    """Test that predict works with all image types and a few estimators."""
    X_image, X, y = dummy_model_data
    estimator = estimator.fit(X, y)

    X_wrapped = wrap(X_image, type=image_type.cls)
    y_pred = unwrap(predict(X_wrapped, estimator=estimator))

    assert y_pred.ndim == 3
    assert_array_equal(y_pred.shape, (X_image.shape[0], X_image.shape[1], y.shape[-1]))


@parametrize_image_types
def test_kneighbors_with_distance(dummy_model_data, image_type):
    """Test kneighbors works with all image types when returning distance."""
    k = 3
    X_image, X, y = dummy_model_data
    estimator = KNeighborsRegressor(n_neighbors=k).fit(X, y)

    X_wrapped = wrap(X_image, type=image_type.cls)
    dist, nn = kneighbors(X_wrapped, estimator=estimator, return_distance=True)
    dist = unwrap(dist)
    nn = unwrap(nn)

    assert dist.ndim == 3
    assert nn.ndim == 3

    assert_array_equal(dist.shape, (X_image.shape[0], X_image.shape[1], k))
    assert_array_equal(nn.shape, (X_image.shape[0], X_image.shape[1], k))


@parametrize_image_types
def test_kneighbors_without_distance(dummy_model_data, image_type):
    """Test kneighbors works with all image types when NOT returning distance."""
    k = 3
    X_image, X, y = dummy_model_data
    estimator = KNeighborsRegressor(n_neighbors=k).fit(X, y)

    X_wrapped = wrap(X_image, type=image_type.cls)
    nn = kneighbors(X_wrapped, estimator=estimator, return_distance=False)
    nn = unwrap(nn)

    assert nn.ndim == 3

    assert_array_equal(nn.shape, (X_image.shape[0], X_image.shape[1], k))