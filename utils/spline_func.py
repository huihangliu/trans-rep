
from scipy.interpolate import BSpline, splev
import numpy as np

def apply_bspline_transform(x, knots, degree=3):
    # Number of basis functions
    n_basis = len(knots) - (degree + 1)

    # Apply B-spline transformation
    transformed = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        coefficients = np.zeros(n_basis)
        coefficients[i] = 1
        spline = BSpline(knots, coefficients, degree)
        transformed[:, i] = spline(x)

    return transformed

def apply_cubic_spline_transform(x, knots, degree=3):
  # Generate cubic spline basis functions for each knot
  if degree != 3: raise ValueError('degree of cubic spline method should be 3!')

  transformed = np.zeros((len(x), len(knots) + degree - 1))
  for i in range(len(knots)):
    t = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))
    c = np.zeros(len(knots) + degree - 1)
    c[i] = 1
    transformed[:, i] = splev(x, (t, c, degree))
  return transformed

def transform_multidimensional_X(X, knots, degree=3, flag_spline=None):
  if flag_spline is None: flag_spline = 'cubic'
  if flag_spline == 'bspline':
    transform_func = apply_bspline_transform
  elif flag_spline == 'cubic':
    transform_func = apply_cubic_spline_transform

  n_samples, n_features = X.shape
  if flag_spline == 'bspline':
    n_basis = len(knots) - (degree + 1)
  elif flag_spline == 'cubic':
    n_basis = len(knots) + degree - 1
  n_transformed_features = n_features * n_basis
  X_transformed = np.zeros((n_samples, n_transformed_features))

  current_col = 0
  for i in range(n_features):
    transformed_feature = transform_func(X[:, i], knots, degree)
    n_cols = transformed_feature.shape[1]
    X_transformed[:, current_col:current_col + n_cols] = transformed_feature
    current_col += n_cols

  return X_transformed
