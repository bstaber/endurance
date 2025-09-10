"""Homoscedastic example with hetGPy."""

from hetgpy import homGP
from sklearn.datasets import make_regression

x_train, y_train = make_regression(
    n_samples=100, n_features=32, n_informative=10, noise=0.1, random_state=42
)

kmodel = homGP()
kmodel.mle(x_train, y_train, covtype="Matern5_2", lower=None, upper=None)
