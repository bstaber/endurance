"""Homoscedastic example with hetGPy."""

import matplotlib.pyplot as plt
import numpy as np
from hetgpy import homGP

x_train = np.random.uniform(-3, 3, (100, 1))
y_train = (
    np.cos(2.0 * x_train)
    + 2.0 * np.sin(x_train)
    + 0.25 * np.random.normal(0, 1, (100, 1))
)

kmodel = homGP()
kmodel.mle(x_train, y_train, covtype="Matern5_2", lower=None, upper=None)

x_test = np.linspace(-3, 3, 100).reshape(-1, 1)
preds = kmodel.predict(
    x=x_test, interval="predictive", interval_lower=0.05, interval_upper=0.95
)
interval = preds["predictive_interval"]

plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, "kx", label="Training Data")
plt.plot(x_test, preds["mean"], "b-", label="Predictive Mean")
plt.fill_between(
    x_test.flatten(),
    interval["lower"],
    interval["upper"],
    color="lightblue",
    alpha=0.5,
    label="90% Confidence Interval",
)
plt.title("Homoscedastic GP Regression with hetGPy")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.savefig("1d_homoscedastic_hetgpy.png")
