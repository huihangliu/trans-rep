import numpy as np

def func0(x):
  return x


def func1(x):
  return np.sin(x)


def func2(x):
  return np.clip(np.sqrt(np.abs(x+0.5)) * 2 - 1, a_min=-2, a_max=2)


def func3(x):
  return np.clip((1 - np.abs(x-0.5)) ** 2, a_min=None, a_max=2)


def func4(x):
  return 1 / (1.0 + np.exp(x))


def func5(x):
  return np.cos(np.pi * x) # / 2


def func6(x):
  return np.sin(np.pi * x) # / 2


def func7(x):
  return np.cos(2 * x)


def func8(x):
  return -np.cos(x)


def func9(x):
  return np.clip(np.tan(x + 0.1), a_min=-2, a_max=2)


def func10(x):
  positive_values = np.clip(x + 1.5, a_min=0.15, a_max=8)
  return np.log(positive_values)


def func11(x):
  return np.clip(np.exp(x), a_min=None, a_max=2)


def func12(x):
  return np.clip(x**2, a_min=None, a_max=2)


def func13(x):
  return np.arctan(x)
