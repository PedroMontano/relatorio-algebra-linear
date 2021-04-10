import numpy as np
from sklearn.metrics import mean_squared_error

def absolute_error( solution: np.ndarray, reference: np.ndarray ):
    return max( abs( solution - reference ) )

def absolute_mean_error(solution: np.ndarray, reference: np.ndarray):
    return np.mean( abs( solution - reference ) )

def absolute_mean_squared_error(solution: np.ndarray, reference: np.ndarray):
    return mean_squared_error( solution, reference )

def relative_error( ):
    pass

def mean_relative_error( ):
    return