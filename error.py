import numpy as np

def absolute_error( solution: np.ndarray, reference: np.ndarray ):
    return max( abs( solution - reference ) )

def mean_absolute_error( solution: np.ndarray, reference: np.ndarray ):
    return np.mean( abs( solution - reference ) )

def relative_error( ):
    pass

def mean_relative_error( ):
    return