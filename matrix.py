import numpy as np
from random import uniform


def hilbert_matrix( n: int ):
    """Hilbert Matrix

    :param n: Dimension.
    :return: Matrix as np.ndarray.
    """
    matrix = np.asarray( [ [ 1 / ( i + j - 1 ) for j in range( 1, n + 1, 1 ) ] for i in range( 1, n + 1, 1 ) ] )
    return matrix


def toeplitz_matrix( n: int, coef: list = None, start: float = None, stop: float = None ):
    """Toeplitz Matrix

    :param n: Dimension.
    :param coef List of coefficients.
    :param start: Lower limit of the random values.
    :param stop: Upper limit of the random values.
    :return: Matrix as np.ndarray.
    """
    if coef is None:
        coef_upper = [ uniform( start, stop ) for _ in range( n ) ]
        coef_lower = [ uniform( start, stop ) for _ in range( n - 1 ) ]
    else:
        coef_lower = coef[ : n - 1 ]
        coef_upper = coef[ n - 1 : ]

    matrix = np.full( ( n, n ), 0. )
    for i in range( n ):
        for j in range( n ):
            if i <= j:
                matrix[ i, j ] = coef_upper[ j - i ]
            else:
                matrix[ i, j ] = coef_lower[ j - i ]
    return matrix


def vandermonde_matrix( n: int, coef: list = None, start: int = None, stop: int = None ):
    """Vandermonde Matrix

    :param n: Dimension.
    :param coef: List of the common ratio of the geometric sequence.
    :param start: Lower limit of the random values.
    :param stop: Upper limit of the random values.
    :return: Matrix as np.ndarray.
    """
    if coef is None:
        coef = [ uniform( start, stop ) for _ in range( n ) ]
    matrix = np.asarray( [ [ coef[ i ] ** j for j in range( n ) ] for i in range( n ) ] )
    return matrix


def cauchy_matrix( n: int, x: list = None, y: list = None, start: int = None, stop: int = None ):
    """Cauchy Matrix

    :param n: Dimension.
    :param x: List of real numbers.
    :param y: List of real numbers.
    :param start: Lower limit of the random values.
    :param stop: Upper limit of the random values.
    :return: Matrix as np.ndarray.
    """
    if x is None and y is None:
        x = [ uniform( start, stop ) for _ in range( n ) ]
        y = [ uniform( start, stop ) for _ in range( n ) ]
    matrix = np.asarray( [ [ 1 / ( x[ i ] - y[ j ] ) for j in range( n ) ] for i in range( n ) ] )
    return matrix