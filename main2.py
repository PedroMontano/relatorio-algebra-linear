import numpy as np
from matrix import hilbert_matrix, vandermonde_matrix, cauchy_matrix, toeplitz_matrix
from methods import cramer


n = 12
x = np.full( n, 1. )

_COLORS = { 'HILBERT': 'blue',
            'VANDERMONDE': 'red',
            'CAUCHY': 'green',
            'TOEPLITZ': 'darkmagenta' }

# Hilbert matrix
H = hilbert_matrix( n )

# Vandermonde matrix
alpha = [ i + 1 for i in range( n ) ]
V = vandermonde_matrix( n = n, coef = alpha )

# Cauchy matrix
xi = [ 4.32, -31.42, -19.26, -34.53, 72.48, -97.49, -15.22, -10.88, 96.19, 87.66, -86.33, 43.60 ]
yi = [ 4.29, -31.33, -19.39, -34.55, 72.41, -97.31, -15.27, -10.99, 96.17, 87.51, -86.23, 43.57 ]

C = cauchy_matrix( n = n, x = list( xi ), y = list( yi ) )

# Toeplitz matrix
coef = [ 1.19, 8.29, -1.71, 7.06, 7.47, 3.84, 2.41, 2.58, -4.67, -8.84, -8.06, 92.8, -0.72,
         -4.04, 3.79, 3.83, 2.36, 7.37, -8.55, -8.75, -0.47, 6.95, -4.35 ]
T = toeplitz_matrix( n = n, coef = coef )

_ALL_A = { 'HILBERT': H,
           'VANDERMONDE': V,
           'CAUCHY': C,
           'TOEPLITZ': T }

_ALL_B = { m: np.dot( _ALL_A[ m ], x ) for m in _ALL_A }



var = 0