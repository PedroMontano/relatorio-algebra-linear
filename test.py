import numpy as np
from linear_methods import gaussian_elimination, jacobi, gauss_seidel, gauss_jordan, lu_decomposition
from matrix import hilbert_matrix, vandermonde_matrix, cauchy_matrix, toeplitz_matrix
from random import uniform

n = 3
max_error = 1.e-6
max_iter = 10000
guess =  np.asarray( [ 1., 1., 1. ] )
# guess = np.full( n, 0.2 )
# guess = np.asarray( [ 0., 0. ] )

# A = np.asarray( [ [ 1., -1., -1. ], [ 3., -4., -2. ], [ 2., -3., -2. ] ] )
A = np.asarray( [ [ 7., 2., 3. ], [ 3., 10., 1. ], [ 2., 4., 13. ] ] )
b = np.asarray( [ 1., 19., -4. ] )

x = np.full( n, 1 )
# A = hilbert_matrix( n )
# A = vandermonde_matrix( n = n, start = 1, stop = 2 )
# A = cauchy_matrix( n = n, start = -10, stop = 10 )
# b = np.dot( A, x )

method_1 = gaussian_elimination( )
solution_1 = method_1.solve( A = A, b = b )
time_1 = method_1.time

method_2 = jacobi( guess = guess, max_error = max_error, max_iter = max_iter )
solution_2 = method_2.solve( A = A, b = b )
time_2 = method_2.time

method_3 = gauss_seidel( guess = guess, max_error = max_error, max_iter = max_iter )
solution_3 = method_3.solve( A = A, b = b )
time_3 = method_3.time

method_4 = gauss_jordan( )
solution_4 = method_4.solve( A = A, b = b )
time_4 = method_4.time

method_5 = lu_decomposition( )
solution_5 = method_5.solve( A = A, b = b )
time_5 = method_5.time

var = 0