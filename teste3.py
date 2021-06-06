from decomposition import gram_schmidt, qr_decomposition
from eigen_methods import qr_method, singular_value_decomposition
import numpy as np


A = np.asarray( [ [ 12, -51, 4 ], [ 6, 167, -68 ], [ -4, 24, -41 ] ] )
gs = gram_schmidt( A = A, normalize = True )
Q, R = qr_decomposition( A )
Q2, R2 = np.linalg.qr( A )

A = np.asarray( [ [ 4, 1, -1 ], [ 2, 5, -2 ], [ 1, 1, 2 ] ] )
# A = np.asarray( [ [ 3, 2, 2 ], [ 2, 3, -2 ] ] )
# A = np.asarray( [ [ 0, 1 ], [ -2, -3 ] ] )
# eigenvalues, eigenvectors = qr_method( A = A, max_error = 1e-10, max_iter = 200 )

# eigenvalues2, eigenvectors2 = np.linalg.eig( A )

U, S, V = singular_value_decomposition( A )

var = 0