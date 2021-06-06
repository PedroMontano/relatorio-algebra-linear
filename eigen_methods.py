from decomposition import qr_decomposition, normalize_vector
from error import absolute_error
import numpy as np
import copy


def qr_method( A: np.ndarray, max_error: float, max_iter: int ):
    n = A.shape[ 0 ]
    error = 1.
    n_iter = 0
    Ai = copy.deepcopy( A )
    eigenvectors = np.identity( n )
    eigenvalues = np.full( n, 1. )
    eigenvalues_history = [ eigenvalues ]
    while error > max_error and n_iter < max_iter:
        Qi, Ri = qr_decomposition( Ai )
        Ai = np.dot( Ri, Qi )
        eigenvectors = np.dot( eigenvectors, Qi )

        # calculate error
        eigenvalues = np.diagonal( Ai )
        error = absolute_error( solution = eigenvalues, reference = eigenvalues_history[ -1 ] )
        eigenvalues_history.append( eigenvalues )

        n_iter += 1

    return eigenvalues, eigenvectors


def singular_value_decomposition( A: np.ndarray ):
    A2 = np.dot( A.transpose( ), A )
    # eigenvalues, eigenvectors = qr_method( A = A2, max_error = 1e-6, max_iter = 1000 )
    eigenvalues, eigenvectors = np.linalg.eig( A2 )

    n = eigenvalues.shape[ 0 ]
    S = np.identity( n )
    for i  in range( n ): # fill matrix with singular values
        S[ i, i ] = np.sqrt( eigenvalues[ i ] )

    for i in range( n ): # normalize eigenvectors
        vi = eigenvectors[ :, i ]
        normalized_vi = normalize_vector( vi )
        eigenvectors[ :, i ] = normalized_vi[ : ]

    U = [ ]
    for i in range( n ): # calculate column vectors of U
        ui = np.dot( A, eigenvectors[ :, i ] ) / S[ i, i ]
        U.append( normalize_vector( ui ) )
    U = np.asarray( U ).transpose( )

    return U, S, eigenvectors



