import numpy as np


def qr_decomposition( A: np.ndarray ):
    orthogonal = gram_schmidt( A = A, normalize = True )
    Q_trans = np.asarray( orthogonal )
    Q = Q_trans.transpose( )
    R = np.dot( Q_trans, A )
    return Q, R


def gram_schmidt( A: np.ndarray, normalize: bool = True ):
    n_vectors = A.shape[ 1 ]
    orthogonal = [ ]
    for i in range( n_vectors ):
        ai = A[ :, i ]
        ui = ai
        for j in range( i ):
            ui = ui - proj( orthogonal[ j ], ai ) # calculate projection
        orthogonal.append( ui )

    if normalize:
        for i in range( n_vectors ):
            ui = orthogonal[ i ]
            orthogonal[ i ] = normalize_vector( ui )

    return orthogonal


def proj( u: np.ndarray, a: np.ndarray ):
    return ( inner_product( u, a ) / inner_product( u, u ) ) * u


def norm_vector( v: np.ndarray ):
    return np.sqrt( np.sum( [ vi ** 2 for vi in v ] ) )


def inner_product( u: np.ndarray, v: np.ndarray ):
    return np.sum( [ ui * vi for ui, vi in zip( list( u ), list( v ) ) ] )


def normalize_vector( v: np.ndarray ):
    norm = norm_vector( v )
    if norm != 0:
        return v / norm
    return v