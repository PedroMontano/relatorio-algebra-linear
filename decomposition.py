import numpy as np

def qr_decomposition( A: np.ndarray ):
    orthogonal = gram_schmidt( A )
    Q_trans = np.asarray( orthogonal )
    Q = Q_trans.transpose( )
    R = np.dot( Q_trans, A )
    return Q, R


def gram_schmidt( A: np.ndarray ):
    n_vectors = A.shape[ 1 ]

    orthogonal = [ ]
    for i in range( n_vectors ):
        ai = A[ :, i ]
        ui = ai
        for j in range( i ):
            ui -= proj( orthogonal[ -1 ], ai )
        orthogonal.append( ui )

    return orthogonal


def proj( u: np.ndarray, a: np.ndarray ):
    return ( np.cross( u, a ) / np.cross( u, u ) ) * u