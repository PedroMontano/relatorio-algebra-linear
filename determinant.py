import numpy as np
from random import randint


def laplace( A: np.ndarray ):

    # print( f'Dimension: {A.shape}' )
    if A.shape == ( 2, 2 ):
        det = ( A[ 0, 0 ] * A[ 1, 1 ] ) - ( A[ 0, 1 ] * A[ 1, 0 ] )
        return det

    else:
        n = A.shape[ 0 ]

        # choose row / column
        i = randint( 0, n - 1 )
        # i = 0

        # choose direction ( 0 for row / 1 for column )
        axis = randint( 0, 1 )

        # get coefficients
        if axis == 0:
            coef = A[ i , : ]

        else:
            coef = A[ :, i ]
        # coef = A[ i , : ]

        # calculate determinant by Laplace formula
        det = .0
        for k in range( coef.shape[ 0 ] ):
            if axis == 0:
                M = np.delete( A, i, axis = 0 )
                M = np.delete( M, k, axis = 1 )

            else:
                M = np.delete( A, i, axis = 1 )
                M = np.delete( M, k, axis = 0 )
            # M = np.delete( A, i, axis = 0 )
            # M = np.delete( M, k, axis = 1 )

            cofactor = ( -1 ) ** ( i + k + 2 ) * laplace( M )
            det += coef[ k ] * cofactor

        return det