import numpy as np
from random import randint


def laplace( A: np.ndarray ):

    if A.shape == ( 2, 2 ):
        det = ( A[ 0, 0 ] * A[ 1, 1 ] ) - ( A[ 0, 1 ] * A[ 1, 0 ] )
        return det

    else:
        # choose i, j pair
        n_rows = A.shape[ 0 ]
        n_columns = A.shape[ 1 ]

        i = randint( 0, n_rows - 1 )
        j = randint( 0, n_columns - 1 )

        # choose direction ( 0 for row / 1 for column )
        axis = randint( 0, 1 )

        # get coefficients
        if axis == 0:
            coef = A[ i , : ]
            p = i

        else:
            coef = A[ :, j ]
            p = j

        # calculate determinant by Laplace formula
        det = .0
        for k in range( coef.shape[ 0 ] ):
            if axis == 0:
                M = np.delete( A, i, axis = 0 )
                M = np.delete( M, k, axis = 1 )

            else:
                M = np.delete( A, j, axis = 1 )
                M = np.delete( M, k, axis = 0 )

            cofactor = ( -1 ) ** ( p + k + 2 ) * laplace( M )
            det += coef[ k ] * cofactor

        return det