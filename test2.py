import numpy as np
from determinant import laplace


A = np.asarray( [ [ 2, -5, 3 ],
                  [ 0, 7, -2 ],
                  [ -1, 4, 1 ] ] )

det0 = np.linalg.det( A )
det = laplace( A )

var = 0