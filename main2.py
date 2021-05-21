import numpy as np
from matrix import hilbert_matrix, vandermonde_matrix, cauchy_matrix, toeplitz_matrix
from methods import cramer
import shelve


def persist( filename: str, tag: str, content ):
    with shelve.open( filename ) as db:
        db[ tag ] = content

def load( filename: str, tag: str = None ) :
    if tag is not None :
        with shelve.open( filename ) as db :
            persisted = db.get( tag )
        if persisted is None:
            persisted = { }
        return persisted
    else :
        with shelve.open( filename ) as db :
            persisted = { }
            for k in db.keys( ) :
                persisted[ k ] = db.get( k )
        return persisted


_PERSIST_FILE = './persistence/SOLVERS'
n = 9
_DIMENSIONS = [ _ for _ in range( 2, n + 1, 1 ) ]
x = np.full( n, 1. )

_COLORS = { 'HILBERT': 'blue',
            'VANDERMONDE': 'red',
            'CAUCHY': 'green',
            'TOEPLITZ': 'darkmagenta' }

_MATRIX_NAMES = [ 'HILBERT', 'VANDERMONDE', 'CAUCHY', 'TOEPLITZ' ]

# Hilbert matrix
# H = hilbert_matrix( n )
#
# # Vandermonde matrix
# alpha = [ i + 1 for i in range( n ) ]
# V = vandermonde_matrix( n = n, coef = alpha )
#
# # Cauchy matrix
set1 = [ 4.32, -31.42, -19.26, -34.53, 72.48, -97.49, -15.22, -10.88, 96.19, 87.66, -86.33, 43.60 ]
set2 = [ 4.29, -31.33, -19.39, -34.55, 72.41, -97.31, -15.27, -10.99, 96.17, 87.51, -86.23, 43.57 ]
#
# C = cauchy_matrix( n = n, x = list( xi ), y = list( yi ) )
#
# # Toeplitz matrix
coef = [ 7.84, 1.19, 8.29, -1.71, 7.06, 7.47, 3.84, 2.41, 2.58, -4.67, -8.84, -8.06, 92.8, -0.72,
         -4.04, 3.79, 3.83, 2.36, 7.37, -8.55, -8.75, -0.47, 6.95, -4.35, -2.59 ]
# T = toeplitz_matrix( n = n, coef = coef )
#
# _ALL_A = { 'HILBERT': H,
#            'VANDERMONDE': V,
#            'CAUCHY': C,
#            'TOEPLITZ': T }
#
# _ALL_B = { m: np.dot( _ALL_A[ m ], x ) for m in _ALL_A }
#
# Solving
# _RESULT = { }
# for m in _ALL_A:
#     print( f'matrix: {m}' )
#     A = _ALL_A.get( m )
#     b = _ALL_B.get( m )
#
#     solver = cramer( )
#     solver.solve( A = A, b = b )
#
#     try:
#         solver.calculate_true_error( reference = x )
#     except:
#         pass
#
#     persist( filename = _PERSIST_FILE, tag = m, content = solver )
#     _RESULT[ m ] = solver
#
# _RESULT = load( filename = _PERSIST_FILE, tag = None )

_RESULT = { 'HILBERT':[ ],
            'VANDERMONDE': [ ],
            'CAUCHY': [ ],
            'TOEPLITZ': [ ] }

for i, ni in enumerate( _DIMENSIONS ):
    print( f'dimension: {ni}' )
    xi = np.full( ni, 1. )

    # Hilbert matrix
    print( f'matrix: HILBERT' )
    H = hilbert_matrix( ni )
    b_H = np.dot( H, xi )

    solver_H = cramer( )
    solver_H.solve( H, b_H )
    _RESULT[ 'HILBERT' ].append( solver_H )

    # Vandermonde matrix
    print( f'matrix: VANDERMONDE' )
    alpha = [ i + 1 for i in range( ni ) ]
    V = vandermonde_matrix( n = ni, coef = alpha )
    b_V = np.dot( V, xi )

    solver_V = cramer( )
    solver_V.solve( V, b_V )
    _RESULT[ 'VANDERMONDE' ].append( solver_V )

    # Cauchy matrix
    print( f'matrix: CAUCHY' )
    C = cauchy_matrix( n = ni, x = list( set1 ), y = list( set2 ) )
    b_C = np.dot( C, xi )

    solver_C = cramer( )
    solver_C.solve( C, b_C )
    _RESULT[ 'CAUCHY' ].append( solver_C )

    # Toeplitz matrix
    print( f'matrix: TOEPLITZ' )
    center = int( ( 1 + len( coef ) - 1 ) / 2 )
    coefi = coef[ 1 + center - ni : center + ni ]
    T = toeplitz_matrix( n = ni, coef = coefi )
    b_T = np.dot( T, xi )

    solver_T = cramer( )
    solver_T.solve( T, b_T )
    _RESULT[ 'TOEPLITZ' ].append( solver_T )

# persist( filename = _PERSIST_FILE, tag = 'DIMENSION', content = _RESULT )









var = 0