import numpy as np
from methods import gaussian_elimination, jacobi, gauss_seidel, gauss_jordan, lu_decomposition
from matrix import hilbert_matrix, vandermonde_matrix, cauchy_matrix, toeplitz_matrix
import plot as p

def write_file( data, path ):
    with open( path, 'w' ) as file:
        file.writelines( data )

n = 15
max_error = 1.e-5
max_iter = 10000
x = np.full( n, 1. )
guess = np.full( n, 0.5 )
kwargs = { 'guess': guess, 'max_error': max_error, 'max_iter': max_iter }

_ALL_SOLVER = { 'GAUSSIAN_ELIMINATION': gaussian_elimination,
                 'JACOBI': jacobi,
                 'GAUSS-SEIDEL': gauss_seidel,
                 'GAUSS-JORDAN': gauss_jordan,
                 'LU_DECOMPOSITION': lu_decomposition }

_COLORS = { 'GAUSSIAN_ELIMINATION': 'blue',
            'JACOBI': 'red',
            'GAUSS-SEIDEL': 'green',
            'GAUSS-JORDAN': 'gold',
            'LU_DECOMPOSITION': 'darkmagenta' }

# Hilbert matrix
H = hilbert_matrix( n )

# Vandermonde matrix
alpha = [ i + 1 for i in range( n ) ]
V = vandermonde_matrix( n = n, coef = alpha )

# Cauchy matrix
# xi = np.asarray( [ np.random.uniform( -100., 100. ) for _ in range( n ) ] )
# noise = np.random.normal( 0., 0.1, xi.shape[ 0 ] )
# yi = xi + noise
xi = [ 53.77, 4.32, -31.42, -19.26, -34.53, 72.48, 47.32, -97.49, -15.22, 57.20, -10.88, 96.19, 87.66, -86.33, 43.60 ]
yi = [ 53.74, 4.29, -31.33, -19.39, -34.55, 72.41, 47.37, -97.31, -15.27, 57.16, -10.99, 96.17, 87.51, -86.23, 43.57 ]

C = cauchy_matrix( n = n, x = list( xi ), y = list( yi ) )

# Toeplitz matrix
coef = [ 1.19, 7.84, 8.29, -1.71, 4.44, 7.06, 7.47, 3.84, 2.41, 2.58, -4.67, 2.53, -8.84, -8.06, 92.8, 0.38, -0.72,
         -4.04, 3.79, 3.83, 2.36, 7.37, -8.55, 9.74, -8.75, -0.47, 6.95, -2.59, -4.35 ]
T = toeplitz_matrix( n = n, coef = coef )

_ALL_A = { 'HILBERT': H,
           'VANDERMONDE': V,
           'CAUCHY': C,
           'TOEPLITZ': T }

_ALL_B = { m: np.dot( _ALL_A[ m ], x ) for m in _ALL_A }

# Solving
_RESULT = { }
for m in _ALL_A:
    _RESULT[ m ] = { }
    for s in _ALL_SOLVER:
        A = _ALL_A.get( m )
        b = _ALL_B.get( m )
        method = _ALL_SOLVER.get( s )

        if s == 'GAUSS-SEIDEL' or s == 'JACOBI':
            solver = method( **kwargs )
        else:
            solver = method( )
        solver.solve( A = A, b = b )
        solver.calculate_true_error( reference = x )
        _RESULT[ m ][ s ] = solver

# plot time x method
for m in _RESULT:
    time_list = [ ]
    label_list = [ ]
    color_list = [ ]
    for s in _RESULT[ m ]:
        solver = _RESULT[ m ][ s ]
        if solver.converged:
            time_list.append( solver.time )
            label_list.append( s )
            color_list.append( _COLORS[ s ] )

    time_list = np.asarray( time_list ) * 1000.
    f = p.bar_plot( height_list = time_list,
                    labels = label_list,
                    colors = color_list,
                    y_label = 'Tempo [ms]',
                    title = f'Tempo de Processamento ({m})' )
    f.savefig( f'./figures/1_time_{m}.png' )

    data = [ str( "{:.32f}".format( time_list[ i ] ) ) + f'\t{label_list[ i ]}\n' for i in range( len( time_list ) ) ]
    write_file( data = data, path = f'./tables/1_time_{m}.txt' )

# plot error x method
for m in _RESULT:
    error_list = [ ]
    label_list = [ ]
    color_list = [ ]
    for s in _RESULT[ m ]:
        solver = _RESULT[ m ][ s ]
        if solver.converged:
            error_list.append( solver.true_error )
            label_list.append( s )
            color_list.append( _COLORS[ s ] )

    if m != 'VANDERMONDE':
        # error_list = np.log10( np.asarray( error_list ) )
        error_list = np.asarray( error_list )
    else:
        error_list = np.asarray( error_list )
    f = p.bar_plot( height_list = error_list,
                    labels = label_list,
                    colors = color_list,
                    y_label = 'Erro Absoluto Médio',
                    title = f'Erro Absoluto Médio ({m})' )
    f.savefig( f'./figures/2_erro_absoluto_medio_{m}.png' )

    data = [ str( "{:.32f}".format( error_list[ i ] ) ) + f'\t{label_list[ i ]}\n' for i in range( len( error_list ) ) ]
    write_file( data = data, path = f'./tables/2_erro_absoluto_medio_{m}.txt' )

# plot error history for jacobi and gauss-seidel
for m in _RESULT:
    x_list = [ ]
    y_list = [ ]
    label_list = [ ]
    color_list = [ ]

    solver1 = _RESULT[ m ][ 'JACOBI' ]
    solver2 = _RESULT[ m ][ 'GAUSS-SEIDEL' ]

    check = 0
    n_iter1 = 0
    n_iter2 = 0
    if solver1.converged:
        history = np.log10( np.asarray( solver1.error_history ) )
        n_iter1 = len( history )
        y_list.append( history )
        x_list.append( [ i + 1 for i in range( len( history ) ) ] )
        label_list.append( 'JACOBI' )
        color_list.append( _COLORS[ 'JACOBI' ] )
        check += 1

        data = [ str( "{:.32f}".format( history[ i ] ) ) + f'\t{i + 1}\n' for i in range( len( history ) ) ]
        write_file( data = data, path = f'./tables/3_erro_absoluto_medio_iteracao_jacobi_{m}.txt' )

    if solver2.converged:
        history = np.log10( np.asarray( solver2.error_history ) )
        n_iter2 = len( history )
        y_list.append( history )
        x_list.append( [ i + 1 for i in range( len( history ) ) ] )
        label_list.append( 'GAUSS-SEIDEL' )
        color_list.append( _COLORS[ 'GAUSS-SEIDEL' ] )
        check += 1

        data = [ str( "{:.32f}".format( history[ i ] ) ) + f'\t{i + 1}\n' for i in range( len( history ) ) ]
        write_file( data = data, path = f'./tables/3_erro_absoluto_medio_iteracao_gauss_seidel_{m}.txt' )

    if check > 0:
        n_iter = max( n_iter1, n_iter2 )
        f = p.line_plot( x_list = x_list,
                         y_list = y_list,
                         labels = label_list,
                         colors = color_list,
                         y_label = 'Erro Absoluto Médio [log10]',
                         x_label = 'Iteração',
                         title = f'Erro Absoluto Médio x Iteração ({m})' )
        f.savefig( f'./figures/3_erro_absoluto_medio_iteracao_jacobi_gauss_seidel_{m}.png' )

var = 0