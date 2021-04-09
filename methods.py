import numpy as np
import copy
from timeit import default_timer as timer
from error import absolute_error, mean_absolute_error


class gauss_jordan:
    def __init__( self ):
        self._A = None
        self._b = None
        self._n = None
        self._solution = None
        self._time = None
        self._true_error = None
        self._status = False

    def solve( self, A: np.ndarray, b: np.ndarray ):
        self._A = copy.deepcopy( A )
        self._n = self._A.shape[ 0 ]
        self._b = b
        self._solution = np.full( self._n, 0. )

        tmp_A = copy.deepcopy( self._A )
        tmp_b = copy.deepcopy( self._b )

        ti = timer( )
        # cancel the elements located below the main diagonal
        for j in range( self._n - 1 ):
            for i in range( j + 1, self._n, 1 ):
                mij = tmp_A[ i, j ] / tmp_A[ j, j ]
                tmp_b[ i ] = tmp_b[ i ] - mij * tmp_b[ j ]

                for k in range( j, self._n ):
                    tmp_A[ i, k ] = tmp_A[ i, k ] - mij * tmp_A[ j, k ]

        # cancel the elements located above the main diagonal
        for j in range( self._n - 1, 0, -1 ):
            for i in range( j - 1, -1, -1 ):
                mij = tmp_A[ i, j ] / tmp_A[ j, j ]
                tmp_b[ i ] = tmp_b[ i ] - mij * tmp_b[ j ]
                tmp_A[ i, j ] = tmp_A[ i, j ] - mij * tmp_A[ j, j ]

        # set each element of the main diagonal equal to 1
        for i in range( self._n ):
            self._solution[ i ] = tmp_b[ i ] / tmp_A[ i, i ]
        tf = timer( )
        self._time = tf - ti

        self._status = True
        return self._solution

    def calculate_true_error( self, reference: np.ndarray ):
        self._true_error = mean_absolute_error( solution = self._solution, reference = reference )
        return self._true_error

    @property
    def time( self ):
        return self._time

    @property
    def solution( self ):
        return self._solution

    @property
    def true_error( self ):
        return self._true_error

    @property
    def converged( self ):
        return self._status


class gauss_seidel:
    def __init__( self, guess: np.ndarray, max_error: float = 1.e-3, max_iter: int = 1000 ):
        self._A = None
        self._b = None
        self._n = None
        self._solution = None
        self._guess = guess
        self._max_error = max_error
        self._error = None
        self._max_iter = max_iter
        self._n_iter = None
        self._time = None
        self._true_error = None
        self._status = False

    def solve( self, A: np.ndarray, b: np.ndarray ):
        self._A = copy.deepcopy( A )
        self._n = self._A.shape[ 0 ]
        self._b = b

        tmp_A = copy.deepcopy( self._A )
        tmp_b = copy.deepcopy( self._b )

        solutions = [ self._guess ]
        error = 1.
        x = None
        n_iter = 0

        ti = timer( )
        while error > self._max_error and n_iter <= self._max_iter:
            x = copy.deepcopy( solutions[ -1 ] )
            xi = solutions[ -1 ]
            for i in range( self._n ):
                count = tmp_b[ i ] / tmp_A[ i, i ]
                for j in range( self._n ):
                    if i != j:
                        count += ( - tmp_A[ i, j ] / tmp_A[ i, i ] ) * x[ j ]
                x[ i ] = count

            error = absolute_error(solution = xi, reference= x)
            solutions.append( x )
            n_iter += 1
        tf = timer( )
        self._time = tf - ti

        if n_iter != self._max_iter + 1:
            self._status = True

        self._solution = x
        self._error = error
        self._n_iter = n_iter
        return self._solution

    def calculate_true_error( self, reference: np.ndarray ):
        self._true_error = mean_absolute_error( solution = self._solution, reference = reference )
        return self._true_error

    @property
    def time( self ):
        return self._time

    @property
    def n_iter(self):
        return self._n_iter

    @property
    def error(self):
        return self._error

    @property
    def solution( self ):
        return self._solution

    @property
    def true_error( self ):
        return self._true_error

    @property
    def converged( self ):
        return self._status


class jacobi:
    def __init__( self, guess: np.ndarray, max_error: float = 1.e-3, max_iter: int = 1000 ):
        self._A = None
        self._b = None
        self._n = None
        self._solution = None
        self._guess = guess
        self._max_error = max_error
        self._error = None
        self._max_iter = max_iter
        self._n_iter = None
        self._time = None
        self._true_error = None
        self._status = False

    def solve( self, A: np.ndarray, b: np.ndarray ):
        self._A = copy.deepcopy( A )
        self._n = self._A.shape[ 0 ]
        self._b = b

        tmp_A = copy.deepcopy( self._A )
        tmp_b = copy.deepcopy( self._b )

        solutions = [ self._guess ]
        error = 1.
        x = None
        n_iter = 0

        ti = timer( )
        while error > self._max_error and n_iter <= self._max_iter:
            x = np.full( self._n, 0. )
            xi = solutions[ -1 ]
            for i in range( self._n ):
                count = tmp_b[ i ] / tmp_A[ i, i ]
                for j in range( self._n ):
                    if i != j:
                        count += ( - tmp_A[ i, j ] / tmp_A[ i, i ] ) * xi[ j ]
                x[ i ] = count

            error = absolute_error(solution = xi, reference= x)
            solutions.append( x )
            n_iter += 1
        tf = timer( )
        self._time = tf - ti

        if n_iter != self._max_iter + 1:
            self._status = True

        self._solution = x
        self._error = error
        self._n_iter = n_iter
        return self._solution

    def calculate_true_error( self, reference: np.ndarray ):
        self._true_error = mean_absolute_error( solution = self._solution, reference = reference )
        return self._true_error

    @property
    def time( self ):
        return self._time

    @property
    def n_iter( self ):
        return self._n_iter

    @property
    def error( self ):
        return self._error

    @property
    def solution( self ):
        return self._solution

    @property
    def true_error( self ):
        return self._true_error

    @property
    def converged( self ):
        return self._status


class gaussian_elimination:
    def __init__( self ):
        self._A = None
        self._b = None
        self._n = None
        self._solution = None
        self._time = None
        self._true_error = None
        self._status = False

    def solve( self, A: np.ndarray, b: np.ndarray ):
        self._A = copy.deepcopy( A )
        self._n = self._A.shape[ 0 ]
        self._b = b
        self._solution = np.full( self._n, 0. )

        tmp_A = copy.deepcopy( self._A )
        tmp_b = copy.deepcopy( self._b )

        ti = timer( )
        for j in range( self._n - 1 ):
            for i in range( j + 1, self._n, 1 ):
                mij = tmp_A[ i, j ] / tmp_A[ j, j ]
                tmp_b[ i ] = tmp_b[ i ] - mij * tmp_b[ j ]

                for k in range( j, self._n ):
                    tmp_A[ i, k ] = tmp_A[ i, k ] - mij * tmp_A[ j, k ]

        for i in range( self._n - 1, -1, -1 ):
            count = tmp_b[ i ] / tmp_A[ i, i ]
            for j in range( self._n - 1, i, -1 ):
                count += ( - tmp_A[ i, j ] / tmp_A[ i, i ] ) * self._solution[ j ]
            self._solution[ i ] = count
        tf = timer( )
        self._time = tf - ti

        self._status = True
        return self._solution

    def calculate_true_error( self, reference: np.ndarray ):
        self._true_error = mean_absolute_error( solution = self._solution, reference = reference )
        return self._true_error

    @property
    def time( self ):
        return self._time

    @property
    def solution( self ):
        return self._solution

    @property
    def true_error( self ):
        return self._true_error

    @property
    def converged( self ):
        return self._status


class lu_decomposition:
    def __init__( self ):
        self._A = None
        self._b = None
        self._n = None
        self._solution = None
        self._time = None
        self._L = None
        self._U = None
        self._true_error = None
        self._status = False

    def solve( self, A: np.ndarray, b: np.ndarray ):
        self._A = copy.deepcopy( A )
        self._n = self._A.shape[ 0 ]
        self._b = b
        self._solution = np.full( self._n, 0. )
        self._L = np.identity( self._n )

        tmp_A = copy.deepcopy( self._A )
        tmp_b = copy.deepcopy( self._b )

        ti = timer( )
        for j in range( self._n - 1 ):
            for i in range( j + 1, self._n, 1 ):
                mij = tmp_A[ i, j ] / tmp_A[ j, j ]
                self._L[ i, j ] = mij

                for k in range( j, self._n ):
                    tmp_A[ i, k ] = tmp_A[ i, k ] - mij * tmp_A[ j, k ]
        self._U = tmp_A

        # solving for L
        for i in range( self._n ):
            count = tmp_b[ i ] / self._L[ i, i ]
            for j in range( i ):
                count += ( - self._L[ i, j ] / self._L[ i, i ] ) * self._solution[ j ]
            self._solution[ i ] = count

        # solving for U
        for i in range( self._n - 1, -1, -1 ):
            count = self._solution[ i ] / self._U[ i, i ]
            for j in range( self._n - 1, i, -1 ):
                count += ( - self._U[ i, j ] / self._U[ i, i ] ) * self._solution[ j ]
            self._solution[ i ] = count
        tf = timer( )
        self._time = tf - ti

        self._status = True
        return self._solution

    def calculate_true_error( self, reference: np.ndarray ):
        self._true_error = mean_absolute_error( solution = self._solution, reference = reference )
        return self._true_error

    @property
    def L( self ):
        return self._L

    @property
    def U( self ):
        return self._U

    @property
    def time( self ):
        return self._time

    @property
    def solution( self ):
        return self._solution

    @property
    def true_error( self ):
        return self._true_error

    @property
    def converged( self ):
        return self._status