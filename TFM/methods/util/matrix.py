# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp


def simmetrize(matrix):
    '''
    Simetriza una matriz M. Matemáticamente esta transformación implica lo
    siguiente:
    $$M_{i, j} = max(M^T_{i, j}, M_{i, j})$$
    Es decir, para cada elemnto se toma el máximo entre ese elemento y el
    elemento en la misma posición en la matriz transpuesta.

    Argumentos
    ------------------------
    matrix: numpy.ndarray shape(m, n).
        Matriz a simetrizar.

    Retorno
    -------------------------
    simmetrized: numpy.ndarray shape(m, n)
        `matrix` simetrizada.
    '''
    return np.max(np.array([matrix, matrix.T]), axis=0)


def construct_L(num_source, num_target):
    '''
    Crea la matriz L con coeficientes que ayudan a calcular la versión
    kernelizada de la MMD. L se define como:
                   1 / (n_S ^ 2) si i, j pertenecen a fuente
        L[i, j] =  1 / (n_T ^ 2) si i, j pertenecen a target
                  -1 / (n_S + n_T) en caso contrario

    Argumentos
    --------------------------
    num_instances_source: int.
        Número de instancias en el dominio fuente.
    num_instances_target: int.
        Número de instancias en el dominio objetivo.

    Retorno
    -------------------
    L: numpy.ndarray shape (num_source + num_target, num_source + num_target).
        Matriz de coeficientes.
    '''
    # se inicializa todo al caso default
    L = - np.ones((num_source + num_target, ) * 2)
    L /= num_source * num_target

    # se fija el caso fuente
    select = (slice(num_source), ) * 2
    L[select] *= - num_target / num_source

    # se fija el caso objetivo
    select = (slice(-num_target, L.shape[0]), ) * 2
    L[select] *= - num_source / num_target

    return L


def l21_norm(matrix):
    '''
    Calcula la norma l2,1 para una matriz.

    Se define como:
        $$||M||_{2,1} = \sum_i \sqrt(sum_j M_{ij})$$

    Argumentos
    -----------------------
    matrix: numpy.ndarray shape(m, n).
        matriz sobre la que se va a calcular la norma.

    Retorno
    --------------------------
    norm: float.
        Valor de la norma.
    '''
    return np.linalg.norm(matrix, axis=1).sum()


def top_eigenvectors(matrix, number=10):
    '''
    Devuelve los `number` eigenvectores de `matrix` con mayor magnitud,
    ordenados de mayor a menor de acuerdo con sus eigenvalues.

    Argumentos
    ---------------------------
    matrix: numpy.ndarray shape (m, m).
        Matriz cuadrada de la que se obtendrán los eigenvectores.
    number: int. opcional (default=10).
        Número de eigenvectores a extraer.

    Retorno
    -----------------------------
    eig: numpy.ndarray shape(m, number). Array con los
        eigenvectores. Cada columna es un eigenvector.
    '''
    _, P = sp.sparse.linalg.eigs(
                matrix,
                k=number,
                which='LM',
                ncv=2 * number + int(number / 2)
    )

    return P.real  # Si no sklearn lanza excepción porque se queja de complex


def centering_matrix(dimension):
    '''
    Crea una matriz cuadrada de centrado M . Esta matriz se define como:
                1 - 1 / `dimension` si i = j
    M[i, j] =   -1 / `dimension`    en caso contrario

    Argumentos
    -----------------
    dimension: int.
         La dimension de la matriz de centrado creada.

    Retorno
    ------------------
    centering: numpy.ndarray shape(dimension, dimension).
        Matriz de centrado.
    '''
    shape = (dimension, ) * 2
    return np.identity(dimension) - (1 / dimension) * np.ones(shape)
