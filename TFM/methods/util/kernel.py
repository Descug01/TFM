# -*- coding: utf-8 -*-
"""
Definición de funciones kernel
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def linear(X, Y=None):
    '''
    Calcula el resultado del kernel lineal para cada par de elmentos de X. Si
    se pasa Y entonces se calcula para cada par de instancias (x, y).

    Se asume que cada columna corresponde a una feature y las filas
    corresponden a una instancia.

    El kernel lineal de dos vectores a y b se define como:
        k(a, b) = <a, b>   (dot product de toda la vida)

    Argumentos
    -----------------------------
    X: numpy.ndarray shape(n, k).
        Matriz de datos 1.
    Y: numpy.ndarray shape(m, k). opcional (default=None)
        Matriz de datos 2.

    Retorno
    ------------------------
    gram: numpy.ndarray shape(n + m, n + m).
        Gram matrix con kernel lineal para cada par de elemntos de X. Si `Y` no
        es `None` entonces es de shape (n, m) para cada par <X, Y>
    '''
    if Y is not None:
        return X.dot(Y.T)
    return X.dot(X.T)


def polynomial(X, Y=None, exponent=2, coefficient=0):
    '''
    Calcula el resultado de un kernel polinomial para cada par de elmentos de
    X. Si se pasa Y entonces se calcula para cada par de instancias (x, y).

    Se asume que cada columna corresponde a una feature y las filas
    corresponden a una instancia.

    El kernel polinomial de dos vectores a y b se define como:
        k(a, b) = (<a, b> + c) ^ p

    Argumentos
    --------------------------------------
    X: numpy.ndarray shape(n, k).
        Matriz de datos 1.
    Y: numpy.ndarray shape(m, k). opcional (default=None)
        Matriz de datos 2.
    exponent: int. opcional (default=2)
        El término 'p' que determina el exponente.
    coefficient: Number.  opcional (default=0)
        El término 'c' que se suma al dot product.

    Retorno
    ---------------------------
    gram: numpy.ndarray shape(n + m, n + m).
        Gram matrix con kernel polinomial para cada par de elemntos de X. Si
        `Y` no es `None` entonces es de shape (n, m) para cada par <X, Y>
    '''
    return (linear(X, Y) + coefficient) ** exponent


def rbf(X, Y=None, sigma=1):
    '''
    Calcula el resultado de un kernel rbf para cada par de elmentos de X. Si
    se pasa Y entonces se calcula para cada par de instancias (x, y).

    Se asume que cada columna corresponde a una feature y las filas
    corresponden a una instancia.

    El kernel rbf de dos vectores a y b se define como:
        k(a, b) = e ^ -((||a - b||_2)^2 / 2 * sigma)

        ||a - b||_2 es la norma euclidiana

    Argumentos
    -------------------------------
    X: numpy.ndarray shape(n, k).
        Matriz de datos 1.
    Y: numpy.ndarray shape(m, k). opcional (default=None)
        Matriz de datos 2.
    sigma: int. opcional (default=1)
        El término 'sigma'.

    Retorno
    -----------------------------------
    gram: numpy.ndarray shape(n + m, n + m).
        Gram matrix con kernel polinomial para cada par de elemntos de X. Si
        `Y` no es `None` entonces es de shape (n, m) para cada par <X, Y>
    '''
    if Y is not None:
        exponent = euclidean_distances(X, Y, squared=True)
    else:
        exponent = euclidean_distances(X, squared=True)
    exponent /= - 2 * sigma

    return np.exp(exponent)


KERNELS = {'linear': linear, 'polynomial': polynomial, 'rbf': rbf}
