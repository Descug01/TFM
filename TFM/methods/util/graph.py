# -*- coding: utf-8 -*-
import numpy as np
from .matrix import simmetrize


def nearest_same_class_neighbors(distances, labels=None, column_labels=None,
                                 n_neighbors=3, reverse=False):
    '''
    Devuelve un array booleano con que indica los vecinos más cercanos de la
    misma clase a cada instancia correspondiente a cada fila de `distances`.

    Si column_labels no es `None` se considera que la matriz `distances`
    contiene las distancias a pares entre dos conjunto de datos distintos, de
    modo que distances[i, j] contiene la distancia de la instancia `i` del
    primer conjunto de datos a la instancia `j` del segundo. En caso contrario
    distances[i, j] contiene las distancias entre las instancias i, j del mismo
    conjunto de datos.

    Argumentos
    ----------------------
    distances: numpy.ndarray shape(m, n).
         Array de distancias entre instancias. distances[i, j] contiene la
         distancia de la instancia i a la instancia j. Cualquier métrica
         que cumpla la definición de distancia es válida como argumento.
    labels: numpy.ndarray shape (m, ). opcional (default=None)
         Etiquetas correspondientes a las instancias de las filas de
         `distances`. Si es `None` entonces se considera que todas las
         instancias de las filas tienen las mismas etiquetas.
    column_labels: numpy.ndarray shape (n, ). opcional (default=None)
         Etiquetas correspondientes a las instancias de las columnas de
         `distances`. Si es `None` entonces se considera que `distances` es una
         matriz cuadrada (m = n) y simétrica. Es decir, que
         distances[i, j] = distances[j, i] y por lo tanto la instancia de la
         fila `i` y la columna `i` son la misma.
    n_neighbors: int. opcional (default=3).
         Número de vecinos más cercanos a obtener.
    reverse: boolean. opcional (default=False).
         Si es `True` se obtienen los vecinos más lejanos en lugar de los más
         cercanos.

    Retorno
    ---------------------------
    nearest_neighbors_boolean_indices: numpy.ndarray<boolean> shape(m, n).
        Array de booleanos que indica los vecinos más cercanos. Si
        nearest_neighbors_boolean_indices[i, j] es `True`entonces la instancia
        `j` es uno de los `n_neighbors` vecinos más cercanos de la instancia
        `i`.
    '''
    nearest_indices = np.argsort(distances, axis=1, kind='mergesort')
    if reverse:
        nearest_indices = nearest_indices[:, ::-1]
    tuple_nearest_indices = (
            np.arange(nearest_indices.shape[0])[:, None], nearest_indices
    )
    if labels is None:
        labels = np.full((distances.shape[0]), True)
    if column_labels is not None:
        label_equality = labels[:, None] != column_labels[None, :]
        offset = 0
    else:
        label_equality = labels[:, None] != labels[None, :]
        offset = 1
    equal_labels = (
                label_equality
    )[tuple_nearest_indices]
    nearest_same_class_indices = np.argsort(
            equal_labels, axis=-1, kind='mergesort'
    )[:, offset:offset + n_neighbors]
    nearest_neighbors_indices = nearest_indices[
            np.arange(nearest_same_class_indices.shape[0])[:, None],
            nearest_same_class_indices
    ]

    nearest_neighbors_index_tuple = (
        np.arange(nearest_neighbors_indices.shape[0])[:, None],
        nearest_neighbors_indices
    )
    nearest_neighbors_boolean_indices = np.full((distances.shape), False)

    nearest_neighbors_boolean_indices[nearest_neighbors_index_tuple] = (
        (~label_equality)[nearest_neighbors_index_tuple]
    )

    return nearest_neighbors_boolean_indices


def adjacency_matrix(distances, n_neighbors, labels=None, reverse=False):
    '''
    Construye una matriz de adyacencia para un conjunto de datos a partir de
    las distancias entre entre las distancias.

    La matriz de adyacencia es una matriz M tal que M[i, j] contiene la
    distancia entre las instancias i, j si j es uno de los vecinos más cercanos
    de la misma clase de i o viceversa. Por lo tanto el resultado será una
    matriz simétrica.

    `distances`debe ser una matriz cuadrada y simétrica.

    Argumentos
    -------------------------
    distances: numpy.ndarray shape(m, m).
         Array de distancias entre instancias. distances[i, j] contiene la
         distancia de la instancia i a la instancia j. Cualquier métrica
         que cumpla la definición de distancia es válida como argumento. Debe
         ser cuadrada y simétrica.
    labels: numpy.ndarray shape (m, ). opcional (default=None)
         Etiquetas correspondientes a las instancias de `distances`. Si es
         `None` entonces se considera que todas las instancias de las filas
         tienen las mismas etiquetas.
    n_neighbors: int. opcional (default=3).
         Número de vecinos más cercanos a obtener.
    reverse: boolean. opcional (default=False).
         Si es `True` se fijan las distancias a los vecinos más lejanos en
         lugar de los más cercanos.

    Retorno
    --------------------------
    adjacency: numpy.ndarray shape(m, m).
         matriz de adyacencia de los datos.
    '''
    mask = nearest_same_class_neighbors(
            distances, labels, n_neighbors=n_neighbors, reverse=reverse
    )
    distances[~mask] = 0
    return simmetrize(distances)


def laplacian(adjacency):
    '''
    Calcula la matriz laplaciana a partir de una matriz de adyacencia.

    La matriz laplaciana L se obtiene a partir de una matriz de adyacencia  M
    del siguiente modo:
    1. Calcular D. D es una matriz diagonal tal que
       $$D_{i,i} = \sum_j M_{i,j}$$.
    2. $$L = D - M$$

    Argumentos
    ------------------------
    adjacency: numpy.ndarray shape(m, m).
         matriz de adyacencia.

    Retorno
    ------------------------
    laplacian: numpy.ndarray shape(m, m).
        matriz laplaciana.
    '''
    D = np.diag(adjacency.sum(axis=1))
    return D - adjacency
