# -*- coding: utf-8 -*-
import sklearn.preprocessing
import sklearn.model_selection
import numpy as np
import itertools


def train_test_split(data, labels, test_fraction=0.9):
    '''
    Divide datos y etiquetas en conjuntos de entrenamiento y test,
    conservando la proporción de elementos de diferentes clases

    Argumentos
    ---------------------------
    data: numpy.ndarray shape(m,k)
        Un conjunto de datos. Se entiende que cada fila es una instancia.
    labels: numpy.ndarray shape(m, )
        Etiquetas correspondientes a los datos

    Retorno
    ----------------------------
    train_split: tuple (
            numpy.ndarray shape((1 - test_fraction)m, k),
            numpy.ndarray shape((1 - test_fraction)m, )
    )
        datos y etiquetas de entrenamiento
    test_split: tuple (
            numpy.ndarray shape(test_fraction * m, k),
            numpy.ndarray shape(test_fraction * m, )
    )
        datos y etiquetas de test
    '''
    train, test = sklearn.model_selection.train_test_split(
            np.hstack((data, labels[:, None])), stratify=labels,
            test_size=test_fraction
    )
    return (train[:, :-1], train[:, -1]), (test[:, :-1], test[:, -1])


def reduce_to_unit_ball(data):
    '''
    Normaliza (divide cada instancia entre su norma) un conjunto de datos.

    Argumentos
    ---------------------------
    data: numpy.ndarray shape(m,k)
        Un conjunto de datos. Se entiende que cada fila es una instancia.

    Retorno
    ----------------------------
    reduced_data: numpy.ndarray shape(m, k)
        data normalizada.
    '''
    return data / np.linalg.norm(data, axis=1, keepdims=True)


def normalize(source, target):
    '''
    Función de conveniencia que aplica reduce_to_unit_ball a dos conjuntos
    de datos.

    Argumentos
    ---------------------------
    source: numpy.ndarray shape(m,k)
        Un conjunto de datos. Se entiende que cada fila es una instancia.
    target: numpy.ndarray shape(n, l)
        Un conjunto de datos. Se entiende que cada fila es una instancia.

    Retorno
    ----------------------------
    reduced_source: numpy.ndarray shape(m, k)
        source normalizada.
    reduced_target: numpy.ndarray shape(n, l)
        target normalizado.
    '''
    return (
        reduce_to_unit_ball(source),
        reduce_to_unit_ball(target)
    )


def standarize(source, target):
    '''
    Estandariza (resta media y divide entre desviación típica) dos conjuntos de
    datos. La media y desviación típica se obtiene únicamente del argumento
    target.

    Argumentos
    ------------------------
    source: numpy.ndarray shape(m,k)
        Un conjunto de datos. Se entiende que cada fila es una instancia.
    target: numpy.ndarray shape(n, l)
        Un conjunto de datos. Se entiende que cada fila es una instancia.

    Retorno
    ----------------------------
    standarized_source: numpy.ndarray shape(m, k)
        source estandarizada.
    standarized_target: numpy.ndarray shape(n, l)
        target estandarizado.
    '''
    sc = sklearn.preprocessing.StandardScaler().fit(target)
    return sc.transform(source), sc.transform(target)


def binarize_labels(labels):
    '''
    Binariza un array de etiquetas. La binarización consiste en transformar
    cada etiqueta en un array de longitud `c` donde `c` es el número de
    elementos únicos. El vector tendrá tendrá el valor ``1` en la posición
    correspondiente a su etiqueta y `0` en las demás.

    Ej: labels[2,1,3,2] -> [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]

    Argumentos
    ---------------------------
    labels: numpy.ndarray shape(n, )
        Array de etiquetas.

    Retorno
    ---------------------------
    binarized_labels: numpy.ndarray shape(n, c)
        Etiquetas binarizadas.
    '''
    labels_reshaped = labels.reshape(-1, 1)
    binarizer = sklearn.preprocessing.OneHotEncoder(
                sparse=False
    ).fit(labels_reshaped)
    return binarizer.transform(labels_reshaped)


def split_data_label(ndarray, target=None, shuffle=True):
    '''
    Separa una columna de un array de numpy. Devuelve por separado los datos de
    ndarray sin la columna `target` y la columna `target` separada.

    Argumentos
    -------------------------
    df: pandas.DataFrame shape (n, k).
        Datos originales.
    target: object. optional (default=None)
        Columna de df a separar. Si es `None` se toma la
        última columna según el orden en df.columns.
    shuffle: boolean. optional (default=True).
        Determina si se randomiza el orden de las filas de `df`.

    Retorno
    -------------------------
    without_target: numpy.ndarray shape(n, k - 1).
        Datos de `df` sin la columna target.
    target_column: numpy.ndarray shape(n, ).
        Los datos de la columna target.
    '''
    if shuffle:
        np.random.shuffle(ndarray)
    if not target:
        target = ndarray.shape[1] - 1
    cols = [i for i in range(ndarray.shape[1]) if i != target]
    return ndarray[:, cols], ndarray[:, target]


def structured_split(data, labels, fraction=0.1):
    '''
    Realiza una partición estructurada de los datos y las etiquetas pasados
    como argumento. La partición estructurada implica que se realiza una
    partición aleatoria que conserva la proporción de las clases en los
    datos originales.

    La fracción pasada como argumento es la proporción de datos de
    entrenamiento que se extraerá.

    Argumentos
    -----------------------
    data: numpy.ndarray shape (n, k).
        Datos originales. Se entiende que cada fila es una instancia.
    labels: numpy.ndarray shape(n, ).
        Etiquetas correspondientes a `data`. Se considera que labels[i] es la
        etiqueta correspondiente a la instancia data[i].
    fraction: float 0 < fraction < 1.
        Tanto por uno de datos que formarán parte del conjunto de
        entrenamiento.

    Retorno
    --------------------------
    train: numpy.ndarray shape(~fraction * n, k).
         Conjunto de datos para el entrenamiento.
    train_labels: numpy.ndarray shape(~fraction * n, ).
         Etiquetas correspondiente a las instancias de `train`.
    test: numpy.ndarray shape(~(1 -fraction * n), k).
         Conjunto de datos para el test.
    test_labels: numpy.ndarray shape(~(1 - fraction * n), ).
         Etiquetas correspondiente a las instancias de `test`.
    '''
    stra = sklearn.model_selection.StratifiedKFold(round(fraction * 100))
    gen = stra.split(data, labels)
    train_indices, test_indices = next(gen)
    train = data[test_indices]
    train_labels = labels[test_indices]
    test = data[train_indices]
    test_labels = labels[train_indices]

    return train, train_labels, test, test_labels
