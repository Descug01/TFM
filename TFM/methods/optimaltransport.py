# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:18:14 2019

@author: David

MIT License

Copyright (c) 2016 Rémi Flamary

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Generalized gradient adapted from the above.
"""

import tensorflow as tf
import numpy as np
import functools
from ot.bregman import sinkhorn
from ot.optim import line_search_armijo
import scipy as sp
import sklearn.metrics.pairwise
import sklearn.neighbors
from .util import graph


class OptimalTransport():
    '''
    Argumentos
    ----------------------------
    num_iters_gradient: int.
        Número de iteraciones del algoritmo GCG.
    num_iter_sinkhorn: int.
        Número de iteraciones del algoritmo de Sinkhorn-Knopp para obtener
        gamma.
    labda: float.
        Constante de tradeoff que controla la importancia del término de
        regularización entrópico.
    eta: float.
        Constante de tradeoff que controla la importancia del término
        de regularización de clases.
    convergence_threshold: float.
        Constante que determina la convergencia del algoritmo GCG.
    regularizer: string.
        Indica el tipo de regularización de clases a aplicar. Hay dos
        valores posibles 'group sparse' y 'laplacian'.
    alpha: float. opcional (default=0.5).
        Constante de tradeoff para el regularizador 'laplacian'. Controla
        la importancia que se le da a la laplaciana del dominio objetivo
        respecto a la del fuente.
    n_neighbors: int. opcional (default=8).
        Número de vecinos más cercanos usados para calcular la laplaciana
        en la regularización 'laplacian'.

    Atributos
    ---------------------------------
    graph: tensorflow.Graph
        Grafo de computación que define las operaciones del modelo.
    gamma: numpy.ndarray shape(m, n)
        Matriz de transporte de los datos.
    '''
    def __init__(self, num_iters_gradient, num_iters_sinkhorn, labda, eta,
                 convergence_threshold=1e-4, regularizer='group sparse',
                 alpha=0.5, n_neighbors=8):
        self.labda = labda
        self.eta = eta
        self.num_iters_gradient = num_iters_gradient
        self.num_iters_sinkhorn = num_iters_sinkhorn
        self.convergence_threshold = convergence_threshold
        self.regularizer = regularizer
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        self.graph = None

    def regularizer_and_gradient(self, regularizer_func, *args, **kwargs):
        '''
        Genera las operaciones de cálculo de la regularización de clases
        seleccionada así como su gradiente.

        Argumentos
        -----------------------------
        regularizer_func: function.
            Función con la que calcular el término de regularización.
        *args: tuple.
            Argumentos posicionales para `regularizer_func`.
        **kwargs: dict.
            Argumentos por keyword para `regularizer_func`

        Retorno
        -------------------------------
        regularization_value: Tensor shape [0]
            Tensor que calcula el valor del término de regularización.
        gradient. Tensor shape(n (instancias fuente), m(instancias objetivo))
            Tensor que calcula el gradiente del término de regularización
            respecto a gamma.
        '''
        regularization_value = regularizer_func(*args, **kwargs)
        optimizer = tf.train.AdamOptimizer()
        gradient = optimizer.compute_gradients(
            regularization_value
        )[0][0]

        return regularization_value, gradient

    def laplacian(self, data, labels=None):
        '''
        Calcula una matriz laplaciana de conectividad para la regularización de
        clases 'laplacian'.

        Si se pasa `labels` como argumento la laplaciana solo considerará
        los vecinos de la misma clase. En caso contrario, solo se tiene en
        cuenta la cercanía.

        Argumentos
        ---------------------------
        data: numpy.ndarray shape(l, k).
            Datos sobre los que calcular la laplaciana.
        labels: numpy.ndarray shape(l, ). opcional (default=None)
            etiquetas correspondientes a `data`.

        Retorno
        -----------------------------------
        laplacian: numpy.ndarray shape(l, l)
            Matriz laplaciana de los datos.
        '''
        distances = sp.spatial.distance.cdist(data, data, metric='euclidean')
        mask = graph.nearest_same_class_neighbors(
                distances, labels, n_neighbors=self.n_neighbors
        )
        distances[~mask] = 0
        distances[mask] = 1
        return graph.laplacian(distances)

    def source_product(self, target, tf_gamma, source_laplacian):
        '''
        Calcula la parte correspondiente al dominio fuente de la regularización
        de clases 'laplacian'.

        Argumentos
        -----------------------------------------
        source: numpy.ndarray shape(m, k).
            datos del dominio fuente.
        tf_gamma: Tensor shape(m, n)
            tensor que calcula la matriz de transporte gamma.
        source_laplacian: numpy.ndarray shape(m, m)
            Matriz laplaciana del dominio fuente.

        Retorno
        ---------------------------------
        source_value: tensor shape [0]
            El valor de la parrte del dominio fuente del regularizador.
        '''
        first = tf.matmul(target, tf_gamma, transpose_a=True, transpose_b=True)
        second = tf.matmul(source_laplacian, tf_gamma)
        second = tf.matmul(second, target)

        return tf.linalg.trace(tf.matmul(first, second))

    def target_product(self, source, tf_gamma, target_laplacian):
        '''
        Calcula la parte correspondiente al dominio objetivo de la
        regularización de clases 'laplacian'.

        Argumentos
        -----------------------------------------
        target: numpy.ndarray shape(n, k).
            datos del dominio objetivo.
        tf_gamma: Tensor shape(m, n)
            tensor que calcula la matriz de transporte gamma.
        source_laplacian: numpy.ndarray shape(n, n)
            Matriz laplaciana del dominio fuente.

        Retorno
        ---------------------------------
        target_value: tensor shape [0]
            El valor de la parrte del dominio objetivo del regularizador.
        '''
        return self.source_product(
                source, tf.transpose(tf_gamma), target_laplacian
        )

    def laplacian_regularizer(self, tf_gamma, source, target, labels):
        '''
        Calcula el valor del regularizador de clases 'laplacian'


        Argumentos
        -------------------------------
        source: numpy.ndarray shape(m, k).
            datos del dominio fuente.
        target: numpy.ndarray shape(n, k).
            datos del dominio objetivo.
        tf_gamma: Tensor shape(m, n)
            tensor que calcula la matriz de transporte gamma.
        labels: Tensor shape(m, )
            etiquetas del dominio fuente.

        Retorno
        --------------------------
        regularization_value: Tensor shape [0]
            El valor del término de regularización.
        '''
        source_laplacian = self.laplacian(source, labels)
        target_laplacian = self.laplacian(target)
        source_factor = self.source_product(target, tf_gamma, source_laplacian)
        target_factor = self.target_product(source, tf_gamma, target_laplacian)
        regularization_value = (
                (1 - self.alpha) * source_factor + self.alpha * target_factor
        )

        return regularization_value

    def group_regularizer(self, tensor_gamma, tensor_labels):
        '''
        Calcula el término de regularización 'group_sparse'.

        Argumentos:
        ------------------------------------
        tensor_gamma : tf.Variable shape [n, m].
            Valor de la matriz de transporte gamma
        tensor_labels: tf.Variable shape [n, ].
            Etiquetas del dominio fuente.

        Retorno
        ----------------------------------------
        regularization_value: tf.tensor shape [0].
            El valor del término de regularización
        '''
        unique_labels = tf.unique(tensor_labels)[0][:, tf.newaxis, tf.newaxis]
        comparison_labels = tensor_labels[:, tf.newaxis]
        equality = tf.cast(
            tf.equal(unique_labels, comparison_labels), tf.float64
        )
        coefficients = tensor_gamma * equality
        norms = tf.norm(coefficients, axis=1)
        regularization_value = tf.reduce_sum(norms)

        return regularization_value

    def tf_entropic_regularization(self, tf_gamma):
        '''
        Calcula el término de regularización entrópica.
        Se calcula como:
        $$\sum_{ij} gamma_{ij} * log(gamma_{ij})$$

        Argumentos
        ----------------------------
        tf_gamma : tf.Variable shape [n, m].
            Valor de la matriz de transporte gamma.

        Retorno
        -----------------------------
        entropic: Tensor shape [0]
            Valor del término de regularización entrópica.
        '''
        return tf.reduce_sum(tf_gamma * tf.log(tf_gamma))

    def cost_transport_matrix(self, source, target):
        '''
        Calcula la matriz de coste de transporte entre los dominios fuente y
        objetivo.

        Esta matriz simplemente contiene las distancias euclidianas para cada
        par <source, target>.

        Argumentos
        ---------------------------
        source: numpy.ndarray shape(m, k).
            datos del dominio fuente.
        target: numpy.ndarray shape(n, k).
            datos del dominio objetivo.

        Retorno
        ---------------------------
        cost_matrix: numpy.ndarray shape(m, n)
            La matriz de coste. cost_matrix[i, j] contiene la distancia
            euclidiana entre la instancia `i` de `source` y la instancia `j` de
            `target`.
        '''
        return sklearn.metrics.pairwise.euclidean_distances(
                    source, target, squared=True
        )

    def compute_probs(self, data):
        '''
        Calcula la estimación base de la distribución de un dominio.

        Esto es: un vector con un número de elementos igual al número de
        instancias del dominio y cuyos valores son 1 / <número de instancias>
        para todos los elementos.

        Argumentos
        ------------------------------
        data: numpy.ndarray shape(l, k)
            Conjunto de datos.

        Retorno
        --------------------------------
        probabilities: numpy.ndarray shape(l, )
            Array con el valor 1 / l en todas las posiciones.
        '''
        n = data.shape[0]
        return np.ones(n) / n

    def tf_frobenius_product(self, tf_gamma, tf_cost):
        '''
        Producto de Frobenius con tensores.

        El producto de Frobenius de dos matrices A y B se escribe como:
        $$\sum_{ij} A_{ij} * B_{ij}$$

        Argumentos
        ---------------------------------
        tf_gamma: Tensor shape(m, n)
            Matriz gamma de transporte.
        tf_cost: Tensor shape(m, n)
            matriz de coste de transporte.

        Retorno
        --------------------------------
        product: Tensor shape [0]
            El producto de Frobenius entre `tf_gamma` y `tf_cost`
        '''
        return tf.reduce_sum(tf_gamma * tf_cost)

    def computation_graph(self, source, target, source_labels):
        '''
        Crea el grafo de computación del modelo, con el cálculo del coste,
        el regularizador de clases y sus gradientes respecto de gamma.

        Argumentos
        -------------------------------
        source: numpy.ndarray shape(m, k).
            datos del dominio fuente.
        target: numpy.ndarray shape(n, k).
            datos del dominio objetivo.
        source_labels: numpy.ndarray shape(m, )
            etiquetas del dominio fuente.

        Retorno
        -----------------------------------
        cost: Tensor shape[0].
            El valor a optimizar.
        cost_gradient: Tensor shape (m, n).
            El gradiente e `cost` respecto a gamma.
        regularization: Tensor shape [0]
            El valor de la regularización de clases.
        regularization_gradient: Tensor shape (m, n)
            El gradiente de `regularization` respecto a gamma.
        '''
        gamma_placeholder = tf.placeholder(tf.float64)
        tf_gamma = tf.get_variable(
                'gamma', initializer=gamma_placeholder,
                trainable=True, validate_shape=False
        )
        regularization, regularization_gradient = self.process_regularizer(
                tf_gamma, source_labels, source, target
        )

        M = self.cost_transport_matrix(source, target)
        first_term = self.tf_frobenius_product(tf_gamma, M)
        entropic = self.tf_entropic_regularization(tf_gamma)
        optimization_value = (
                first_term + self.labda * entropic
                + self.eta * regularization
        )
        adam_opt = tf.train.AdamOptimizer()
        optimization_gradient = adam_opt.compute_gradients(
                optimization_value, var_list=[tf_gamma]
        )[0][0]

        def execute(gamma, operation=None):
            session = tf.Session(graph=self.graph)
            session.run(
                tf.global_variables_initializer(),
                feed_dict={gamma_placeholder: gamma}
            )
            val = session.run(operation)
            session.close()
            return val

        cost = functools.partial(execute, operation=optimization_value)
        cost_gradient = functools.partial(
                execute, operation=optimization_gradient
        )
        regularization = functools.partial(execute, operation=regularization)
        regularization_gradient = functools.partial(
                execute, operation=regularization_gradient
        )

        return cost, cost_gradient, regularization, regularization_gradient

    def generalized_conditional_gradient(self, source, target, source_labels):
        '''
        Algoritmo GCG: gradiente condicional generalizado. Calcula la matriz
        de transporte gamma.

        Argumentos
        --------------------------------
        source: numpy.ndarray shape(m, k).
            datos del dominio fuente.
        target: numpy.ndarray shape(n, k).
            datos del dominio objetivo.
        source_labels: numpy.ndarray shape(m, )
            etiquetas del dominio fuente.

        Retorno
        -----------------------------------
        G: numpy.ndarray shape(m, n)
            Matriz gamma de transporte.
        '''
        loop = True
        a, b = self.compute_probs(source), self.compute_probs(target)
        G = np.outer(a, b)
        M = self.cost_transport_matrix(source, target)
        cost, cost_gradient, f, df = self.computation_graph(
                source, target, source_labels
        )
        f_val = cost(G)
        it = 0
        while loop:
            it += 1
            old_fval = f_val

            # problem linearization
            Mi = M + self.eta * df(G)

            # solve linear program with Sinkhorn
            Gc = sinkhorn(
                    a, b, Mi, self.labda, numItermax=self.num_iters_sinkhorn
            )
            deltaG = Gc - G
            # line search
            # Que yo sepa la evaluación de la función se hace con G, no con
            # Gc
            dcost = cost_gradient(G)
            alpha, fc, f_val = line_search_armijo(
                    cost, G, deltaG, dcost, f_val
            )
            if alpha is None:  # No se si esto es lo correcto
                return G
            G = G + alpha * deltaG

            # test convergence
            if it >= self.num_iters_sinkhorn:
                loop = False

            # delta_fval = (f_val - old_fval) / abs(f_val)
            delta_fval = (f_val - old_fval)
            # print('fval:', f_val, 'delta_fval:', delta_fval)
            if abs(delta_fval) < self.convergence_threshold:
                loop = False
        return G

    def process_regularizer(self, tf_gamma, tf_labels, source, target):
        '''
        Devuelve dos tensores que calculan el valor de la regularización de
        clases y su gradiente respecto a gamma.

        Argumentos
        ------------------------------------
        tf_gamma: Tensor shape(m, n)
            tensor que calcula la matriz de transporte gamma.
        tf_labels: Tensor shape(m, )
            etiquetas del dominio fuente.
        source: numpy.ndarray shape(m, k).
            datos del dominio fuente.
        target: numpy.ndarray shape(n, k).
            datos del dominio objetivo.

        Retorno
        -----------------------------------
        reg: Tensor shape [0]
            Valor de la regularización de clases.
        gradient: Tensor shape(m, n)
            Gradiente de `reg` respecto de `tf_gamma`
        '''
        if self.regularizer == 'group sparse':
            reg, gradient = self.regularizer_and_gradient(
                    self.group_regularizer, tf_gamma, tf_labels
            )
        elif self.regularizer == 'laplacian':
            reg, gradient = self.regularizer_and_gradient(
                    self.laplacian_regularizer, tf_gamma, source, target,
                    tf_labels
            )

        return reg, gradient

    def fit(self, source, target, source_labels):
        '''
        Computa la matriz gamma de transporte para la transformación.

        Argumentos
        ----------------------------------
        source: numpy.ndarray shape(m, k).
            datos del dominio fuente.
        target: numpy.ndarray shape(n, k).
            datos del dominio objetivo.
        source_labels: numpy.ndarray shape(m, )
            etiquetas del dominio fuente.

        Retorno
        -------------------------------
        self: OptimalTransport
            La propia instancia
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.gamma = self.generalized_conditional_gradient(
                    source, target, source_labels
            )
        return self

    def transform(self, source, target):
        '''
        Transforma los datos. Tal y como enfocado este método, solo tiene
        sentido transformar los datos fuente, así que el dominio objetivo no se
        modifica de ningún modo.

        Argumentos
        ------------------------------
        source: numpy.ndarray shape(m, k).
            datos del dominio fuente.
        target: numpy.ndarray shape(n, k).
            datos del dominio objetivo.

        Retorno
        ------------------------------
        transformed_source: numpy.ndarray shape(m, k)
            Los datos fuente transformados
        target: numpy.ndarray shape(n, k)
            Las datos objetivo tal cual se pasaron como argumento.
        '''
        return (
                 self.transform_source(source, target),
                 target
        )

    def transform_source(self, source, target):
        '''
        Transforma el dominio fuente

        Argumentos
        --------------------------------
        source: numpy.ndarray shape(m, k).
            datos del dominio fuente.
        target: numpy.ndarray shape(n, k).
            datos del dominio objetivo.

        Retorno
        ------------------------------
        transformed_source: numpy.ndarray shape(m, k)
            Los datos fuente transformados
        '''
        return source.shape[0] * self.gamma.dot(target)

    def transform_target(self, source, target):
        '''
        Transforma el dominio objetivo

        Argumentos
        --------------------------------
        source: numpy.ndarray shape(m, k).
            datos del dominio fuente.
        target: numpy.ndarray shape(n, k).
            datos del dominio objetivo.

        Retorno
        ------------------------------
        transformed_targete: numpy.ndarray shape(n, k)
            Los datos objetivo transformados
        '''
        return target.shape[0] * self.gamma.T.dot(source)

    def fit_transform(self, source, target, source_labels):
        '''
        Combina fit y transform.

        Argumentos
        ----------------------------------
        source: numpy.ndarray shape(m, k).
            datos del dominio fuente.
        target: numpy.ndarray shape(n, k).
            datos del dominio objetivo.
        source_labels: numpy.ndarray shape(m, )
            etiquetas del dominio fuente.

        Retorno
        ------------------------------
        transformed_source: numpy.ndarray shape(m, k)
            Los datos fuente transformados
        target: numpy.ndarray shape(n, k)
            Las datos objetivo tal cual se pasaron como argumento.
        '''
        self.fit(source, target, source_labels)
        return self.transform(source, target)
