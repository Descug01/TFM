# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:17:23 2019

@author: David
"""
import numpy as np
import scipy as sp
from .util import matrix
from .util import graph
from .util import kernel


class TIT():
    '''
    Instancia la clase.

    Argumentos
    --------------------------
    num_features: int.
        Número de features de los datos transformados.
    n_neighbors: int. opcional (default=3)
        Número de vecinos a tener en cuenta a la hora de construir la
        matriz laplaciana para los cálculos.
    labda: float. opcional (default=0.1)
        Constante de tradeoff. Un mayor valor da una mayor importancia a la
        conservación de la estructura original de los datos.
    beta: float. opcional (default=0.01)
        Constante de tradeoff. Un mayor valor ayuda a conservar la
        estructura de los datos del dominio objetivo.
    gamma: float. opcional (default=1.0)
        Constante de tradeoff. Un mayor valor fuerza la regularización de
        la matriz de proyección de los datos.
    max_iters: int. opcional (default=10)
        Máximo número de iteraciones para resolver el problema de
        optimización.
    landmarks: boolean. (default=False)
        Indica si usar el procedimiento de obtención de landmarks para
        obtener pesos óptimos para el entrenamiento de un modelo.
    epsilon: float
        umbral de convergencia

    Atributos
    -----------------------------------
    K: numpy.ndarray shape(m +n, m + n)
        Matriz kernel computada en fit con los datos fuente y obejtivo.
    P: numpy.ndarray shape(d, m + n)
        Matriz de proyección de los datos.
    '''
    def __init__(self, num_features, n_neighbors=3, labda=0.1, beta=0.01,
                 gamma=1.0, max_iters=10, landmarks=False, epsilon=1e-4):
        self.num_features = num_features
        self.n_neighbors = n_neighbors
        self.labda = labda
        self.beta = beta
        self.gamma = gamma
        self.max_iters = max_iters
        self.landmarks = landmarks
        self.epsilon = epsilon

    def construct_X(self, source, target):
        '''
        Construye la matriz X, que agrupa los datos de los dominios fuente y
        objetivo.

        La matriz se define de forma distinta según se los dominios tienen el
        mismo número de features o no.

        Si el número de features es el mismo:
        X = source
            target
        Si no

        X = source   0
              0     target

        Argumentos
        -------------------------------
        source: numpy.ndarray shape(m, k)
            datos del dominio fuente
        target: numpy.ndarray shape(n, l)
            datos del dominio objetivo.

        Retorno
        -----------------------------------
        X: numpy.ndarray shape(m + n, k + l)
            La combinación de los datos de ambos dominios. Si `source` y
            `target` tienen el mismo número de features (k = l) la dimensión
            pasa a ser (m + n, k)
        '''
        if source.shape[1] == target.shape[1]:
            X = np.vstack((source, target))
        else:
            X = np.zeros(
                (
                    source.shape[0] + target.shape[0],
                    source.shape[1] + target.shape[1]
                )
            )
            X[:source.shape[0], :source.shape[1]] = source
            X[-target.shape[0]:, -target.shape[1]:] = target
        return X

    def construct_M(self, num_source, num_target):
        '''
        Construye la matriz M de coeficientes MMD.

        Argumentos
        --------------------------
        num_source: int.
            Número de instancias en el dominio fuente.
        num_target: int.
            Número de instancias en el dominio objetivo.

        Retorno
        -------------------
        M: numpy.ndarray shape (num_source+num_target, num_source+num_target).
            Matriz de coeficientes.
        '''
        M = matrix.construct_L(num_source, num_target)
        M = M / np.linalg.norm(M, ord='fro')
        return M

    def construct_I(self, num_source, num_target):
        '''
        Construye la matriz I de regularización.

        I es una matriz diagonal de dimension
        (instances_source + instances_target) tal que
        I[i, i] = 1 si i < instances_cource
                  0 si no

        Argumentos
        --------------------------
        num_source: int.
            Número de instancias en el dominio fuente.
        num_target: int.
            Número de instancias en el dominio objetivo.

        Retorno
        ----------------------------
        I: numpy.ndarray shape(num_source+num_target, num_source+num_target)
            Matriz diagonal de regularización.
        '''
        I_hat = np.ones(num_source + num_target)
        I_hat[:num_source] = 0
        I_hat = np.diag(I_hat)
        return I_hat

    def graph_laplacian(self, data, labels):
        '''
        Construye una matriz laplaciana para lso datos.

        Aegumentos
        -----------------------------
        data: numpy.ndarray shape(s, k)
            Datos sobre los que calcular la laplaciana.
        labels: numpy.ndarray shape(s, )
            etiquetas correspondientes a los datos.

        Retorno
        -------------------------------
        L: numpy.ndarray shape(s, s)
            matriz laplaciana.
        '''
        distances = sp.spatial.distance.cdist(data, data, metric='cosine')
        adjacency = graph.adjacency_matrix(
                distances, labels=labels, n_neighbors=self.n_neighbors
        )
        adjacency[np.diag_indices(adjacency.shape[0])] = 0
        return graph.laplacian(adjacency)

    def construct_graph_laplacian(self, source, target, source_labels, model,
                                  target_labeled=None, target_labels=None):
        '''
        Crea una matriz laplaciana inicial según la disponibilidad de datos.
        Si se tienen datos del dominio objetivo como referencia, se pueden
        predecir etiquetas para el dominio objetivo y así contruir una
        laplaciana completa. En caso contrario, la parte de la laplaciana
        correspondiente al dominio objetivo queda a cero al no tenr etiquetas.

        Argumentos
        -------------------------
        source: numpy.ndarray shape(m, k)
            Datos del dominio fuente
        target: numpy.ndarray shape(n, l)
            datos del dominio objetivo
        source_labels: numpy.ndarray shape(m, )
            Etiquetas del dominio fuente
        model: un clasificador de sklearn.
            Modelo con le realizar predicciones.
        target_labeled: numpy.ndarray shape (s, l)
            datos del dominio objetivo de referencia
        target_labels: numpy.ndarray shape(s, )
            etiquetas correspondientes a `target_labeled`

        Retorno
        -----------------------------
        L: numpy.ndarray shape(m + n, m + n)
            Matriz laplaciana.
        '''
        if target_labeled is not None:
            train_data = np.vstack((source, target_labeled))
            train_labels = np.concatenate((source_labels, target_labels))
            pred = self.model_predict(model, train_data, train_labels, target)
            total_data = np.vstack((train_data, target))
            total_labels = np.concatenate((train_labels, pred))
        else:
            total_data = source
            total_labels = source_labels
        laplacian = self.graph_laplacian(total_data, total_labels)
        L = np.zeros((source.shape[0] + target.shape[0], ) * 2)
        L[:laplacian.shape[0], :laplacian.shape[1]] = laplacian

        return L

    def construct_matrices(self, source, target, source_labels, model,
                           target_labeled=None, target_labels=None):
        '''
        Función de conveniencia para calcular varias matrices necesarias para
        el método

        Argumentos
        ---------------------------------
        source: numpy.ndarray shape(m, k)
            Datos del dominio fuente
        target: numpy.ndarray shape(n, l)
            datos del dominio objetivo
        source_labels: numpy.ndarray shape(m, )
            Etiquetas del dominio fuente
        model: un clasificador de sklearn.
            Modelo con le realizar predicciones.
        target_labeled: numpy.ndarray shape (s, l)
            datos del dominio objetivo de referencia
        target_labels: numpy.ndarray shape(s, )
            etiquetas correspondientes a `target_labeled`

        Retorno
        -----------------------------
        X: numpy.ndarray shape(m + n, k + l)
            La combinación de los datos de ambos dominios. Si `source` y
            `target` tienen el mismo número de features (k = l) la dimensión
            pasa a ser (m + n, k).
            Hay que notar que si  `target_labeled`no es `None` el orden
            de las instancias será: source, target_labeled, target. Este orden
            es muy imporatante porque es el que se asume en el resto de métodos
            de esta clase.
        M: numpy.ndarray shape (m + n, m + n).
            Matriz de coeficientes.
        H: numpy.ndarray shape(m + n, m + n)
            Matriz de centrado de los datos.
        I: numpy.ndarray shape(num_source+num_target, num_source+num_target)
            Matriz diagonal de regularización.
        K: numpy.ndarray shape(m + n, m + n)
            Matriz kernel de los datos. El kernel aplicado es lineal
        G: numpy.ndarray shape(m + n, m + n)
            Matriz de subgradiente.
        L: numpy.ndarray shape(s, s)
            matriz laplaciana.
        '''
        num_source = source.shape[0]
        num_target = target.shape[0]
        total_target = target
        if target_labeled is not None:
            num_target += target_labeled.shape[0]
            total_target = np.vstack((target_labeled, target))
        X = self.construct_X(source, total_target)
        M = self.construct_M(num_source, num_target)
        H = matrix.centering_matrix(num_source + num_target)
        I_hat = self.construct_I(num_source, num_target)
        K = kernel.linear(X)
        G = np.identity(K.shape[0])
        L = self.construct_graph_laplacian(
            source, target, source_labels, model, target_labeled, target_labels
        )

        return X, M, H, I_hat, K, G, L

    def compute_P(self, K, M, L, I, H, G):
        '''
        Calcula la matriz de proyección de los datos.

        Argumentos
        --------------------------------------
        K: numpy.ndarray shape(m + n, m + n)
            Matriz kernel de los datos.
        M: numpy.ndarray shape (m + n, m + n).
            Matriz de coeficientes.
        L: numpy.ndarray shape(s, s)
            matriz laplaciana.
        I: numpy.ndarray shape(num_source+num_target, num_source+num_target)
            Matriz diagonal de regularización.
        H: numpy.ndarray shape(m + n, m + n)
            Matriz de centrado de los datos.
        G: numpy.ndarray shape(m + n, m + n)
            Matriz de subgradiente.

        Retorno
        -------------------------------
        P: numpy.ndarray shape(m + n, self.num_features)
            Matriz de proyección de los datos.
        '''
        first = np.linalg.multi_dot(
                (K, M + self.labda * L - self.beta * I, K.T)
            ) + self.gamma * G
        try:
            first_term = np.linalg.inv(first)
        except np.linalg.LinAlgError:
            first_term = np.linalg.pinv(first)
        total = np.linalg.multi_dot((first_term, K, H, K.T))
        P = matrix.top_eigenvectors(total, self.num_features)

        return P

    def projection(self, P, K, num_train):
        '''
        Calcula las proyecciones de un conjunto de datos de entrenamiento y el
        resto de instancias.

        Si num_train es igual al número de instancias del dominio fuente este
        método devuelve las proyecciones de fuente y objetivo, pero se
        generaliza porque si se pasan instancias objetivo como referencia estas
        formarán parte del conjunto de entrenamiento.

        Argumentos
        -------------------------------
        P: numpy.ndarray shape(m + n, self.num_features)
            Matriz de proyección de los datos.
        K: numpy.ndarray shape(m + n, m + n)
            Matriz kernel de los datos.
        num_train: int.
            Número de instancias de entrenamiento.

        Retorno
        ------------------------------
        train_projection: numpy.ndarray shape(num_train, self.num_features)
            Proyección de las instancias de entrenamiento
        test_projection: numpy.ndarray shape (num_train, self.num_features)
            Proyección del resto de instancias.
        '''
        projection = P.T.dot(K).astype('float64')
        train_projection = projection[:, :num_train].T
        test_projection = projection[:, num_train:].T

        return train_projection, test_projection

    def compute_G(self, P, source_instances, source_dimension,
                  target_dimension):
        '''
        Calcula la matriz de subgradiente G.

        El cálculo varía en función de si los dominios fuente y obejtivo
        tienen el mismo número de features.

        G es una matriz diagonal que se actualiza del siguiente modo:
        G[i, i] = 1 / || P [i] || ** 2

        Si el número de features en ambos dominios es el mismo entonces:
        G[i, i] = 1 / || P [i] || ** 2 si i < source_instances

        Es decir, no se actualizan las entradas correspondientes al dominio
        objetivo.

        Argumentos
        -------------------------------
        P: numpy.ndarray shape(m + n, self.num_features)
            Matriz de proyección de los datos.
        source_instances: int
            Número de instancias en el dominio objetivo
        source_dimension: int.
            Número de features del dominio fuente
        target_dimension: int.
            Número de instancias del dominio objetivo.

        Retorno
        ---------------------------------------------
        G: numpy.ndarray shape(m + n, m + n)
            Matriz de subgradiente
        '''
        if source_dimension == target_dimension:
            diag = np.ones(source_instances)
            indices = (P[:source_instances] != 0).any(axis=1)
            diag[indices] /= (np.linalg.norm(
                    P[:source_instances][indices], axis=1
            ) ** 2)
            tot = np.concatenate(
                    (diag, np.ones(P.shape[0] - source_instances))
            )
            return np.diag(tot)
        diagonal = np.zeros((P.shape[0], ))
        indices = (P != 0).any(axis=1)
        diagonal[indices] = 1 / (np.linalg.norm(P[indices], axis=1) ** 2)
        return np.diag(diagonal)

    def model_predict(self, model, train_data, train_labels, predict_data):
        '''
        Entrena un modelo y predice las etiquetas de un conjunto de datos.

        Argumentos
        --------------------------
        model: un clasificador de sklearn
            El modelo para predecir.
        train_data: numpy.ndarray shape(s, r)
            Los datos de entrenamiento
        train_labels: numpy.ndarray shape(s, )
            Las etiquetas correspondientes a los datos de entrenamiento
        predict_data: numpy.ndarray shape (q, r)
            Datos cuyas etiquetas se van a predecir.

        Retorno
        ----------------------------
        predicted_labels: numpy.ndarray shape(q, )
            Etiquetas predichas para `predict_data`
        '''
        return model.fit(train_data, train_labels).predict(predict_data)

    def update_total_labels(self, K, P, source_labels, model,
                            target_labels=None):
        '''
        Devuelve las etiquetas asignadas a todas las instancias. Ya se dispone
        de las etiquetas del dominio fuente, así que únicamente se predicen las
        del dominio objetivo.

        Argumentos
        ------------------------
        K: numpy.ndarray shape(m + n, m + n)
            Matriz kernel de los datos.
        P: numpy.ndarray shape(m + n, self.num_features)
            Matriz de proyección de los datos.
        source_labels: numpy.ndarray shape(m, )
            Etiquetas del dominio fuente.
        model: un clasificador de sklearn
            El modelo para predecir.
        target_labels: numpy.ndarray shape (q, ). opcional (default=None)
            Etiquetas de referencia del dominio objetivo. Si no es `None`
            estas etiquetas junto a sus datos se usan para el entrenamiento
            del modelo.

        Retorno
        -----------------------
        total_labels: numpy.ndarray shape(m + n, )
            Etiquetas para todas las instancias. Primero van las etiquetas del
            dominio fuente y luego las del objetivo. Únicamente se predicen las
            etiquetas del dominio objetivo de las que no se dispone, el resto
            vienen fijadas.
        '''
        train_data_lenght = source_labels.size
        train_labels = source_labels
        if target_labels is not None:
            train_data_lenght += target_labels.size
            train_labels = np.concatenate((source_labels, target_labels))
        train_projection, predict_projection = self.projection(
                P, K, train_data_lenght
        )
        predicted_target_labels = self.model_predict(
                model, train_projection, train_labels, predict_projection
        )
        return np.concatenate((train_labels, predicted_target_labels))

    def update_weights(self, weights, P, K, labels, train_data_length):
        '''
        Actualiza los pesos asignados a las instancias para su uso en el
        entrenamiento de modelos.

        Los pesos se actualizan sumando a cada instancia de entrenamiento el
        número de instancias del dominio obejtivo para las cuales está entre
        sus vecinos más cercanos (según self.n_neighbors).

        Argumentos
        -------------------------------
        P: numpy.ndarray shape(m + n, self.num_features)
            Matriz de proyección de los datos.
        K: numpy.ndarray shape(m + n, m + n)
            Matriz kernel de los datos.
        labels: numpy.ndarray shape(m + n, )
            Etiquetas para todas las instancias. Primero van las etiquetas del
            dominio fuente y luego las del objetivo.
        train_data_lenght: int.
            Número de instancias de reservadas para el entrenamiento. Esto
            incluye tanto número de instancias del dominio fuente como
            instancias de ldominio objetivo que se tengan como referencia.

        Retorno
        ---------------------------------------
        weights: numpy.ndarray shape(train_data_length)
            Pesos de las instancias de entrenamiento.
        '''
        train_projection, predict_projection = self.projection(
                P, K, train_data_length
        )
        distances = sp.spatial.distance.cdist(
                predict_projection, train_projection, metric='cosine'
        )
        nearest_neigbors_indices = graph.nearest_same_class_neighbors(
            distances, labels[train_data_length:], labels[:train_data_length],
            n_neighbors=self.n_neighbors
        )
        weights += nearest_neigbors_indices.sum(axis=0)
        return weights

    def compute_optimization_value(self, P, K, M, H, I, L):
        '''
        Calcula el valor actual obtenido en la función a optimizar.

        Argumentos
        --------------------------------
        P: numpy.ndarray shape(m + n, self.num_features)
            Matriz de proyección de los datos.
        K: numpy.ndarray shape(m + n, m + n)
            Matriz kernel de los datos.
        M: numpy.ndarray shape (m + n, m + n).
            Matriz de coeficientes.
        H: numpy.ndarray shape(m + n, m + n)
            Matriz de centrado de los datos.
        I: numpy.ndarray shape(num_source+num_target, num_source+num_target)
            Matriz diagonal de regularización.
        L: numpy.ndarray shape(s, s)
            matriz laplaciana.

        Retorno
        ------------------------------------
        value: float
            Valor actual del problema de optimización.
        '''
        objectives_matrix = M + self.labda * L - self.beta * I
        first_term = np.linalg.multi_dot(
                (P.T, K, objectives_matrix, K.T, P)
        )

        trace = np.einsum('ii', first_term)
        regularization = matrix.l21_norm(P)

        return trace + regularization

    def solve(self, source, target, source_labels, model, target_labeled=None,
              target_labels=None):
        '''
        Resuelve iterativamente el problema de optimización y calcula las
        etiquetas asignadas al `target`.

        Argumentos
        -------------------------
        source: numpy.ndarray shape(m, k)
            Datos del dominio fuente
        target: numpy.ndarray shape(n, l)
            datos del dominio objetivo
        source_labels: numpy.ndarray shape(m, )
            Etiquetas del dominio fuente
        model: un clasificador de sklearn.
            Modelo con le realizar predicciones.
        target_labeled: numpy.ndarray shape (s, l)
            datos del dominio objetivo de referencia
        target_labels: numpy.ndarray shape(s, )
            etiquetas correspondientes a `target_labeled`

        Retorno
        ---------------------------
        predicted: numpy.ndarray shape(n, )
            Etiquetas predichas para `target`.
        weights: numpy.ndarray shape(m + s, )
           Pesos de las instancias de entrenamiento para entrenar modelos. Si
           self.landmarks es `False` entonces weights es un array con todos los
           elementos a 1 (el mismo peso para todas alss instancias)
        '''
        X, M, H, I, K, G, L = self.construct_matrices(
            source, target, source_labels, model, target_labeled, target_labels
        )
        train_data_lenght = source.shape[0]
        if target_labeled is not None:
            train_data_lenght += target_labeled.shape[0]
        weights = np.zeros(source.shape[0])
        last = np.inf
        current = np.inf
        for i in range(self.max_iters):
            P = self.compute_P(K, M, L, I, H, G)
            G = self.compute_G(
                    P, source.shape[0], source.shape[1], target.shape[1]
            )
            labels = self.update_total_labels(
                    K, P, source_labels, model, target_labels
            )
            L = self.graph_laplacian(X, labels)
            if self.landmarks:
                weights = self.update_weights(
                        weights, P, K, labels, train_data_lenght
                )
            last = current
            current = self.compute_optimization_value(P, K, M, H, I, L)
            # print('current', current)
            if current < 0:
                break
            # print('current', current)
            if last > current and (last - current) < self.epsilon:
                break
        self.P = P
        self.K = K
        predicted = labels[train_data_lenght:]
        if self.landmarks:
            return predicted, weights
        weights += 1
        return predicted, weights

    def fit_predict(self, source, target, source_labels, model,
                    target_labeled=None, target_labels=None):
        '''
        Ajusta el modelo y devuelve las predicciones para `target`.

        Argumentos
        -------------------------
        source: numpy.ndarray shape(m, k)
            Datos del dominio fuente
        target: numpy.ndarray shape(n, l)
            datos del dominio objetivo
        source_labels: numpy.ndarray shape(m, )
            Etiquetas del dominio fuente
        model: un clasificador de sklearn.
            Modelo con le realizar predicciones.
        target_labeled: numpy.ndarray shape (s, l)
            datos del dominio objetivo de referencia
        target_labels: numpy.ndarray shape(s, )
            etiquetas correspondientes a `target_labeled`

        Retorno
        ---------------------------
        predicted: numpy.ndarray shape(n, )
            Etiquetas predichas para `target`.
        weights: numpy.ndarray shape(m + s, )
           Pesos de las instancias de entrenamiento para entrenar modelos. Si
           self.landmarks es `False` entonces weights es un array con todos los
           elementos a 1 (el mismo peso para todas alss instancias)
        '''
        return self.solve(
            source, target, source_labels, model, target_labeled,
            target_labels
        )

    def fit_transform(self, source, target, source_labels, model,
                      target_labeled=None, target_labels=None):
        '''
        Ajusta el modelo y las proyecciones de datos de entrenamiento y datos
        de test.

        Los datos de entrenamiento son `source` y `target_labeled`(si no es
        None). Los datos de test son `target`.

        Argumentos
        -------------------------
        source: numpy.ndarray shape(m, k)
            Datos del dominio fuente
        target: numpy.ndarray shape(n, l)
            datos del dominio objetivo
        source_labels: numpy.ndarray shape(m, )
            Etiquetas del dominio fuente
        model: un clasificador de sklearn.
            Modelo con le realizar predicciones.
        target_labeled: numpy.ndarray shape (s, l)
            datos del dominio objetivo de referencia
        target_labels: numpy.ndarray shape(s, )
            etiquetas correspondientes a `target_labeled`

        Retorno
        ---------------------------
        train_projection: numpy.ndarray shape(m+s, self.num_features)
            Proyección de las instancias de entrenamiento
        test_projection: numpy.ndarray shape (n, self.num_features)
            Proyección del resto de instancias.
        weights: numpy.ndarray shape(m + s, )
           Pesos de las instancias de entrenamiento para entrenar modelos. Si
           self.landmarks es `False` entonces weights es un array con todos los
           elementos a 1 (el mismo peso para todas alss instancias)
        '''
        _, weights = self.fit_predict(
                source, target, source_labels, model, target_labeled,
                target_labels
        )
        train_projection, predict_projection = self.projection(
                self.P, self.K, source.shape[0]
        )
        return train_projection, predict_projection, weights
