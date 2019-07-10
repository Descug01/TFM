# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:35:33 2019

@author: David
"""
from .util import matrix
from .util import graph
from .util import data
import numpy as np
import functools
from .util.kernel import KERNELS
from .util import kernel


class BaseTCA():
    '''
    Construye el transformador.

    Argumentos
    ----------------------------------
    num_features: int.
        Número de features con el que se quedan los datos transformados.
    kernel: string, debe estar definido en kernel.KERNELS.
        Indica el tipo de kernel que se va a usar.
    mu: float.
        Parámetro de tradeoff entre la mminimización de divergencia y
        conservación de covarianza. Un valor mayor que 1 da más importancia
        a la covarianza.
    **kwargs: dict.
        Parámetros adicionales para las funciones kernel.

    Atributos
    ----------------------------------------
    fit_data: numpy.ndarray shape(m +n, k)
        Datos de dominio fuente y objetivo pasados a la función fit. se guardan
        para poder realizar la generalización del kernel.
    K: numpy.ndarray shape(m +n, m + n)
        Matriz kernel computada en fit con los datos fuente y obejtivo.
    W: numpy.ndarray shape(d, m + n)
        Matriz de proyección de los datos.
    '''
    def __init__(self, num_features, kernel='linear', mu=1.0,
                 **kwargs):
        self.num_features = num_features
        self.kernel = functools.partial(KERNELS[kernel], **kwargs)
        self.mu = mu

    def generalize_kernel(self, data):
        '''
        Calcula valores de la función kernel para nuevas muestras. Esto
        es necesario porque los datos transformados se obtienen a partir
        de la matriz kernel.

        Argumentos
        -----------------------------
        data: numpy.ndarray shape(i, j)
            Las datos para los que se quiere calcular la función kernel.

        Retorno
        ----------------------------
        K: numpy.ndarray(data.shape[0], self.K.shape[0])
            La matriz kernel obtenida para los datos.
        '''
        return self.kernel(data, self.fit_data)

    def construct_matrices(self, source, target):
        '''
        Función de conveniencia para obtener varias de las matrices
        requeridas en los cálculos.

        Argumentos
        --------------------------------
        source: numpy.ndarray shape(n, k).
            Datos del dominio fuente (sin etiquetas)
        target: numpy.ndarray shape(m, k).
            Datos del dominio objetivo.

        Retorno
        --------------------------------
        K: numpy.ndarray shape(n + m, n + m).
            Matriz kernel
        L: numpy.ndarray shape(n + m, n + m).
            Matriz de coeficientes MMD
        H: numpy.ndarray shape(n + m, n + m).
            Matriz de centrado
        '''
        num_source = source.shape[0]
        num_target = target.shape[0]
        L, H = (
            matrix.construct_L(num_source, num_target),
            matrix.centering_matrix(num_source + num_target)
        )
        union = np.vstack((source, target))
        K = self.kernel(union)
        self.K = K
        self.fit_data = union
        return K, L, H

    def fit_transform(self, source, target, *args, **kwargs):
        '''
        Ajusta los parámetros para la transformación y devuelve los datos
        fuente y destino ya transformados.

        Argumentos
        ---------------------------------
        source: numpy.ndarray shape(n, k).
             Datos del dominio fuente
        target: numpy.ndarray shape(m, k).
             Datos del dominio objetivo.

        Retorno
        ----------------------------------
        source_transformed: numpy.ndarray shape(n, self.num_features).
            Datos fuente transformados
        target_transformed: numpy.ndarray shape(m, self.num_features).
            Datos objetivo transformados
        '''
        self.fit(source, target, *args, **kwargs)
        transformed = self.K.dot(self.W)
        return transformed[:source.shape[0]], transformed[-target.shape[0]:]

    def transform(self, data):
        '''
        Devuelve los datos ya transformados.

        Argumentos
        ------------------------------
        source: numpy.ndarray shape(n, k).
            Datos del dominio fuente
        target: numpy.ndarray shape(m, k).
            Datos del dominio objetivo.

        Retorno
        ----------------------------------
        source_transformed: numpy.ndarray shape(n, self.num_features).
            Datos fuente transformados
        target_transformed: numpy.ndarray shape(m, self.num_features).
            Datos objetivo transformados
        '''
        kernel_evaluation = self.generalize_kernel(data)
        transformed = kernel_evaluation.dot(self.W)
        return transformed


class TCA(BaseTCA):
    '''
    Transfer Component Analysis.
    Adaptación de dominio homogénea.
    Todas las funciones asumen que los datos de entrada tienen las
    instancias en filas y features en columnas.

    Atributos
    num_features: int.
        Número de features con el que se queda la transformación.
    kernel: string, debe estar definido en kernel.KERNELS.
        Indica el tipo de kernel que se va a usar.
    mu: float.
        Parámetro de tradeoff entre la minimización de divergencia
        y conservación de covarianza. Un valor mayor que 1 da más
        importancia a la covarianza.
    K: numpy.ndarray shape(num_instances, num_instances).
        Gram matrix obtenida aplicando kernel a la combinaciñon de datos del
        dominio fuente y objetivo.
    W: numpy.ndarray(num_instances, num_features).
        Matriz de transformación.
    fit_data: numpy.ndarray(num_instances, original_num_features).
        Contiene los datos pasados como argumento al método fit.
    '''

    def fit(self, source, target):
        '''
        Ajusta los parámetros para la transformación.

        Argumentos
        -----------------------------------------
        source: numpy.ndarray shape(n, k).
            Datos del dominio fuente
        target: numpy.ndarray shape(m, k).
            Datos del dominio objetivo.

        Retorno
        ------------------------------
        self: TCA
            La propia instancia
        '''
        K, L, H = self.construct_matrices(source, target)
        first = (
                np.linalg.multi_dot((K, L, K))
                + self.mu * np.identity(K.shape[0])
        )
        try:
            first_term = np.linalg.inv(first)
        except np.linalg.LinAlgError:
            first_term = np.linalg.pinv(first)

        second_term = np.linalg.multi_dot((K, H, K))
        self.W = matrix.top_eigenvectors(
                first_term.dot(second_term),
                number=self.num_features
        )
        return self


class SSTCA(BaseTCA):
    '''
    Semi Supervised Transfer Component Analysis.
    Adaptación de dominio homogénea. Similar a TCA, pero aqui se tienen en
    cuenta etiquetas del dominio fuente.
    Todas las funciones asumen que los datos de entrada tienen las
    instancias en filas y features en columnas.

    Atributos
    num_features: int.
        Número de features con el que se queda la transformación.
    kernel: string, debe estar definido en kernel.KERNELS.
        Indica el tipo de kernel que se va a usar.
    mu: float.
        Parámetro de tradeoff entre la mminimización de divergencia
        y conservación de covarianza. Un valor mayor que 1 da más
        importancia a la covarianza.
    K: numpy.ndarray shape(num_instances, num_instances).
        Gram matrix obtenida aplicando kernel a la combinaciñon de datos del
        dominio fuente y objetivo.
    W: numpy.ndarray(num_instances, num_features).
        Matriz de transformación.
    fit_data: numpy.ndarray(num_instances, original_num_features).
        Contiene los datos pasados como argumento al método fit.
    '''
    def __init__(self, num_features, kernel='linear', n_neighbors=5,
                 mu=1.0, gamma=1.0, labda=1.0, sigma_squared=1.0, **kwargs):
        '''
        Crea el objeto.

        Argumentos
        -----------------------------------
        num_features: int.
            Número de features con el que se queda la transformación.
        kernel: string, debe estar definido en kernel.KERNELS.
            Indica el tipo de kernel que se va a usar.
        k: int >= 1.
            Número de vecinos a tener en cuenta en el cálculo de la matriz
            laplaciana.
        mu: float >=0.
            Parámetro de tradeoff entre la mminimización de divergencia y
            conservación de covarianza. Un valor mayor que 1 da más importancia
            a la covarianza.
        gamma: float >= 0.
            Parámetro de tradeoff entre la preservación de la
            correspondencia con las etiquetas y la maximización de la
            varianza de fuente y objetivo. Valores más grandes priorizan la
            correspondencia con las etiquetas.
        labda: float >= 0.
            Parámetro de tradeoff para controlar la
            importancia que se le da a la preservación de la localidad de
            la proyección de los datos. Valores más grandes le dan más
            importancia.
        sigma: float >=0.
            Parámetro que controla la distancia calculada en
            la estimación del la matriz laplaciana. Se eleva al cuadrado
            en las operaciones.
        '''
        BaseTCA.__init__(self, num_features, kernel, mu, **kwargs)
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.labda = labda
        self.sigma_squared = sigma_squared

    def construct_kernel_label(self, labels, dimension):
        '''
        Matriz kernel para las etiquetas. La función kernel aplicada el el
        kernel lineal.

        Argumentos
        ------------------------------
        labels: numpy.ndarray shape(num_instances_source, ).
            Etiquetas de los datos del dominio fuente.

        Retorno
        ---------------------------------
        kernel: numpy.ndarray shape(self.K.shape).
            Matriz kernel de las etiquetas.
        '''
        binarized_labels = data.binarize_labels(labels)
        identity = np.identity(dimension)
        kernel_matrix = np.zeros((dimension, ) * 2)
        kernel_labels = self.kernel(binarized_labels)
        num_labels = binarized_labels.shape[0]
        kernel_matrix[:num_labels, :num_labels] = kernel_labels

        return self.gamma * kernel_matrix + (1 - self.gamma) * identity

    def construct_graph_laplacian(self, source, target):
        '''
        Calcula la matriz laplaciana correspondiente a los dominios.

        Argumentos
        -------------------------------------
        source: numpy.ndarray shape(n, k).
            Datos del dominio fuente
        target: numpy.ndarray shape(m, k).
            Datos del dominio objetivo.

        Retorno
        -------------------------------
        laplacian: numpy.ndarray shape (n + m, n + m).
            Matriz laplaciana.
        '''
        affinity_matrix = self.compute_affinity_matrix(source, target)
        return graph.laplacian(affinity_matrix)

    def compute_affinity_matrix(self, source, target):
        '''
        Calcula la matriz de afinidad de los datos.

        Esta matriz de afinidad se define de forma un poco rara:
            M_ij = exp(- d_ij ^ 2 / (2 * (sigma ** 2))) si i es vecino de j
        d_ij es la distancia euclidiana, y precisamente es la distancia
        euclidiana la que sirve para determinar la vecindad de los puntos, en
        lugar de usar la propia métrica definida.

        Argumentos
        ----------------------------------
        source: numpy.ndarray shape(n, k).
            Datos del dominio fuente
        target: numpy.ndarray shape(m, k).
            Datos del dominio objetivo.

        Retorno
        -----------------------------------
        affinity: numpy.ndarray shape (n + m, n + m).
            Matriz de afinidad.
        '''
        distances = kernel.rbf(np.concatenate((source, target)))
        distances -= np.identity(distances.shape[0])
        return graph.adjacency_matrix(
                distances, n_neighbors=self.n_neighbors, reverse=True
        )

    def fit(self, source, target, source_labels):
        '''
        Ajusta los parámetros para la transformación.

        Argumentos
        ------------------------------
        source: numpy.ndarray shape(n, k).
            Datos del dominio fuente
        target: numpy.ndarray shape(m, k).
            Datos del dominio objetivo.
        source_labels: numpy.ndarray.shape(n,).
            Etiquetas del dominio fuente.

        Retorno
        ---------------------------------
        self: SSTCA.
            La propia instancia
        '''
        K, L, H = self.construct_matrices(source, target)
        K_y = self.construct_kernel_label(source_labels, K.shape[0])
        laplacian = self.construct_graph_laplacian(source, target)
        first = (
            np.linalg.multi_dot((K, L + self.labda * laplacian, K))
            + self.mu * np.identity(K.shape[0])
        )
        try:
            first_term = np.linalg.inv(first)
        except np.linalg.LinAlgError:
            first_term = np.linalg.pinv(first)

        second_term = np.linalg.multi_dot((K, H, K_y, H, K))

        self.W = matrix.top_eigenvectors(
                first_term.dot(second_term),
                number=self.num_features,
        )
        return self
