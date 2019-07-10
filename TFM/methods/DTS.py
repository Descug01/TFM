# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 17:29:15 2019

@author: David
"""
import numpy as np
from .util import data


class DTS():
    '''
    Implementa el transformador

    Construye la instancia.

    Argumentos
    ------------------------------
    alpha: float. opcional (default=0.1)
        Constante que se usa en la optimización
    beta: float. opcional (default=0.025)
        Constante que se usa en la optimización.
    labda: float. opcional
        Constante que se usa en la optimización.
    max_iter: int.
        Máximo número de iteraciones de la optimización.
    epsilon: float.
        Umbral de convergencia. Cuando una serie de maagnitudes son menores
        detiene la optimización.

    Atributos
    -------------------------------------
    P: numpy.ndarray shape(m, k)
        Matriz de proyección de los datos.
    '''
    def __init__(self, alpha=0.1, beta=0.025, labda=2.67, max_iter=100,
                 epsilon=1e-4):
        self.alpha = alpha
        self.beta = beta
        self.labda = labda
        self.max_iter = max_iter
        self.epsilon = epsilon

    def compute_B(self, Y):
        '''
        Calcula la matriz B que ayuda a la proyección.

        B se define como
                  1 if Y_ij = 1
        b_ij =    -1 si no

        Argumentos
        ---------------------------------------
        Y: numpy.ndarray.shape(d, n).
            Etiquetas del dominio fuente binarizadas.

        Retorno
        ---------------------------------------
        B: numpy.ndarray shape(d, n).
            La matriz B
        '''
        B = Y.copy()
        B[B == 0] = -1
        return B

    def compute_G1(self, Y, B, M):
        '''
        Calcula la matriz G1 del problema de optimización para actualizar
        P.

        Se obtiene como G1 = Y + B * M (* es el producto de Hadamard)

        Argumentos
        --------------------------------
        Y: numpy.ndarray.shape(d, n).
            Etiquetas del dominio fuente binarizadas.
        B: numpy.ndarray shape(d, n).
            Matriz de etiquetas modificadas.
        M: numpy.ndarray shape(d, n).
            Matriz de holgura para la proyección.

        Retorno
        ---------------------------------
        G1: numpy.ndarray shape(d, n)
            La matriz G1 buscada.
        '''
        return Y + B * M

    def compute_G2(self, X_s, X_t, Z):
        '''
        Calcula la matriz G2 del problema de optimización para actualizar
        P.

        Se obtiene como G2 = X_t - X_sZ

        Argumentos
        --------------------------------
        X_s: numpy.ndarray shape(k, n).
            Datos del dominio fuente. Cada columna es una instancia.
        X_t: numpy.ndarray shape(k, m).
            Datos del dominio objetivo. Cada columna es una instancia.
        Z: numpy.ndarray shape(n, m).
            Matriz de reconstrucción de los datos.

        Retorno
        ---------------------------------
        G2: numpy.ndarray shape(d (número de features originales), m)
            La matriz G2 buscada.
        '''
        return X_t - X_s.dot(Z)

    def compute_G3(self, E, Y_1, mu):
        '''
        Calcula la matriz G3 del problema de optimización para actualizar P.

        Se obtiene como:
            G3 = E - Y_1 / mu

        Argumentos
        ------------------------------
        E: numpy.ndarray shape(d, m).
            Matriz de ruido de los datos.
        Y_1: numpy.ndarray shape(d, m).
            Matriz multiplicador usada para resolver el problema de
            optimización.
        mu: float.
            Constante usada para la división.

        Retorno
        --------------------------------
        G3: numpy.ndarray shape(d, m)
            La matriz buscada.
        '''
        return E - Y_1 / mu

    def solve_P(self, X_s, X_t, Z, E, Y, B, M, Y_1, mu):
        '''
        Calcula la matriz de transformación, P.

        Argumentos
        -------------------------------------
        X_s: numpy.ndarray shape(k, n).
            Datos del dominio fuente. Cada columna es una instancia.
        X_t: numpy.ndarray shape(k, m).
            Datos del dominio objetivo. Cada columna es una instancia.
        Z: numpy.ndarray shape(n, m).
            Matriz de reconstrucción de los datos.
        E: numpy.ndarray shape(d, m).
            Matriz de ruido de los datos.
        Y: numpy.ndarray.shape(d, n).
            Etiquetas del dominio fuente binarizadas. Cada columna es una
            etiqueta.
        B: numpy.ndarray shape(d, n).
            Matriz de etiquetas modificadas.
        M: numpy.ndarray shape(d, n).
            Matriz de holgura para la proyección.
        Y_1: numpy.ndarray shape(d, m).
            Matriz multiplicador usada para resolver el problema de
            optimización.
        mu: float.
            Constante usada para el cálculo de G3 y de la propia P.

        Retorno
        ---------------------------------------
        P: numpy.ndarray shape(k, d).
            Matriz de transformación de los datos.
            Con k la dimensionalidad de los datos originales y d la
            dimensionalidad del espacio de proyección.
        '''
        G_1 = self.compute_G1(Y, B, M)
        G_2 = self.compute_G2(X_s, X_t, Z)
        G_3 = self.compute_G3(E, Y_1, mu)
        Id = np.identity(G_2.shape[0])

        first = 2 * X_s.dot(X_s.T) + mu * G_2.dot(G_2.T) + 2 * self.labda * Id
        try:
            first_term = np.linalg.inv(first)
        except np.linalg.LinAlgError:
            first_term = np.linalg.pinv(first)
        second_term = 2 * X_s.dot(G_1.T) + mu * G_2.dot(G_3.T)
        return first_term.dot(second_term)

    def compute_G4(self, X_t, P, E, Y_1, mu):
        '''
        Calcula G4, una matriz usada en los cálculos para actualizar Z.

        Argumentos
        -----------------------------
        X_t: numpy.ndarray shape(k, m).
            Datos del dominio objetivo. Cada columna es una instancia.
        P: numpy.ndarray shape(k, d).
            Matriz de transformación de los datos.
        E: numpy.ndarray shape(d, m).
            Matriz de ruido de los datos.
        Y_1: numpy.ndarray shape(d, m).
            Matriz multiplicador usada para resolver el problema de
            optimización.
        mu: float.
            Constante usada para el cálculo.

        Retorno
        -----------------------------------
        G_4: numpy.ndarray shape(d, m).
            Matriz necesaria para cálculos.
        '''
        return P.T.dot(X_t) - E + Y_1 / mu

    def compute_G5(self, Z_1, Y_2, mu):
        '''
        Calcula G5, una matriz usada en los cálculos para actualizar Z.

        Argumentos
        ------------------------------
        Z_1: numpy.ndarray shape(n, m).
            Matriz usada para resolver el problema de optimización.
        Y_2: numpy.ndarray shape(d, m).
            Matriz multiplicador usada para resolver el problema de
            optimización.
        mu: float.
            Constante usada para el cálculo.

        Retorno
        ------------------------------
        G_5: numpy.ndarray shape(d, m).
            Matriz necesaria para cálculos.
        '''
        return self.compute_G3(Z_1, Y_2, mu)

    def compute_G6(self, Z_2, Y_3, mu):
        '''
        Calcula G5, una matriz usada en los cálculos para actualizar Z.

        Argumentos
        ---------------------------
        Z_2: numpy.ndarray shape(n, m).
            Matriz usada para resolver el problema de optimización.
        Y_3: numpy.ndarray shape(d, m).
            Matriz multiplicador usada para resolver el problema de
            optimización.
        mu: float.
            Constante usada para el cálculo.

        Retorno
        ------------------------------
        G_5: numpy.ndarray shape(d, m).
            Matriz necesaria para cálculos.
        '''
        return self.compute_G3(Z_2, Y_3, mu)

    def solve_Z(self, X_s, X_t, P, E, Y_1, Y_2, Y_3, Z_1, Z_2, mu):
        '''
        Calcula la matriz de reconstrucción, Z.

        Argumentos
        ---------------------------------------
        X_s: numpy.ndarray shape(k, n).
            Datos del dominio fuente. Cada columna es una instancia.
        X_t: numpy.ndarray shape(k, m).
            Datos del dominio objetivo. Cada columna es una instancia.
        P: numpy.ndarray shape(k, d).
            Matriz de transformación de los datos.
        E: numpy.ndarray shape(d, m).
            Matriz de ruido de los datos.
        Y_1: numpy.ndarray shape(d, m).
            Matriz multiplicador usada para resolver el problema de
            optimización.
        Y_2: numpy.ndarray shape(d, m).
            Matriz multiplicador usada para resolver el problema de
            optimización.
        Y_3: numpy.ndarray shape(d, m).
            Matriz multiplicador usada para resolver el problema de
            optimización.
        Z_1: numpy.ndarray shape(n, m).
            Matriz usada para resolver el problema de optimización.
        Z_2: numpy.ndarray shape(n, m).
            Matriz usada para resolver el problema de optimización.
        mu: float.
            Constante usada para el cálculo.

        Retorno
        --------------------------------
        Z: numpy.ndarray shape(n, m).
            Matriz de reconstrucción de los datos.
        '''
        G_4 = self.compute_G4(X_t, P, E, Y_1, mu)
        G_5 = self.compute_G5(Z_1, Y_2, mu)
        G_6 = self.compute_G6(Z_2, Y_3, mu)
        Id = np.identity(X_s.shape[1])

        first = np.linalg.multi_dot((X_s.T, P, P.T, X_s)) + 2 * Id
        try:
            first_term = np.linalg.inv(first)
        except np.linalg.LinAlgError:
            first_term = np.linalg.pinv(first)
        second_term = G_5 + G_6 + np.linalg.multi_dot((X_s.T, P, G_4))

        return first_term.dot(second_term)

    def compute_theta(self, matrix, threshold):
        '''
        Calcula la función theta aplicada para actualizar Z_1

        Argumentos
        -------------------------------
        matrix: numpy.ndarray shape(a,b)
            La matriz a la que se va a aplicar la función.
        threshold: float.
            Valor que sirve como umbral para la operación.

        Retorno
        ------------------------------------
        theta: numpy.ndarray (a, b)
            El resultado de la operación.
        '''
        U, sigma, V_transposed = np.linalg.svd(matrix, full_matrices=False)
        sigma = self.shrink(sigma, threshold)

        return np.linalg.multi_dot((U, np.diag(sigma), V_transposed))

    def solve_Z1(self, Z, Y_2, mu):
        '''
        Actualiza Z1, una matriz necesaria para el problema de optimización.

        Argumentos
        -------------------------------
        X_s: numpy.ndarray shape(k, n).
            Datos del dominio fuente. Cada columna es una instancia.
        X_t: numpy.ndarray shape(k, m).
            Datos del dominio objetivo. Cada columna es una instancia.
        Z: numpy.ndarray shape(n, m).
            Matriz de reconstrucción de los datos.
        Y_2: numpy.ndarray shape(d, m).
            Matriz multiplicador usada para resolver el problema de
            optimización.
        mu: float.
            Constante usada para algunos cálculos.

        Retorno
        -----------------------------------
        Z_1: numpy.ndarray shape(n, m).
            La matriz buscada.
        '''
        return self.compute_theta(Z + Y_2 / mu, 1 / mu)

    def shrink(self, matrix, value):
        '''
        Calcula la función shrink usada en varias operaciones.

        Esta función se define como:
            shrink(x, a) = sign(x) * max(|x| - a, 0)

        Argumentos
        --------------------------------------
        matrix: numpy.ndarray shape (a, b).
            Matriz sobre la que se va a operar.
        value: float.
            Valor `a` que sirve para realizar la operación.

        Retorno
        ---------------------------------------
        numpy.ndarray shape (a, b).
            Resultado de aplicar shrink a matrix.
        '''
        return np.sign(matrix) * np.clip(np.abs(matrix) - value, 0, np.inf)

    def solve_Z2(self, Z, Y_3, mu):
        '''
        Actualiza Z2, una matriz necesaria para el problema de optimización.

        Argumentos
        ------------------------
        Z: numpy.ndarray shape(n, m).
            Matriz de reconstrucción de los datos.
        Y_3: numpy.ndarray shape(d, m).
            Matriz multiplicador usada para resolver el problema de
            optimización.
        mu: float.
            Constante usada para los cálculos.

        Retorno
        ----------------------------
        Z_2: numpy.ndarray shape(n, m).
            La matriz buscada.
        '''
        return self.shrink(Z + Y_3 / mu, self.alpha / mu)

    def solve_E(self, X_s, X_t, P, Z, Y_1, mu):
        '''
        Actualiza E, la matriz de ruido.

        Argumentos
        -----------------------------
        X_s: numpy.ndarray shape(k, n).
            Datos del dominio fuente. Cada columna es una instancia.
        X_t: numpy.ndarray shape(k, m).
            Datos del dominio objetivo. Cada columna es una instancia.
        P: numpy.ndarray shape(k, d).
            Matriz de transformación de los datos.
        Z: numpy.ndarray shape(n, m).
            Matriz de reconstrucción de los datos.
        Y_1: numpy.ndarray shape(d, m).
            Matriz multiplicador usada para resolver el problema de
            optimización.
        mu: float.
            Constante usada para los cálculos.

        Retorno
        -----------------------------
        E: numpy.ndarray shape(d, m).
            Matriz de ruido de los datos.
        '''
        return self.shrink(
                P.T.dot(X_t) - np.linalg.multi_dot((P.T, X_s, Z)) + Y_1 / mu,
                self.beta / mu
        )

    def solve_M(self, X_s, Y, P, B):
        '''
        Calcula la matriz M de holgura de la proyección.

        Argumentos
        -----------------------------
        X_s: numpy.ndarray shape(k, n).
            Datos del dominio fuente. Cada columna es una instancia.
        Y: numpy.ndarray.shape(d, n).
            Etiquetas del dominio fuente binarizadas.  Cada columna es una
            etiqueta.
        P: numpy.ndarray shape(k, d).
            Matriz de transformación de los datos.
        B: numpy.ndarray shape(d, n).
            Matriz de etiquetas modificadas.

        Retorno
        --------------------------------
        M: numpy.ndarray shape(d, n).
            Matriz de holgura para la proyección.
        '''
        return np.clip((P.T.dot(X_s) - Y) * B, 0, np.inf)

    def update_Y1(self, X_s, X_t, P, Z, E, mu):
        '''
        Calcula el incremento en la actualización de Y_1, una matriz usada en
        el problema de optimización.

        Argumentos
        -----------------------
        X_s: numpy.ndarray shape(k, n).
            Datos del dominio fuente. Cada columna es una instancia.
        X_t: numpy.ndarray shape(k, m).
            Datos del dominio objetivo. Cada columna es una instancia.
        P: numpy.ndarray shape(k, d).
            Matriz de transformación de los datos.
        Z: numpy.ndarray shape(n, m).
            Matriz de reconstrucción de los datos.
        E: numpy.ndarray shape(d, m).
            Matriz de ruido de los datos.
        mu: float.
            Constante usada para el cálculo.

        Retorno
        ----------------------------
        increment_Y1: numpy.ndarray shape(d, m).
            Incremento de Y_1.
        '''
        return (
            mu * (P.T.dot(X_t) - np.linalg.multi_dot((P.T, X_s, Z)) - E)
        )

    def update_Y2(self, Z, Z_1, mu):
        '''
        Calcula el incremento en la actualización de Y_2, una matriz usada en
        el problema de optimización.

        Argumentos
        ------------------------
        Z: numpy.ndarray shape(n, m).
            Matriz de reconstrucción de los datos.
        Z_1: numpy.ndarray shape(n, m).
            Matriz usada para resolver el problema de optimización.
        mu: float.
            Constante usada para el cálculo.

        Retorno:
        increment_Y2: numpy.ndarray shape(d, m).
            Incremento de Y_2.
        '''
        return mu * (Z - Z_1)

    def update_Y3(self, Z, Z_2, mu):
        '''
        Calcula el incremento en la actualización de Y_3, una matriz usada en
        el problema de optimización.

        Argumentos
        --------------------------
        Z: numpy.ndarray shape(n, m).
            Matriz de reconstrucción de los datos.
        Z_2: numpy.ndarray shape(n, m).
            Matriz usada para resolver el problema de optimización.
        mu: float.
            Constante usada para el cálculo.

        Retorno
        ------------------------------
        increment_Y3: numpy.ndarray shape(d, m).
            Incremento de Y_3.
        '''
        return self.update_Y2(Z, Z_2, mu)

    def solve_mu(self, mu, rho, max_mu):
        '''
        Actualiza el valor de la constante mu.

        Argumentos
        --------------------------
        mu: float.
            Constante usada para el cálculo.
        rho: float.
            Constante fijada para incrementar el valor de mu
        max_mu: float.
            Máximo valor de mu.

        Retorno
        ---------------------------
        new_mu: float.
            Nuevo valor de mu.
        '''
        return min((rho * mu, max_mu))

    def solve_multipliers(self, X_s, X_t, P, Z, E, Y_1, Y_2, Y_3, Z_1, Z_2,
                          mu, rho, max_mu):
        '''
        Actualiza los multiplicadores del problema de optimización: Y_1, Y_2,
        Y_3 y mu.

        Argumentos
        -----------------------------
        X_s: numpy.ndarray shape(k, n).
            Datos del dominio fuente. Cada columna es una instancia.
        X_t: numpy.ndarray shape(k, m).
            Datos del dominio objetivo. Cada columna es una instancia.
        P: numpy.ndarray shape(k, d).
            Matriz de transformación de los datos.
        Z: numpy.ndarray shape(n, m).
            Matriz de reconstrucción de los datos.
        E: numpy.ndarray shape(d, m).
            Matriz de ruido de los datos.
        Y_1: numpy.ndarray shape(d, m).
            Matriz multiplicador usada para resolver el problema de
            optimización.
        Y_2: numpy.ndarray shape(d, m).
            Matriz multiplicador usada para resolver el problema de
            optimización.
        Y_3: numpy.ndarray shape(d, m).
            Matriz multiplicador usada para resolver el problema de
            optimización.
        Z_1: numpy.ndarray shape(n, m).
            Matriz usada para resolver el problema de optimización.
        Z_2: numpy.ndarray shape(n, m).
            Matriz usada para resolver el problema de optimización.
        mu: float.
            Constante usada para el cálculo.
        rho: float.
            Constante fijada para incrementar el valor de mu
        max_mu: float.
            Máximo valor de mu.

        Retorno
        -------------------------------------
        Y_1: numpy.ndarray shape(d, m).
            Matriz multiplicador actualizada.
        Y_2: numpy.ndarray shape(d, m).
            Matriz multiplicador actualizada.
        Y_3: numpy.ndarray shape(d, m).
            Matriz multiplicador actualizada.
        mu: float.
            Constante actualizada.
        '''
        return (
            Y_1 + self.update_Y1(X_s, X_t, P, Z, E, mu),
            Y_2 + self.update_Y2(Z, Z_1, mu),
            Y_3 + self.update_Y3(Z, Z_2, mu),
            self.solve_mu(mu, rho, max_mu)
        )

    def optimization_values(self, X_s, X_t, P, Z, E, Z_1, Z_2):
        '''
        Calcula los valores de cada uno de los sumandos del problema de
        optimización.

        Argumentos
        -------------------------------
        X_s: numpy.ndarray shape(k, n).
            Datos del dominio fuente. Cada columna es una instancia.
        X_t: numpy.ndarray shape(k, m).
            Datos del dominio objetivo. Cada columna es una instancia.
        P: numpy.ndarray shape(k, d).
            Matriz de transformación de los datos.
        Z: numpy.ndarray shape(n, m).
            Matriz de reconstrucción de los datos.
        E: numpy.ndarray shape(d, m).
            Matriz de ruido de los datos.
        Z_1: numpy.ndarray shape(n, m).
            Matriz usada para resolver el problema de optimización.
        Z_2: numpy.ndarray shape(n, m).
            Matriz usada para resolver el problema de optimización.

        Retorno
        -------------------------------------
        first: float
            Valor del primer sumando
        second: float
            Valor del segundo sumando
        third: float
            Valor del tercer sumando
        '''
        mu = 1
        first = np.linalg.norm(
                self.update_Y1(X_s, X_t, P, Z, E, mu), np.inf
        )
        second = np.linalg.norm(
                self.update_Y2(Z, Z_1, mu), np.inf
        )
        third = np.linalg.norm(
                self.update_Y3(Z, Z_2, mu), np.inf
        )

        return first, second, third

    def converged(self, first, second, third):
        '''
        Determina si el problema de optimización ha convergido.

        self.epsilon es el umbral.

        Argumentos
        -------------------------------
        last_first: float
            Valor anterior del primer sumando
        last_second: float
            Valor anterior del segundo sumando
        last_third: float
            Valor anterior del tercer sumando
        first: float
            Valor del primer sumando
        second: float
            Valor del segundo sumando
        third: float
            Valor del tercer sumando

        Retorno
        -------------------------------------
        converged: boolean.
            Determina si ha convergido o no.
        '''
        first_condition = first < self.epsilon
        second_condition = second < self.epsilon
        third_condition = third < self.epsilon
        return (
            first_condition and second_condition and third_condition
        )

    def solve(self, X_s, X_t, Y):
        '''
        Resuelve iterativamente el problema de optimización

        Argumentos
        ----------------------------------
        X_s: numpy.ndarray shape(k, n).
            Datos del dominio fuente. Cada columna es una instancia.
        X_t: numpy.ndarray shape(k, m).
            Datos del dominio objetivo. Cada columna es una instancia.
        Y: numpy.ndarray.shape(d, n).
            Etiquetas del dominio fuente binarizadas. Cada columna es una
            etiqueta.

        Retorno
        ---------------------------
        P: numpy.ndarray shape(k, d).
            Matriz de proyección de los datos
        '''
        # inicialización de variables
        B = self.compute_B(Y)
        mu = 0.1
        rho = 1.01
        max_mu = 10 ** 6
        M = np.ones(Y.shape)
        Z = Z_1 = Z_2 = np.zeros((X_s.shape[1], X_t.shape[1]))
        E = np.zeros((Y.shape[0], X_t.shape[1]))
        Y_1 = np.zeros(E.shape)
        Y_2 = Y_3 = np.zeros(Z.shape)
        # solución iterativa
        num_iter = 0
        while True:
            P = self.solve_P(X_s, X_t, Z, E, Y, B, M, Y_1, mu)
            Z = self.solve_Z(X_s, X_t, P, E, Y_1, Y_2, Y_3, Z_1, Z_2, mu)
            Z_1 = self.solve_Z1(Z, Y_2, mu)
            Z_2 = self.solve_Z2(Z, Y_3, mu)
            E = self.solve_E(X_s, X_t, P, Z, Y_1, mu)
            M = self.solve_M(X_s, Y, P, B)
            Y_1, Y_2, Y_3, mu = self.solve_multipliers(
                X_s, X_t, P, Z, E, Y_1, Y_2, Y_3, Z_1, Z_2, mu, rho, max_mu
            )
            first, second, third = self.optimization_values(
                    X_s, X_t, P, Z, E, Z_1, Z_2
            )
            if (self.converged(first, second, third)
                    or (num_iter >= self.max_iter)):
                break
            num_iter += 1
        return P

    def fit(self, source, target, source_labels):
        '''
        Ajusta los parámetros para la transformación.

        Argumentos
        --------------------------------
        source: numpy.ndarray shape(n, k).
            Datos del dominio fuente.
        target: numpy.ndarray shape(m, k).
            Datos del dominio objetivo.
        source_labels: numpy.ndarray.shape(n,).
            Etiquetas del dominio fuente.

        Retorno
        -------------------------
        self: DTS.
            El propio objeto.
        '''
        # Se transponen varias matrices: lo requiere el proceso.
        X_s = source.T
        X_t = target.T
        Y = data.binarize_labels(source_labels).T
        self.P = self.solve(X_s, X_t, Y)
        return self

    def transform(self, data):
        '''
        Transforma un conjunto de datos.

        Argumentos
        ----------------------------
        data: numpy.ndarray shape(n, k).
            Conjunto de datos a trnaformar.

        Retorno
        ---------------------------------
        transformed: numpy.ndarray shape(n, d).
            Datos transformados.
        '''
        return data.dot(self.P)

    def fit_transform(self, source, target, source_labels):
        '''
        Combina fit y transform. Devuelve la transformación de `source` y
        `target`.

        Argumentos
        -------------------------------
        source: numpy.ndarray shape(n, k).
            Datos del dominio fuente
        target: numpy.ndarray shape(m, k).
            Datos del dominio objetivo.
        source_labels: numpy.ndarray.shape(n, d).
            Etiquetas del dominio fuente.

        Retorno
        ----------------------------------
        transformed_source: numpy.ndarray shape(n, d).
            Datos fuente transformados.
        transformed_target: numpt.ndarray shape(m, d).
            Datos objetivo transformados.
        '''
        self.fit(source, target, source_labels)
        return self.transform(source), self.transform(target)
