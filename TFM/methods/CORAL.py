# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 10:16:39 2019

@author: David
"""
import numpy as np
import scipy as sp


class CORAL:
    '''
    Correlation Alignment. Adaptación de dominio sencilla. Estandariza datos,
    blanquea objetivo, colorea objetivo.

    Todas las funciones asumen que los datos de entrada tienen las
    instancias en filas y features en columnas.

    Construye la instancia.

    Argumentos
    ------------------------------
    labda: float. opcional (default=1.0)
        Constante que indica los valores en la matriz diagonal que se suma
        a las matrices de covarianza.

    Atributos
    -------------------------------
    covariance_inductor: numpy.ndarray shape(k, k).
         Inverso de la raíz cuadrada de la matriz de covarianza de los datos
         del dominio objetivo. Se fija tras llamar a `fit`.
    '''
    def __init__(self, labda=1.0):
        self.labda = labda

    def standarize(self, data):
        '''
        Estandariza datos: resta media y divide por desviación típica.

        Argumentos
        --------------------------
        data: numpy.ndarray shape(n,k).
            Datos a estandarizar.

        Retorno
        --------------------------
        standarized: numpy.ndarray shape (n, k).
            Datos estandarizados
        '''
        return (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-10)

    def square_root_covariance(self, data, inverse=True):
        '''
        Calcula la raíz cuadrada de la matriz de covarianza
        de los datos pasados como argumento.

        Argumentos
        ---------------------
        data: numpy.ndarray shape(n,k).
            Datos para los que se calcula la raíz cuadrada de la matriz de
            covarianza.
        inverse: boolean. opcional (default=True)
            Si es `True`, calcula el inverso de la raíz cuadrada de la matriz
            de covarianza.

        Retorno:
        root: numpy.ndarray shape(k, k).
            Raíz cuadrada de la matriz de covarianza de `data`.
        '''
        data_covariance = (
            np.cov(data, rowvar=False)
            + self.labda * np.identity(data.shape[1])
        )
        root = sp.linalg.sqrtm(data_covariance)
        if inverse:
            try:
                root = np.linalg.inv(root)
            except np.linalg.LinAlgError:
                root = np.linalg.pinv(root)
        return root

    def fit(self, data):
        '''
        Entrena el modelo

        Argumentos
        ----------------------
        data: numpy.ndarray shape(n, k).
            Datos del dominio fuente

        Retorno
        ----------------------------
        self: CORAL
            La propia instancia
        '''
        standarized = self.standarize(data)
        self.covariance_inductor = self.square_root_covariance(
                standarized, inverse=False
        )

        return self

    def transform(self, data):
        '''
        Transforma `data`
        1. estandarizando
        2. blanqueando (multiplicando por el inverso de la raíz cuadrada de
            la matriz de covarianza de `data`)
        3. coloreando (multiplicando por la raíz cuadrada de la matriz de
            covarianza de los datos usados en `fit`)

        Argumentos:
        data: numpy.ndarray shape(m, k).
            Datos a transformar.

        Retorno
        --------------------
        transformed: numpy.ndarray shape(m, k).
            Datos transformados.
        '''
        standarized = self.standarize(data)
        inverse_covariance_root = self.square_root_covariance(standarized)
        transformed = np.linalg.multi_dot(
            (
                standarized,
                inverse_covariance_root,
                self.covariance_inductor
            )
        )
        if transformed.dtype.kind == 'c':
            return np.real(transformed)
        return transformed

    def fit_transform(self, source, target):
        '''
        Combina fit y transform. `target` es el argumento para fit, lo que
        implica que será source el que resulte transformado, mientras que
        target solo se devolverá estandarizado.

        Argumentos
        -------------------------
        source: numpy.ndarray shape(n, k).
            Datos del dominio fuente
        target: numpy.ndarray shape(m, k).
            Datos del dominio objetivo.

        Retorno
        --------------------------
        transformed_source: numpy.ndarray shape(n, k).
            Datos del dominio fuente transformados.
        transformed_target: numpy.ndarray shape(m, k).
            Datos del dominio objetivo estandarizados.
        '''
        target_standarized = self.standarize(target)
        self.fit(target)
        return self.transform(source), target_standarized
