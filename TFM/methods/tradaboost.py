# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:50:30 2019

@author: David
"""
import functools
import numpy as np
import sklearn.tree


class RareTrAdaBoost():
    '''
    Implementación de Boosting adaptado a transfer learning. Al contrario que
    otros, este método crea un modelo de tipo ensemble y no simplemente
    transforma los datos.

    Argumentos
    ----------------------------------
    num_iter: int.
        El número de clasificadores base del ensemble.
        Para la clasificación una vez el modelo esté entrenado solo
        se tienen en cuenta los (num_iterations / 2) últimos clasificadores
        entrenados.
    base_classifier: una clase clasificador de scikit-learn.
        Los clasificadores entrenados pertenecerán a esta clase. Este
        clasificador debe soportar sample weighting en su método fit.
    *args: tuple.
        argumentos para el constructor de base_classifier.
    **kwargs: dict.
        argumentos para el constructor de base_classifier.

    Atributos
    -------------------------------------
    classifiers: list.
        Lista de clasificadores entrenados durante el proceso de boosting.
    classifier_weights: numpy.ndarray shape(self.num_iter // 2, )
        Peso de los clasificadores, entendido como $$\beta_{tar}$$ calculado
        para el clasificador correspondiente.
    weights: numpy.ndarray shape(m, )
        Lista de pesos para las instancias source y target pasadas como
        argumento a fit. En primer lugar están los pesos de las instancias de
        source y a continuación las de target.
    classes: numpy.ndarray shape(c, )
        Array con las clases únicas de los datos de entrenamiento. Usado para
        la función predict.

    '''
    def __init__(self, num_iter, base_classifier=None, *args, **kwargs):
        self.num_iter = num_iter
        if base_classifier is None:
            base_classifier = functools.partial(
                    sklearn.tree.DecisionTreeClassifier, max_depth=1
            )
        self.base_classifier = functools.partial(
                base_classifier, *args, **kwargs
        )
        self.classifiers = []
        self.classifier_weights = []

    def normalize_weights(self, weights):
        '''
        Normaliza a uno un array correspondiente a pesos de las instancias

        Esta normalización hace que la suma de los pesos sea igual a 1 y se
        realiza dividiendo entre el sumatorio de los pesos.

        Argumentos
        -----------------------------
        weights: numpy.ndarray shape(n, ).
            Array con pesos correspndientes a instancias.

        Retorno
        -----------------------------
        normalized_weights: numpy.ndarray shape(n, ).
            Array original normalizado a suma 1.
        '''
        return weights / weights.sum()

    def predict(self, data):
        '''
        Predice las etiquetas de instancias.

        Argumentos
        ----------------------------
        data: numpy.ndarray shape(n, k).
            Datos sobre los que se va a predecir. Cada fila corresponde a una
            instancia.

        Retorno
        -------------------------------
        predicted: numpy.ndarray shape(n, ).
            Etiquets predichas para cada instancia.
        '''
        predicted = np.array(
            [classifier.predict(data) for classifier in self.classifiers]
        )
        classes = self.classes[:, None, None]
        weights = self.classifier_weights[:, None]
        correctly_predicted = predicted == classes
        points = correctly_predicted * weights
        summated = points.sum(axis=1)
        selected = summated.argmax(axis=0)

        return self.classes[selected]

    def compute_beta_source(self, num_source):
        '''
        Calcula el coeficiente de actualización para los pesos de las
        instancias del dominio fuente.

        Argumentos
        ------------------------------
        num_source: int.
            Número de instancias del dominio fuente.

        Retorno
        ----------------------------
        beta_source: float.
            Constante de actualización de los pesos.
        '''
        return 1 / (1 + np.sqrt(2 * np.log(num_source) / self.num_iter))

    def construct_candidate(self, train_data, train_labels, weights):
        '''
        Construye y enttrena un clasificador candidato para realizar
        predicciones.

        Argumentos
        -----------------------------------
        train_data: numpy.ndarray shape (n, k).
            Datos de entrenamiento. Cada fila es una instancia.
        train_labels: numpy.ndarray shape (n, ).
            Etiquetas correspondientes a las instancias.

        Retorno
        ----------------------------
        classifier: self.base_classifier.
            Clasificador entrenado sobre train_data y train_labels.
        '''
        return self.base_classifier().fit(
                train_data, train_labels, sample_weight=weights
        )

    def weighted_error(self, predicted_labels, true_labels, weights):
        '''
        Calcula el error ponderado para un dominio y devuelve un
        array booleano que indica las instancias mal clasificadas.

        Argumentos
        --------------------------------------
        predicted_labels: numpy.ndarray shape(n, ).
            Etiquetas predichas para las instancias del dominio fuente.
        true_labels: numpy.ndarray shape (n, )
            Etiquetas verdaderas del dominio fuente.
        weights: numpy.ndarray shape (n, ).
            Pesos de las instancias del dominio fuente.

        Retorno
        --------------------
        weighted_error: float.
            Error ponderado.
        missclassified: numpy.ndarray shape(n, ).
            Array booleano que indica que instancias se clasificaron mal con
            el valor True en la posición correspondiente.
        '''
        missclassified = predicted_labels != true_labels
        return np.average(missclassified, weights=weights), missclassified

    def update_weights(self, weights, predicted_labels, true_labels,
                       num_source, num_classes, beta_source):
        '''
        Actualiza los pesos de las instancias.

        Argumentos
        --------------------------------------------
        weigths: numpy.ndarray shape (num_source + num_target, ).
            Pesos de todas las instancias.
        predicted_labels: numpy.ndarray shape(num_source + num_target, ).
            Etiquetas predichas para las instancias del dominio fuente.
        true_labels: numpy.ndarray shape (num_source + num_target, ).
            Etiquetas verdaderas del dominio fuente.
        num_source: int. Número de instancias del dominio fuente.
            Los elementos de las num_source primeras posiciones de los
            anteriores arrays se consideran correspondientes al dominio fuente,
            el resto al dominio objetivo.
        num_classes: int.
            Número de clases diferentes del problema.
        beta_source: float.
            Constante de actualización de los pesos del dominio fuente.

        Retorno
        --------------------------------------
        alpha_t: float.
            Constante de actualización general de los pesos objetivo.
        '''

        source_slice = slice(num_source)
        target_slice = slice(num_source, weights.size)
        source_weights = weights[source_slice]
        target_weights = weights[target_slice]
        source_predicted = predicted_labels[source_slice]
        target_predicted = predicted_labels[target_slice]
        source_true = true_labels[source_slice]
        target_true = true_labels[target_slice]

        error, missclassified = self.weighted_error(
                predicted_labels, true_labels, weights
        )
        if error <= 0:
            return 1
        elif error > 1 - (1 / num_classes):
            return None
        error_source, source_missclassified = self.weighted_error(
                source_predicted, source_true, source_weights
        )
        error_target, target_missclassified = self.weighted_error(
                target_predicted, target_true, target_weights
        )
        alpha_s = (1 / 2) * np.log(1 + np.sqrt(2 * np.sqrt(num_source / self.num_iter)))
        alpha_t = np.log(num_classes - 1) + np.log((1 - error) / error)
        source_weights *= np.exp(-alpha_s * source_missclassified)
        target_weights *= np.exp(alpha_t * target_missclassified)

        return alpha_t

    def fit(self, source, target, source_labels, target_labels):
        '''
        Construye el modelo.

        Como esto es básicamente un modelo, hay que tener en cuenta que los
        datos fuente y objetivo pasados como argumento deben corresponder a
        la fracción destinada al entrenamiento, con datos de test reservados
        aparte.

        Argumentos
        -------------------------------
        source: numpy.ndarray shape(n, k).
            Datos del dominio fuente.
        target: numpy.ndarray shape(m, k).
            Datos del dominio objetivo.
        source_labels: numpy.ndarray.shape(n,).
            Etiquetas del dominio fuente.
        target_labels: numpy.ndarray.shape(m,).
            Etiquetas del dominio objetivo.

        Retorno
        -------------------------------
        self: RareTrAdaboost
            La propia instancia.
        '''
        num_source = source.shape[0]
        train_data = np.vstack((source, target))
        train_labels = np.concatenate((source_labels, target_labels))
        self.classes = np.unique(train_labels)
        num_classes = self.classes.size
        weights = np.ones(train_data.shape[0])
        beta_source = self.compute_beta_source(num_source)
        for iteration in range(self.num_iter):
            weights = self.normalize_weights(weights)
            candidate = self.construct_candidate(
                    train_data, train_labels, weights
            )
            predictions = candidate.predict(train_data)
            beta_target = self.update_weights(
                    weights, predictions, train_labels, num_source,
                    num_classes, beta_source
            )
            if beta_target is not None:
                self.classifiers.append(candidate)
                self.classifier_weights.append(beta_target)
        self.classifier_weights = np.array(self.classifier_weights)
        return self

    def fit_predict(self, source, target, source_labels, target_labels,
                    predict_data):
        '''
        Construye el modelo y predice.

        Argumentos
        -------------------------------
        source: numpy.ndarray shape(n, k).
            Datos del dominio fuente.
        target: numpy.ndarray shape(m, k).
            Datos del dominio objetivo.
        source_labels: numpy.ndarray.shape(n,).
            Etiquetas del dominio fuente.
        target_labels: numpy.ndarray.shape(m,).
            Etiquetas del dominio objetivo.
        data: numpy.ndarray shape(n, k).
            Datos sobre los que se va a predecir. Cada fila corresponde a una
            instancia.

        Retorno
        -------------------------------
        predicted: numpy.ndarray shape(n, ).
            Etiquets predichas para cada instancia.
        '''
        self.classifiers = []
        self.classifier_weights = []
        self.fit(source, target, source_labels, target_labels)
        return self.predict(predict_data)
