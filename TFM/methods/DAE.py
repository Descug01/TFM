# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:23:15 2019

@author: David
"""
import tensorflow as tf
import numpy as np
import functools
from .util import data

ACTIVATIONS = {"sigmoid": tf.nn.sigmoid, "relu": tf.nn.relu,
               'tanh': tf.nn.tanh, "leaky_relu": tf.nn.leaky_relu}


class DAE():
    '''
    Deep autoencoder para adaptación de dominio.
    4 capas:
    1. Embedding. Reduce la dimensionalidad.
    2. Clasificación. Usa una activación softmax siempre.
    3. Inversión de clasificación. Reconstruye el embedding.
    4. Reconstrucción. Invierte el embedding para devolver los datos de
        entrada.

    Inicializa la instancia

    Argumentos
    --------------------------
    num_features: int.
        Número de features que tendrán los datos transformados.
    alpha: float. 0 < alpha
        Constante de tradeoff que determina la importancia que se le da al
        término de convergencia de Kullback-Leibler en el problema de
        optimización.
    beta: float. 0 < beta.
        Constante de tradeoff que determina la importancia que se le da al
        término del error de clasificación del dominio fuente en el
        problema de optimización.
    gamma: float 0 < gamma.
        Constante de tradeoff que determina la importancia que se le da al
        término de regularización en el problema de optimización.
    num_epochs: int.
        Número de iteraciones de entrenamiento del modelo. En estas
        iteraciones se intenta minimizar el problema completo.
    activation: string. Definida en `ACTIVATIONS`. opcional (default='sigmoid')
        Función de activación de las capas ocultas.
    weight_initialization_epochs: int.
        Número de iteraciones previas al entrenamiento para inicializar los
        pesos. En estas iteraciones únicamente se minimiza el error de
        reconstrucción de los datos de entrada.
    convergence_threshold: float. convergence_threshold << 1.
        Constante que determina la parada del entrenamiento cuando la
        magnitud de la mejora del problema de optimización sea menor
        que este valor.

    Atributos
    -------------------------------------
    graph: tensorflow.Graph
        Grafo de computación que define las operaciones del modelo.
    session: tensorflow.Session
        Sesión para ejecutar las opearciones del grafo y obtener resultados
        numéricos.
    reconstruction_error: Tensor shape [0]
        Tensor que computa el error de reconstrucción.
    loss: Tensor shape [0]
        Tensor que computa el coste del prroblema de optimización del modelo.
    weight_init: Tensor
        Operación que optimiza `reconstructuon_error` con el método Adam.
    operation: Tensor
        Operación que optimiza `loss` con el método Adam.
    '''
    def __init__(self, num_features, alpha, beta, gamma,
                 num_epochs=100, activation="sigmoid",
                 weight_initialization_epochs=0, convergence_threshold=1e-4):
        self.num_features = num_features
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.graph = None
        self.num_epochs = num_epochs
        self.weight_initialization_epochs = weight_initialization_epochs
        self.activation = ACTIVATIONS[activation]
        self.convergence_threshold = convergence_threshold
        self.session = None

    def tensor_inputs(self, source, target, source_labels):
        '''
        Inicializa las entradas de datos del modelo en el grafo de ejecución
        de tensorflow.

        Argumentos
        -------------------------
        source: numpy.ndarray shape(m, k).
            Datos del dominio fuente.
        target: numpy.ndarray shape(n, k).
            Datos del dominio obejtivo.
        source_labels: numpy.ndarray shape(m, ).
            Etiquetas del dominio fuente.
        '''
        self.Xs = tf.placeholder(
                dtype=tf.dtypes.float64, shape=[None, source.shape[1]],
                name="Xs"
        )
        self.Xt = tf.placeholder(
                dtype=tf.dtypes.float64, shape=[None, target.shape[1]],
                name="Xt"
        )
        num_unique_labels = np.unique(source_labels).size
        self.Ys = tf.placeholder(
                dtype=tf.dtypes.float64, shape=[None, num_unique_labels],
                name="Ys"
        )

    def initialize_weights(self, source, target, source_labels):
        '''
        Crea las variables correspondientes a los pesos y bias de cada capa
        del modelo en el grafo de ejecución.

        Argumentos
        -------------------------
        source: numpy.ndarray shape(m, k).
            Datos del dominio fuente.
        target: numpy.ndarray shape(n, k).
            Datos del dominio obejtivo.
        source_labels: numpy.ndarray shape(m, ).
            Etiquetas del dominio fuente.
        '''
        num_unique_labels = np.unique(source_labels).size
        weight_initializer = tf.initializers.truncated_normal(
                mean=0.1, stddev=0.01, seed=0, dtype=tf.dtypes.float64
        )
        bias_initializer = tf.initializers.zeros(dtype=tf.dtypes.float64)

        embedding_shape = [source.shape[1], self.num_features]
        labeling_shape = [self.num_features, num_unique_labels]
        embedding_bias_shape = [1, self.num_features]
        labeling_bias_shape = [1, num_unique_labels]
        reconstruction_bias_shape = [1, source.shape[1]]
        weight_create = functools.partial(
                tf.get_variable, initializer=weight_initializer,
                dtype=tf.dtypes.float64
        )
        bias_create = functools.partial(
                tf.get_variable, initializer=bias_initializer,
                dtype=tf.dtypes.float64
        )
        self.W1 = weight_create(
                name="W1", shape=embedding_shape, trainable=True
        )
        self.W2 = weight_create(
                name="W2", shape=labeling_shape, trainable=True
        )
        self.W2_hat = weight_create(
                name="W2_hat", shape=labeling_shape[::-1], trainable=True
        )
        self.W1_hat = weight_create(
                name="W1_hat", shape=embedding_shape[::-1], trainable=True
        )
        self.b1 = bias_create(
                name="b1", shape=embedding_bias_shape, trainable=True
        )
        self.b2 = bias_create(
                name="b2", shape=labeling_bias_shape, trainable=True
        )
        self.b2_hat = bias_create(
                name="b2_hat", shape=embedding_bias_shape, trainable=True
        )
        self.b1_hat = bias_create(
                name="b1_hat", shape=reconstruction_bias_shape, trainable=True
        )

    def layer_pass_single(self, data, weights, bias, activation):
        '''
        Realizada una pasada por una capa a los datos pasados como argumento.

        La forma de una pasada con una función de activación f, unos pesos W,
        entrada X y bias b es f(XW + b).

        Argumentos
        -------------------------
        data: Tensor shape(a, k).
            Datos de entrada a la capa.
        weights: tensorflow.Variable shape(k, m)
            Tensor con los pesos de las neuronas de la capa.
        bias: tensorflow.Variable shape(1, m).
            Tensor con los términos de bias de las neuronas de la capa.
        activation: function. Debe estrar definida en `ACTIVATIONS`
            Función que recibe un Tensor y devuelve otro Tensor de las mismas
            dimensiones al que se le ha aplicado una función.

        Retorno
        ----------------------------
        process: function.
            Recibe como argumento un Tensor que representa la entrada a la capa
            y devuelve la salida de dicha capa en forma de Tensor.
        '''
        return activation(tf.add(tf.matmul(data, weights), bias))

    def layer_pass(self, source_input, target_input, weights, bias,
                   activation):
        '''
        Calcula la salida de una capa para las entrada fuente y objetivo.

        Argumentos
        -------------------------
        source_input: Tensor shape (a, k).
            Entradas correspondientes al dominio fuente.
        target_input: Tensor shape (b, k).
            Entradas correspondientes al dominio fuente.
        weights: tensorflow.Variable shape(k, m)
            Tensor con los pesos de las neuronas de la capa.
        bias: tensorflow.Variable shape(1, m).
            Tensor con los términos de bias de las neuronas de la capa.
        activation: function. Debe estrar definida en `ACTIVATIONS`
            Función que recibe un Tensor y devuelve otro Tensor de las mismas
            dimensiones al que se le ha aplicado una función.

        Retorno
        --------------------------
        processed_source: Tensor shape (a, m).
            Salida de la capa para la entrada del dominio fuente.
        processed_target: Tensor shape (b, m).
            Salida de la capa para la entrada del dominio objetivo.
        '''
        processed_source = self.layer_pass_single(
                source_input, weights, bias, activation
        )
        processed_target = self.layer_pass_single(
                target_input, weights, bias, activation
        )

        return processed_source, processed_target

    def compute_reconstruction_error(self, source, reconstructed_source,
                                     target, reconstructed_target):
        '''
        Calcula el término del problema de optimización correspondiente
        al error de reconstrucción de las instancias. El error es la suma
        de los errores de reconstrucción del dominio objetivo y fuente.

        Para un conjunto de instancias X y su reconstrucción Xhat, el error de
        reconstrucción es:

        $$\sum_i || X_i - \hat{X}_i ||$$

        Argumentos
        ----------------------------------
        source: Tensor shape(a, k)
            Instacias originales del dominio fuente.
        reconstructed_source: shape(a, k)
            Instancias reconstrucidas por el modelo del dominio fuente
        target: Tensor shape(b, k)
            Instacias originales del dominio objetivo.
        reconstructed_target: shape(b, k)
            Instancias reconstrucidas por el modelo del dominio objetivo

        Retorno
        ------------------------------
        reconstruction_error: Tensor shape(,)
            Error de reconstrucción.
        '''
        return (
                tf.losses.mean_squared_error(source, reconstructed_source)
                + tf.losses.mean_squared_error(target, reconstructed_target)
        )

    def kl(self, source, target):
        '''
        Calcula la divergencia de Kullback-Leibler simétrica entre las
        distribuciones source y target.

        La divergencia de Kullback-Leibler entre dos distribuciones se calcula
        como:
        $$KL(S||T) = \sum_i S_i * log(S_i / T_i)$$

        Este cálculo se realiza de forma simétrica, es decir como
        $$KL(S||T) + KL(T||S)$$.

        Argumentos
        ---------------------------
        source: Tensor shape(k,)
            Distribución del dominio fuente.
        target: Tensor shape(k,)
            Distribución del dominio objetivo.

        Retorno
        ------------------------------
        kl: Tensor shape(, )
            Divergencia de Kullback-Leibler simétrica entre `source`y `target`.
        '''
        return (
                tf.reduce_sum(source * tf.log(source / target))
                + tf.reduce_sum(target * tf.log(target / source))
        )

    def feature_mean(self, data):
        '''
        Calcula la media para cada una de las columnas (features) de los datos
        y las reduce para que sumen 1 dividiendo entre las suma de dichas
        medias.

        Argumentos
        ------------------------
        data: Tensor shape(a, k)
            Datos cuyas medias se van a calcular.

        Retorno
        -------------------------
        reduced: Tensor shape(k, )
            Medias reducidas de las features.
        '''
        mean = tf.reduce_mean(data, axis=0)
        return mean / tf.reduce_sum(mean)

    def KL_divergence(self, epsilon_source, epsilon_target):
        '''
        Calcula el término del problema de optimización correspondiente a
        la divergencia de Kullback-Leibler.

        Argumentos
        ------------------------------
        epsilon_source: Tensor shape(a, k)
            Embedding de las instancias del dominio fuente.
        epsilon_target: Tensor shape(b, k)
            Embedding de las instancias del dominio objetivo.

        Retorno
        ----------------------------
        kl: Tensor shape(,)
            Divergencia simétrica de Kullback-Leibler entre fuente y obejtivo.
        '''
        means_s = self.feature_mean(epsilon_source)
        means_t = self.feature_mean(epsilon_target)

        return self.kl(means_s, means_t)

    def classification_error(self, predicted, true_labels):
        '''
        Calcula el término del problema de optimización correspondiente al
        error de clasificación cometido en las instancias del dominio fuente.

        El error se calcula como:
        $$\sum_i log(predicted[i, correct]) / a$$
        Donde correct es el índice de la clase verdadera correspondiente a la
        instancia en `true_labels`

        Argumentos
        -------------------------
        predicted: Tensor shape(a, c)
            Salida en bruto de la capa de clasificación que contiene en cada
            fila las probabilidades de que la instancia correspondiente
            pertenezca a una de las c clases.
        true_labels: Tensor shape(a, c)
            Etiquetas reales binarizadas de las instancias.

        Retorno
        -----------------------------
        classification_error: Tensor shape(, )
            Error de clasificación.
        '''
        pred = predicted
        argmax = tf.argmax(true_labels, axis=1)
        # La dimensión no está definida así que .shape dará ? y eso no es
        # un entero. tf.shape() fuerza una evaluación del grafo de operaciones
        # lo que permite obtener la dimensión actual.
        row_range = tf.range(
                tf.cast(tf.shape(argmax)[0], tf.dtypes.int64),
                dtype=tf.dtypes.int64
        )
        indices = tf.stack([row_range, argmax], axis=1)
        selected = tf.gather_nd(pred, indices)
        loss = - tf.reduce_mean(tf.math.log(selected))

        return loss

    def regularization_term(self):
        '''
        Calcula el término correspondiente a la regularización del problema de
        optimización. Este término se define como la suma del cuadrado de
        las normas l2 de cada uno de los pesos y bias de cada capa. La norma l2
        de una matriz M se define como
        $$||M|| = \sum_i sqrt(\sum_j M^2_{i,j})$$

        Retorno
        -------------------------
        regularization: Tensor shape(, )
            Valor del término de regularización.
        '''
        return tf.reduce_sum(
                [2 * tf.nn.l2_loss(var) for var in tf.trainable_variables()]
        )

    def until_convergence(self, operation, gradient, feed_dict, max_iters):
        '''
        Ejecuta el proceso de entramiento hasta que se alcance convergencia o
        se realicen las iteraciones definidas.

        La convergencia viene definida en self.convergence_threshold, que es la
        magnitud de la mejora (reducción del valor de optimización) mínima que
        debe producirse para que se siga ejecutando.

        Argumentos
        ---------------------------------
        operation: Tensor shape(, )
            El error que se busca minimizar.
        gradient: Tensor shape(, )
            El gradiente de `operation` (el error).
        feed_dict: dict.
            Las entradas del grafo de computación del que se obtiene
            `operation`.
        max_iters: int.
            Número máximo de iteraciones de entrenamiento.
        '''
        last = np.inf
        current = np.inf
        converged = False
        num_iters = 0
        # opt = tf.train.AdamOptimizer().minimize(operation)
        while not converged and num_iters < max_iters:
            # print(num_iters)
            last = current
            current = self.session.run(operation, feed_dict=feed_dict)
            # print(current)
            self.session.run(gradient, feed_dict=feed_dict)
            # Esto asume que el error va descender monótonamente, si llega a
            # diverger no parará
            if (last > current
                    and (last - current) < self.convergence_threshold):
                converged = True
            num_iters += 1

    def train(self, source, target, source_labels):
        '''
        Crea el grafo de computación del modelo y  realiza el entrenamiento.

        Argumentos
        -------------------------------
        source: numpy.ndarray shape(m, k).
            Datos del dominio fuente.
        target: numpy.ndarray shape(n, k).
            Datos del dominio obejtivo.
        source_labels: numpy.ndarray shape(m, ).
            Etiquetas del dominio fuente.

        Retorno
        ------------------------------------
        self: DAE.
            La propia instancia.
        '''
        binarized_labels = data.binarize_labels(source_labels)
        self.session = tf.Session(graph=self.graph)
        self.session.run(tf.global_variables_initializer())
        feed_dict = {self.Xs: source, self.Xt: target}
        # for i in range(self.num_epochs):
        # Un número muy elevado de iteraciones de inicialización parece
        # que perjudica el rendimiento.
        # for i in range(self.weight_initialization_epochs):
        #    self.session.run(self.weight_init, feed_dict=feed_dict)
        self.until_convergence(
                self.reconstruction_error, self.weight_init, feed_dict,
                self.weight_initialization_epochs
        )
        feed_dict[self.Ys] = binarized_labels
        # for i in range(self.num_epochs):
        #    self.session.run(self.operation, feed_dict=feed_dict)
        self.until_convergence(
                self.loss, self.operation, feed_dict, self.num_epochs
        )
        # self.session.close()
        return self

    def embedding(self):
        '''
        Calcula la salida de la capa de embedding.

        Retorno
        -----------------------------
        processed_source: Tensor shape (a, m).
            Salida de la capa para la entrada del dominio fuente.
        processed_target: Tensor shape (b, m).
            Salida de la capa para la entrada del dominio objetivo.
        '''
        return self.layer_pass(
                self.Xs, self.Xt, self.W1, self.b1, self.activation
        )

    def softmax(self):
        '''
        Calcula la salida de la capa de calsificación.

        Retorno
        -----------------------------
        processed_source: Tensor shape (a, m).
            Salida de la capa para la entrada del dominio fuente.
        processed_target: Tensor shape (b, m).
            Salida de la capa para la entrada del dominio objetivo.
        '''
        return self.layer_pass(
                self.epsilon_s, self.epsilon_t, self.W2, self.b2, tf.nn.softmax
        )

    def inverse_softmax(self, z_s, z_t):
        '''
        Calcula la salida de la capa de inversión de clasificación.

        Retorno
        -----------------------------
        processed_source: Tensor shape (a, m).
            Salida de la capa para la entrada del dominio fuente.
        processed_target: Tensor shape (b, m).
            Salida de la capa para la entrada del dominio objetivo.
        '''
        return self.layer_pass(
                z_s, z_t, self.W2_hat, self.b2_hat, self.activation
        )

    def reconstruction(self, epsilon_s_hat, epsilon_t_hat):
        '''
        Calcula la salida de la capa de inversión de embedding.

        Retorno
        -----------------------------
        processed_source: Tensor shape (a, m).
            Salida de la capa para la entrada del dominio fuente.
        processed_target: Tensor shape (b, m).
            Salida de la capa para la entrada del dominio objetivo.
        '''
        return self.layer_pass(
                epsilon_s_hat, epsilon_t_hat, self.W1_hat, self.b1_hat,
                self.activation
        )

    def train_operations(self, reconstruction_s, reconstruction_t, z_s):
        '''
        Añade al grafo de computación las operaciones de cálculo de error
        global y de reconstrucción, así como sus gradientes.

        Argumentos
        ---------------------------
        reconstruction_s: Tensor shape(a, k)
            reconstrucción de las instancias del dominio fuente
        reconstruction_t: Tensor shape(b, k)
            reconstrucción de las instancias del dominio obejtivo
        z_s: tensor shape(a, c)
            salida de la capa de clasificación de las instancias del dominio
            fuente.
        '''
        reconstruction_error = self.compute_reconstruction_error(
                self.Xs, reconstruction_s, self.Xt, reconstruction_t
        )
        KL_divergence = self.KL_divergence(self.epsilon_s, self.epsilon_t)
        classification_error = self.classification_error(z_s, self.Ys)
        regularization = self.regularization_term()
        self.reconstruction_error = tf.cast(
                reconstruction_error, tf.dtypes.float64
        )
        self.loss = (
                self.reconstruction_error + self.alpha * KL_divergence
                + self.beta * classification_error
                + self.gamma * regularization
        )

        self.weight_init = tf.train.AdamOptimizer().minimize(
                self.reconstruction_error
        )
        self.operation = tf.train.AdamOptimizer().minimize(self.loss)

    def fit(self, source, target, source_labels):
        '''
        Inicializa y entrena el modelo.

        Argumentos
        ------------------------------
        source: numpy.ndarray shape(m, k).
            Datos del dominio fuente.
        target: numpy.ndarray shape(n, k).
            Datos del dominio obejtivo.
        source_labels: numpy.ndarray shape(m, ).
            Etiquetas del dominio fuente.

        Retorno
        ----------------------------
        self: DAE.
            La propia instancia.
        '''
        if self.session:
            self.session.close()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tensor_inputs(source, target, source_labels)
            self.initialize_weights(source, target, source_labels)
            self.epsilon_s, self.epsilon_t = self.embedding()
            z_s, z_t = self.softmax()
            epsilon_s_hat, epsilon_t_hat = self.inverse_softmax(z_s, z_t)
            reconstruction_s, reconstruction_t = self.reconstruction(
                    epsilon_s_hat, epsilon_t_hat
            )
            self.train_operations(reconstruction_s, reconstruction_t, z_s)
            self.train(source, target, source_labels)
        return self

    def fit_transform(self, source, target, source_labels):
        '''
        Inicializa y entrena el modelo y además devuelve la trnasformación
        de los dominios fuente y objetivo.

        Argumentos
        ------------------------------
        source: numpy.ndarray shape(m, k).
            Datos del dominio fuente.
        target: numpy.ndarray shape(n, k).
            Datos del dominio obejtivo.
        source_labels: numpy.ndarray shape(m, ).
            Etiquetas del dominio fuente.

        Retorno
        ----------------------------
        transformed_source: numpy.ndarray shape(m, self.num_features)
            Datos del dominio fuente transformados
        transformed_target: numpy.ndarray shape(n, self.num_features)
            Datos del dominio obejtivo transformados
        '''
        self.fit(source, target, source_labels)
        return self.session.run(
                [self.epsilon_s, self.epsilon_t],
                feed_dict={self.Xs: source, self.Xt: target}
        )
