# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:47:08 2019

@author: David
"""
import os
import pandas as pd
from methods.util import data
import numpy as np


def files(domains, path):
    '''
    Crea un diccionario con claves iguales a los elementos de `domains`,
    correspondientes a nombres de dominios de un dataset y valores
    correspondiente a la ruta del fichero que contiene los datos de ese
    dominio.

    Se asume que los archivos dentro `path` tienen el mismo nombre que los
    reflejados en `domains` y los datos que contienen corresponden a ese
    dominio. Por ejemplo, si domains es ['a', 'b'] dentro de `path` debe haber
    2 ficheros 'a.csv' y 'b.csv' que contienen los datos del dominio.

    Argumentos
    ---------------------------------------
    domains: iterable.
        Contiene los nombres de los diferentes dominios del dataset.
    path: string
        Ruta del directorio con los archivos de los dominios del dataset.

    Retorno
    ---------------------------------------
    files_dict: dict.
        El con las claves y valores descritos.
    '''
    return {domain: os.path.join(path, domain + '.csv') for domain in domains}


PATH = r'../datos'
OFFICE_CALTECH_DIRECTORY = os.path.join(PATH, r'office_caltech')
OFFICE_CALTECH_DOMAINS = ['amazon', 'caltech10', 'dslr', 'webcam']
OFFICE_CALTECH_FILES = files(OFFICE_CALTECH_DOMAINS, OFFICE_CALTECH_DIRECTORY)

IMAGENET_DIRECTORY = os.path.join(PATH, r'imagenet')
IMAGENET_DOMAINS = ['ambulance', 'jeep', 'minivan', 'passenger_car', 'taxi']
IMAGENET_FILES = files(IMAGENET_DOMAINS, IMAGENET_DIRECTORY)

ROBOTS_DIRECTORY = os.path.join(PATH, r'robots')
ROBOTS_MAJORITY = 'dos'
ROBOTS_DOMAINS = {'normal', 'dos', 'spoofing'}
ROBOTS_FILES = files(ROBOTS_DOMAINS, ROBOTS_DIRECTORY)

URL_DIRECTORY = os.path.join(PATH, r'URL')
URL_MAJORITY = 'defacement'
URL_DOMAINS = {'normal', 'defacement', 'malware', 'phishing', 'spam'}
URL_FILES = files(URL_DOMAINS, URL_DIRECTORY)

NSLKDD_DIRECTORY = os.path.join(PATH, r'NSL_KDD')
NSLKDD_MAJORITY = 'dos'
NSLKDD_DOMAINS = {'normal', 'dos', 'probe', 'u2r', 'r2l'}
NSLKDD_FILES = files(NSLKDD_DOMAINS, NSLKDD_DIRECTORY)

ANDROID_DIRECTORY = os.path.join(PATH, r'AndroidMalware')
ANDROID_MAJORITY = 'adware'
ANDROID_DOMAINS = {'normal', 'adware', 'ransomware', 'scareware', 'smsmalware'}
ANDROID_FILES = files(ANDROID_DOMAINS, ANDROID_DIRECTORY)


def load(file_source, file_target):
    '''
    Carga 2 ficheros de datos y separa las features de las etiquetas.

    Los argumentos deben ser claves del diccionario FILES del dataset
    correspondiente.

    Argumentos
    ----------------------------
    files_source: string.
        Nombre del archivo del primer dominio sin extension.
    file_target: string.
        Nombre del archivo del segundo dominio sin extension.
    target: object.
        Nombre de columna de las etiquetas.

    Retorno
    ------------------------------
    first_domain: tuple
        Tupla que contiene los siguientes elementos:
            first_data: numpy.ndarray shape (n, k - 1). Dominio uno con la
                columna target eliminada.
            first_target: numpy.ndarray shape (n, ). Columna target del
                primer dominio.
    second_domain: tuple
        Tupla que contiene los siguientes elementos:
            second_data: numpy.ndarray shape (n, k - 1). Dominio uno con la
                columna target eliminada.
            second_target: numpy.ndarray shape (n, ). Columna target del
                primer dominio.
    '''
    df1 = pd.read_csv(file_source, sep=',', header=None)
    df2 = pd.read_csv(file_target, sep=',', header=None)
    return (data.split_data_label(df1),
            data.split_data_label(df2))


def get_paths(dictionary, source_domain, target_domain):
    return (dictionary['normal'], dictionary[source_domain],
            dictionary[target_domain])


def load_file(path):
    return pd.read_csv(path, sep=',', header=None)


def load_data_problem(path_normal, path_source, path_target):
    source = load_file(path_source).values
    normal = load_file(path_normal).values
    target = load_file(path_target).values

    proportion_source = source.shape[0] / normal.shape[0]
    mask = np.random.rand(normal.shape[0]) < proportion_source
    normal_source = normal[mask]

    source = np.vstack([source, normal_source])
    normal = normal[~mask]

    proportion_target = target.shape[0] / normal.shape[0]
    mask = np.random.rand(normal.shape[0]) < proportion_target
    normal_target = normal[mask]

    target = np.vstack([target, normal_target])

    return (
        data.split_data_label(source),
        data.split_data_label(target)
    )


def load_robots(source_domain, target_domain):
    return load_data_problem(
            *get_paths(ROBOTS_FILES, source_domain, target_domain)
    )


def load_url(source_domain, target_domain):
    return load_data_problem(
            *get_paths(URL_FILES, source_domain, target_domain)
    )


def load_android(source_domain, target_domain):
    return load_data_problem(
            *get_paths(ANDROID_FILES, source_domain, target_domain)
    )


def load_nslkdd(source_domain, target_domain):
    return load_data_problem(
            *get_paths(NSLKDD_FILES, source_domain, target_domain)
    )


def load_caltech(domain1, domain2):
    '''
    Carga dos dominios del dataset office-caltech.

    Argumentos
    ------------------------------------------
    domain1: string
        Nombre del primer dominio. Debe estar registrado en
        OFFICE_CALTECH_DOMAINS.
    domain2: string
        Nombre del segundo dominio. Debe estar registrado en
        OFFICE_CALTECH_DOMAINS.

    Retorno
    ------------------------------
    first_domain: tuple
        Tupla que contiene los siguientes elementos:
            first_data: numpy.ndarray shape (n, k - 1). Dominio uno con la
                columna target eliminada.
            first_target: numpy.ndarray shape (n, ). Columna target del
                primer dominio.
    second_domain: tuple
        Tupla que contiene los siguientes elementos:
            second_data: numpy.ndarray shape (n, k - 1). Dominio uno con la
                columna target eliminada.
            second_target: numpy.ndarray shape (n, ). Columna target del
                primer dominio.
    '''
    return load(OFFICE_CALTECH_FILES[domain1], OFFICE_CALTECH_FILES[domain2])


def load_imagenet(domain1, domain2):
    '''
    Carga dos dominios del dataset imagenet.

    Argumentos
    ------------------------------------------
    domain1: string
        Nombre del primer dominio. Debe estar registrado en
        IMAGENET_DOMAINS.
    domain2: string
        Nombre del segundo dominio. Debe estar registrado en
        IMAGENET_DOMAINS.

    Retorno
    ------------------------------
    first_domain: tuple
        Tupla que contiene los siguientes elementos:
            first_data: numpy.ndarray shape (n, k - 1). Dominio uno con la
                columna target eliminada.
            first_target: numpy.ndarray shape (n, ). Columna target del
                primer dominio.
    second_domain: tuple
        Tupla que contiene los siguientes elementos:
            second_data: numpy.ndarray shape (n, k - 1). Dominio uno con la
                columna target eliminada.
            second_target: numpy.ndarray shape (n, ). Columna target del
                primer dominio.
    '''
    return load(IMAGENET_FILES[domain1], IMAGENET_FILES[domain2])


# DATASETS = {
#    'caltech': {'load_func': load_caltech, 'domains': OFFICE_CALTECH_DOMAINS},
#    'imagenet': {'load_func': load_imagenet, 'domains': IMAGENET_DOMAINS}
# }

DATASETS = {
        'url': {
                'load_func': load_url, 'domains': URL_DOMAINS,
                'majority': URL_MAJORITY
        },
        'robots': {
                'load_func': load_robots, 'domains': ROBOTS_DOMAINS,
                'majority': ROBOTS_MAJORITY
        },
        'android': {
                'load_func': load_android, 'domains': ANDROID_DOMAINS,
                'majority': ANDROID_MAJORITY
        },
        'nslkdd': {
                'load_func': load_nslkdd, 'domains': NSLKDD_DOMAINS,
                'majority': NSLKDD_MAJORITY
        }
}
