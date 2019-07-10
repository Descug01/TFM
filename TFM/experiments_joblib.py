# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:25:22 2019

@author: David
"""
from datasets import DATASETS

import numpy as np
from methods.TCA import TCA, SSTCA
from methods.TIT import TIT
from methods.tradaboost import RareTrAdaBoost
from methods.DTS import DTS
from methods.CORAL import CORAL
import sklearn
import sklearn.ensemble
import sklearn.svm
import sklearn.preprocessing
import sklearn.multiclass
from methods.DAE import DAE
from methods.optimaltransport import OptimalTransport
import itertools
import optunity
import sklearn.tree
import sklearn.neighbors
import sklearn.svm
import sklearn.ensemble
import sklearn.neural_network
import functools
from methods.util import data
import datasets
import time
import memory_profiler
import os
import gc
from joblib import Parallel, delayed
import sklearn.model_selection
import json
import plots
import pandas as pd
import copy

RESULT_DIRECTORY = 'resultados'

METHODS = {
    'TCA': TCA,
    'SSTCA': SSTCA,
    'DTS': DTS,
    'CORAL': CORAL,
    'DAE': DAE,
    'opt': OptimalTransport,
    'ada': RareTrAdaBoost,
    'TIT': TIT,
    'no_transfer_train_source': lambda x, y: (x, y),
    'no_transfer_train_target': data.train_test_split,
    'no_transfer_mixed': functools.partial(
            data.train_test_split, test_fraction=0.3
    )
}

SEARCH_SPACES_METHODS = {
    'TCA': {
            'num_features': [3, 10],
            'kernel': {
                'linear': None,
                'polynomial': {
                    'exponent': [2, 10], 'coefficient': [1, 100]
                },
                'rbf': {'sigma': [0.01, 10]}
            },
            'mu': [1e-3, 1e3]
    },
    'SSTCA': {
            'num_features': [3, 10],
            'kernel': {
                'linear': None,
                'polynomial': {
                    'exponent': [2, 10], 'coefficient': [1, 100]
                },
                'rbf': {'sigma': [0.01, 10]}
            },
            'mu': [1e-3, 1e3],
            'n_neighbors': [3, 20],
            'gamma': [1e-5, 1e5]
    },
    'DTS': {
            'alpha': [1e-5, 1e5],
            'beta': [1e-5, 1e5],
            'labda': [1e-5, 1e5],
            'max_iter': [5, 20]
    },
    'CORAL': {
            'labda': [1, 1000]
    },
    'DAE': {
            'num_features': [10, 50],
            'alpha': [0, 10],
            'beta': [0, 10],
            'gamma': [0, 10],
            'num_epochs': [100, 2500]
    },
    'opt': {
            'num_iters_gradient': [100, 1000],
            'num_iters_sinkhorn': [100, 1000],
            'labda': [0.001, 1000],
            'eta': [0.001, 1000],
            'convergence_threshold': [1e-9, 1e-4],
            'regularizer': {'group sparse': None, 'laplacian': None},
            'alpha': [0, 1],
            'n_neighbors': [3, 20]
    },
    'ada': {
            'num_iter': [10, 200]
    },
    'TIT': {
            'num_features': [3, 10],
            'n_neighbors': [3, 20],
            'labda': [0.001, 10],
            'beta': [0.001, 10],
            'gamma': [0.001, 10],
            'max_iters': [2, 20]
    }
}


SOURCE_TARGET = ['TCA', 'CORAL']
SOURCE_TARGET_SLABELS = ['DTS', 'TIT', 'DAE', 'opt', 'SSTCA']
SOURCE_TARGET_SLABELS_TLABELS = ['ada']
INT_ARGS = [
        'num_features', 'max_iters', 'num_iter_gradient', 'num_iters_sinkhorn',
        'n_neighbors', 'num_iter', 'num_epochs', 'n_jobs', 'n_estimators'
]

CLASSIFIERS = {
        'knn': functools.partial(
                sklearn.neighbors.KNeighborsClassifier, n_neighbors=11),
        # 'lr': functools.partial(
        #        sklearn.linear_model.LogisticRegression, solver='lbfgs'
        #        ),
        # 'dt': sklearn.tree.DecisionTreeClassifier,
        'rf': functools.partial(
                sklearn.ensemble.RandomForestClassifier, n_estimators=100
                ),
        'mlp': functools.partial(
                sklearn.neural_network.MLPClassifier,
                hidden_layer_sizes=(100, 50), learning_rate='adaptive',
                max_iter=1500, n_iter_no_change=30
               )
}

SEARCH_SPACES_CLASSIFIERS = {
    'knn': {
        'n_neighbors': [1, 100],
        'weights': {'uniform': None, 'distance': None},
        'p': [1, 10]
    },
    'lr': {
        'n_jobs': [1, 34]
    },
    'dt': {
        'criterion': {'gini': None, 'entropy': None},
    },
    'rf': {
        'n_estimators': [10, 100],
        'criterion': {'gini': None, 'entropy': None}
    },
    'mlp': {
        'activation': {
            'identity': None, 'logistic': None, 'tanh': None, 'relu': None
        },
        'alpha': [0.0001, 1],
        'max_iter': [1000, 5000]
    }
}


def adjust_num_features(max_features):
    for key, dictionary in SEARCH_SPACES_METHODS.items():
        if 'num_features' in dictionary:
            dictionary['num_features'] = [5, int(max_features / 2)]


def do_method(source, target, source_labels, target_labels, method_name,
              classifier=None, optimal_params_method={}, profile=False):
    clase = METHODS[method_name]
    target_test_labels = np.array([])
    if method_name == 'ada':
        target_train, target_train_labels, target_test, target_test_labels = (
                data.structured_split(target, target_labels)
        )

        def func():
            return clase(**optimal_params_method).fit_predict(
                    source, target_train, source_labels, target_train_labels,
                    target_test
            ), target_test_labels
    elif method_name == 'TIT':
        def func():
            return clase(**optimal_params_method).fit_transform(
                    source, target, source_labels, classifier()
            )[:2]
        # func = functools.partial(
        #        clase(**optimal_params_method).fit_transform, source, target,
        #        source_labels, classifier()
        # )
    elif method_name in SOURCE_TARGET_SLABELS:
        func = functools.partial(
                clase(**optimal_params_method).fit_transform, source, target,
                source_labels
        )
    elif method_name in SOURCE_TARGET:
        func = functools.partial(
                clase(**optimal_params_method).fit_transform, source, target
        )
    elif method_name == 'no_transfer_train_source':
        func = functools.partial(clase, source, target)
    elif method_name == 'no_transfer_train_target':
        func = functools.partial(clase, target, target_labels)
    elif method_name == 'no_transfer_mixed':
        func = functools.partial(
                clase, np.vstack((source, target)),
                np.concatenate((source_labels, target_labels))
        )

    if profile:
        func = profile_function(func)

    return func()


def compute_optimal_parameters(source, target, source_labels, target_labels,
                               method_name, classifier, num_evals=50):
    if 'no_transfer' in method_name:
        return {}

    def to_maximize(**kwargs):
        kwargs = process_space_dict(kwargs, INT_ARGS)
        returned = do_method(source, target, source_labels, target_labels,
                             method_name, classifier, kwargs)
        if method_name == 'ada':
            return sklearn.metrics.balanced_accuracy_score(*reversed(returned))
        pred = classifier().fit(returned[0], source_labels).predict(
                returned[1]
        )
        # See https://stackoverflow.com/questions/25652663/scipy-sparse-eigensolver-memoryerror-after-multiple-passes-through-loop-without
        gc.collect()
        return sklearn.metrics.balanced_accuracy_score(target_labels, pred)

    space = SEARCH_SPACES_METHODS[method_name]
    optimal_params, _, _ = optunity.maximize_structured(
            to_maximize, space, num_evals=num_evals
    )
    optimal_params = process_space_dict(optimal_params, INT_ARGS)
    return optimal_params


def time_function(func):
    def timed(*args, **kwargs):
        initial = time.perf_counter()
        returned = func(*args, **kwargs)
        total = time.perf_counter() - initial
        return returned, total
    return timed


def profile_function(func):
    '''
    Ejecuta una función y devuelve el máximo consumo de memoria, así como el
    tiempo de ejecución además del valor de retorno de la propia función.
    '''
    timed = time_function(func)

    def profiled(*args, **kwargs):
        max_memory, returned = memory_profiler.memory_usage(
            proc=(timed, args, kwargs), max_usage=True, retval=True
        )
        return returned, max_memory[0]

    return profiled


def create_subdirs(base, subdirs):
    created = []
    for subdir in subdirs:
        created.append(os.path.join(base, subdir))
        os.mkdir(created[-1])
    return created


def create_base_directory(dataset_dict, classifier_dict, method_dict,
                          train_majority=True):
    try:
        os.mkdir(RESULT_DIRECTORY)
    except FileExistsError:
        pass
    write_dir = os.path.join(
            RESULT_DIRECTORY, time.strftime(r'%Y_%m_%d__%H_%M_%S')
    )
    os.mkdir(write_dir)

    for dataset, dictionary in dataset_dict.items():
        dataset_dir = os.path.join(write_dir, dataset)
        os.mkdir(dataset_dir)
        if train_majority:
            domain_pairs = itertools.product(
                [dictionary['majority']], dictionary['domains']
                - {'normal', dictionary['majority']}
            )
        else:
            domain_pairs = itertools.product(
                dictionary['domains'] - {'normal', dictionary['majority']},
                [dictionary['majority']]
            )
        for pair in domain_pairs:
            domain_dir = os.path.join(dataset_dir, '_'.join(pair))
            os.mkdir(domain_dir)
            created = create_subdirs(domain_dir, classifier_dict)
            for new in created:
                create_subdirs(new, method_dict)

    return write_dir


def convert_to_int(dictionary, convert_list):
    '''
    Optunity solo soporta floats para pasar argumentos a las funciones a
    optimizar y hay que convertirlos a int en algunos casos. Con esta función
    hago esto, teniendo en cuenta que no todos los método tienen las mismas
    keys.

    Argumentos
    -------------------------
    dictionary: dict
    convert_list: list<String>
        Nombres de las claves de dictionary cuyos valores se van a convertir
        a int.
    '''
    for key in convert_list:
        try:
            dictionary[key] = int(dictionary[key])
        except KeyError:
            pass


def delete_none(dictionary):
    return dict(filter(lambda x: x[-1] is not None, dictionary.items()))


def process_space_dict(dictionary, convert_list):
    convert_to_int(dictionary, convert_list)
    return delete_none(dictionary)


def construct_iterable(dataset_dict, classifier_dict, method_dict,
                       train_majority=True):
    list_dataset = []
    for key, dictionary in dataset_dict.items():
        dataset_pair = [(key, dictionary['load_func'])]
        if train_majority:
            domain_pairs = itertools.product(
                [dictionary['majority']],
                dictionary['domains'] - {'normal', dictionary['majority']}
            )
        else:
            domain_pairs = itertools.product(
                dictionary['domains'] - {'normal', dictionary['majority']},
                [dictionary['majority']]
            )
        dataset_iter = itertools.product(dataset_pair, domain_pairs)
        list_dataset = itertools.chain(list_dataset, dataset_iter)

    return (
        list_dataset,
        itertools.product(classifier_dict, reversed(list(method_dict.keys())))
    )


def execute_mixed(source, source_labels, target, target_labels, dataset, pair,
                  classifier, method, write_dir, optimal):
    if (method == 'no_transfer_mixed'):
        (train_data, train_labels), (test_data, test_labels) = do_method(
                source, target, source_labels, target_labels, method,
                classifier, optimal_params_method=optimal, profile=False
        )
    elif method == 'ada':
        first, second = do_method(
            source, target, source_labels, target_labels, method,
            classifier, optimal_params_method=optimal, profile=False
        )
        return second, first
    else:
        first, second = do_method(
            source, target, source_labels, target_labels, method,
            classifier, optimal_params_method=optimal, profile=False
        )
        (train_data, train_labels), (test_data, test_labels) = (
            data.train_test_split(
                np.vstack((first, second)),
                np.concatenate((source_labels, target_labels)),
                test_fraction=0.3
            )
        )
    pred = classifier().fit(train_data, train_labels).predict(test_data)
    true = test_labels
    return true, pred


def execute(source, source_labels, target, target_labels, dataset, pair,
            classifier_name, method, write_dir, classifier_dict, num_evals=50):
    classifier = classifier_dict[classifier_name]
    print(dataset, pair, classifier_name, method)

    predicted, *target_true, time, memory, optimal = trial(
                source, target, source_labels, target_labels, classifier,
                method, num_evals=num_evals
    )

    target_true = target_true[0] if len(target_true) > 0 else target_labels
    compute_and_write(
                write_dir, source, target, source_labels, target_labels,
                dataset, pair, classifier_name, method, target_true, predicted,
                time, memory, optimal
    )

    if method == 'no_transfer_mixed' or 'no_transfer' not in method:
        true, pred = execute_mixed(
                source, source_labels, target, target_labels, dataset, pair,
                classifier, method, write_dir, optimal
        )
        compute_and_write(
                write_dir, source, target, source_labels, target_labels,
                dataset, pair, classifier_name, method, true,
                pred, time, memory, optimal, train_source=False
        )


def experiment(dataset_dict=None, classifier_dict=None, method_dict=None,
               train_majority=True, num_evals=50):
    dataset_dict = dataset_dict or datasets.DATASETS
    classifier_dict = classifier_dict or CLASSIFIERS
    method_dict = method_dict or METHODS
    write_dir = create_base_directory(
            dataset_dict, classifier_dict, method_dict,
            train_majority=train_majority
    )
    data_params, rest_params = construct_iterable(
            dataset_dict, classifier_dict, method_dict,
            train_majority=train_majority
    )
    rest_params = list(rest_params)
    for ((dataset, load_func), pair) in data_params:
        (source, source_labels), (target, target_labels) = load_func(*pair)
        source, target = data.normalize(source, target)
        adjust_num_features(source.shape[1])
        Parallel(n_jobs=-1, max_nbytes=1000, prefer='processes')(
            delayed(execute)(
                source, source_labels, target, target_labels, dataset, pair,
                classifier_name, method, write_dir, classifier_dict,
                num_evals=num_evals
            ) for classifier_name, method in rest_params
        )
    return write_dir


def trial(source, target, source_labels, target_labels, classifier,
          method_name, return_classifier=False, num_evals=50):
    optimal = compute_optimal_parameters(
            source, target, source_labels, target_labels, method_name,
            classifier, num_evals=num_evals
    )
    (((first, second), time), memory) = do_method(
            source, target, source_labels, target_labels, method_name,
            classifier, optimal_params_method=optimal, profile=True
    )
    if method_name == 'ada':
        classifier = classifier()
        tup = first, second, time, memory, optimal
    elif (method_name == 'no_transfer_train_target'
          or method_name == 'no_transfer_mixed'):
        classifier = classifier().fit(*first)
        pred = classifier.predict(second[0])
        tup = pred, second[1], time, memory, optimal
    else:
        classifier = classifier().fit(first, source_labels)
        pred = classifier.predict(second)
        tup = pred, time, memory, optimal

    if return_classifier:
        tup = tup + (classifier, )
    return tup


def generate_comparisons(method_dir, methods, metrics, tables=True,
                         images=True, json_file='results.json', suffix='',
                         substitute=True):
    method_metrics = extract_metrics_dir(
            method_dir, methods, metrics, json_file, substitute
    )
    if method_metrics:
        if tables:
            generate_tables(method_dir, method_metrics, suffix)
        if images:
            generate_comparison_images(method_dir, method_metrics, suffix)


def generate_metric_comparisons(directory, methods=None, metrics=None,
                                tables=True, images=True, substitute=True):
    methods = methods or list(METHODS.keys())
    metrics = metrics or ['accuracy', 'balanced_accuracy', 'precision',
                          'recall', 'time', 'max_memory', 'auc', 'roc curve']
    methods_directories = get_methods_directories(directory)
    for method_dir in methods_directories:
        generate_comparisons(
                method_dir, methods, metrics, tables, images, 'results.json',
                substitute=substitute
        )
        generate_comparisons(
                method_dir, methods, metrics, tables, images, 'results2.json',
                suffix='2', substitute=substitute
        )


def generate_comparison_images(method_dir, method_metrics, suffix=''):
    image_save_path = os.path.join(method_dir, 'images')
    try:
        os.mkdir(image_save_path)
    except FileExistsError:
        pass
    for metric in method_metrics[list(method_metrics.keys())[0]]:
        method_names, method_metric_values = zip(
                *[(method, d[metric]) for method, d in method_metrics.items()]
        )
        method_names = shorten_method_names(method_names)
        if metric == 'roc curve':
            plots.plot_comparison_roc_curve(
                    image_save_path, method_names, method_metric_values,
                    suffix
            )
        else:
            plots.plot_comparison_metrics(
                image_save_path, method_names, method_metric_values, metric,
                suffix
            )


def shorten_method_names(method_names):
    def substitute(name):
        if name == 'no_transfer_train_source':
            return 'ntts'
        elif name == 'no_transfer_train_target':
            return 'nttt'
        elif name == 'no_transfer_mixed':
            return 'ntm'
        elif name == 'ada':
            return 'ada'
        else:
            return name
    return [substitute(name) for name in method_names]


def generate_tables(method_dir, method_metrics, suffix=''):
    metrics = set(method_metrics[list(method_metrics.keys())[0]])
    if 'roc curve' in metrics:
        metrics.remove('roc curve')
        method_metrics = {
                method: filter_dict_keys(method_metrics[method], metrics)
                for method in method_metrics
        }
    df = pd.DataFrame(method_metrics).T
    try:
        df = df[['accuracy', 'auc', 'precision', 'recall', 'time', 'max_memory']]
    except KeyError:
        print(method_dir)
    #df.to_csv(
    #        os.path.join(method_dir, ('table' + suffix + '.csv')),
    #        sep=',', index=True, header=True
    #)
    string = df.to_csv(
            None,
            sep=',', index=True, header=True, float_format='%.3f',
            line_terminator='\\\\ \n'
    )
    string = string.replace(',', ' & ')
    with open(os.path.join(method_dir, ('table' + suffix + '.csv')), 'w') as f:
        f.write(string)


def extract_metrics_dir(method_dir, methods, metrics,
                        json_file='results.json', substitute=True):
    dictionary = {}
    for method in methods:
        filtered = extract_filtered_json(
                os.path.join(method_dir, method, json_file), metrics,
                substitute
        )
        if filtered:
            dictionary[method] = filtered

    return dictionary


def extract_filtered_json(json_file, metrics, substitute=True):
    data = {}
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        if substitute and (os.path.basename(json_file) != 'results.json'):
            results_path = os.path.join(
                os.path.dirname(json_file), 'results.json'
            )
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    data = json.load(f)
    finally:
        return filter_dict_keys(data, metrics)


def filter_dict_keys(dictionary, selected_keys):
    return dict(filter(lambda x: x[0] in selected_keys, dictionary.items()))


def get_methods_directories(directory):
    return set(
            map(lambda x: os.path.dirname(x), list_json(directory))
    )


def generate_images(directory):
    jsons = list_json(directory)

    for directory in jsons:
        images_from_json(directory, 'results.json')
        images_from_json(directory, 'results2.json', suffix='2')


def images_from_json(directory, json_name, suffix=''):
    try:
        with open(os.path.join(directory, json_name)) as f:
            dictionary = json.load(f)
            plots.save_confusion_matrix_image(
                dictionary['confusion matrix'],
                os.path.join(directory, 'cm' + suffix)
            )
            plots.save_roc(
                *dictionary['roc curve'],
                os.path.join(directory, 'roc' + suffix)
            )
            plots.save_learning_curve(
                *dictionary['learning_curve'],
                os.path.join(directory, 'lc' + suffix)
            )
    except FileNotFoundError:
        pass


def list_json(directory):
    a = os.walk(directory)
    b = map(
            lambda tup: map(
                lambda x: tup[0], filter(lambda x: '.json' in x, tup[-1])
            ),
            a
    )
    return itertools.chain(*b)


def learning_curve_accuracy(source, target, source_labels, target_labels,
                            classifier, method_name, optimal, fraction):
    source_frac, source_labels_frac = data.random_extract(
            source, source_labels, fraction
    )
    target_frac, target_labels_frac = data.random_extract(
            target, target_labels, fraction
    )
    first, second = do_method(
            source_frac, target_frac, source_labels_frac, target_labels_frac,
            method_name, classifier, optimal
    )
    if method_name == 'ada':
        pred, true = first, second
    elif (method_name == 'no_transfer_train_target'
          or method_name == 'no_transfer_mixed'):
        pred = classifier().fit(*first).predict(second[0])
        true = second[1]
    else:
        pred = classifier().fit(first, source_labels_frac).predict(second)
        true = target_labels_frac

    return sklearn.metrics.accuracy_score(true, pred)


def learning_curve(source, target, source_labels, target_labels,
                   classifier_name, method_name, optimal):
    classifier = CLASSIFIERS[classifier_name]
    train_sizes = np.linspace(.1, 1.0, 5)
    train_accuracies = []
    test_accuracies = []
    for fraction in train_sizes:
        train_accuracies.append(
                learning_curve_accuracy(
                        source, target, source_labels, target_labels,
                        classifier, method_name, optimal, fraction
                )
        )
        temp = []
        for i in range(10):
        # See https://stackoverflow.com/questions/25652663/scipy-sparse-eigensolver-memoryerror-after-multiple-passes-through-loop-without
            gc.collect()
            temp.append(
                    learning_curve_accuracy(
                        source, target, source_labels, target_labels,
                        classifier, method_name, optimal, fraction
                    )
            )
        test_accuracies.append(sum(temp) / len(temp))
    return [train_accuracies, test_accuracies]


def learning_curve2(source, target, source_labels, target_labels,
                    classifier_name, method_name, optimal):
    if method_name == 'ada':
        return ([], [])
    classifier = CLASSIFIERS[classifier_name]
    train_sizes = np.linspace(.1, 1.0, 5)
    cv = sklearn.model_selection.ShuffleSplit(n_splits=10, test_size=0.3)
    first, second = do_method(
            source, target, source_labels, target_labels,
            method_name, classifier, optimal
    )
    if (method_name == 'no_transfer_train_target'
            or method_name == 'no_transfer_mixed'):
        (xs, ls), (xt, lt) = first, second
        X = np.vstack((first[0], second[0]))
        y = np.concatenate((first[1], second[1]))
    else:
        xs, ls, xt, lt = first, source_labels, second, target_labels
    X = np.vstack((xs, xt))
    y = np.concatenate((ls, lt))

    train_sizes, train_scores, test_scores = (
        sklearn.model_selection.learning_curve(
            classifier(), X, y, cv=cv, n_jobs=None, train_sizes=train_sizes
        )
    )
    return [np.mean(train_scores, axis=1).tolist(),
            np.mean(test_scores, axis=1).tolist()]


def cross_validate_metrics(datos, labels, classifier, n=10, test_fraction=0.9):
    scoring = [
            'accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc'
    ]
    scores = sklearn.model_selection.cross_validate(
                classifier, datos, labels,
                scoring=scoring, cv=data.split_gen(
                        np.arange(datos.shape[0]), n,
                        test_fraction=test_fraction
                ),
                return_train_score=False
    )
    original_keys = set(scores.keys())
    for key in original_keys:
        scores[key.split('_', 1)[1]] = np.mean(scores[key])
        del scores[key]
    scores['auc'] = scores['roc_auc']
    del scores['roc_auc']
    del scores['time']
    return scores


def compute_metrics(source, target, source_labels, target_labels,
                    classifier_name, method_name, execution_time, max_memory,
                    optimal, true_labels, predicted_labels,
                    train_source=True):
    dictionary = {}
    dictionary['optimal'] = optimal
    if train_source:
        dictionary['time'] = execution_time
        dictionary['max_memory'] = max_memory
        dictionary['accuracy'] = sklearn.metrics.accuracy_score(
            true_labels, predicted_labels
        )
        dictionary['balanced_accuracy'] = (
            sklearn.metrics.balanced_accuracy_score(
                true_labels, predicted_labels
            )
        )
        summary_dict = sklearn.metrics.classification_report(
            true_labels, predicted_labels, output_dict=True
        )
        dictionary['recall'] = summary_dict['weighted avg']['recall']
        dictionary['precision'] = summary_dict['weighted avg']['precision']
        dictionary['auc'] = sklearn.metrics.roc_auc_score(
            true_labels, predicted_labels, average='weighted'
        )
        dictionary['learning_curve'] = learning_curve(
            source, target, source_labels, target_labels, classifier_name,
            method_name, optimal
        )
    else:
        if method_name == 'no_transfer_train_target':
            data = target
            labels = target_labels
            test_fraction = 0.9
        elif (method_name == 'no_transfer_mixed'
              or method_name != 'ada' or 'no_transfer' not in method_name):
            data = np.vstack((source, target))
            labels = np.concatenate((source_labels, target_labels))
            test_fraction = 0.3

        scores = cross_validate_metrics(
                data, labels, CLASSIFIERS[classifier_name](),
                test_fraction=test_fraction
        )
        dictionary.update(scores)
        dictionary['learning_curve'] = learning_curve2(
            source, target, source_labels, target_labels,
            classifier_name, method_name, optimal
        )

    dictionary['confusion matrix'] = sklearn.metrics.confusion_matrix(
            true_labels, predicted_labels
    ).tolist()
    dictionary['roc curve'] = [
        list(arr) for arr in sklearn.metrics.roc_curve(
                true_labels, predicted_labels, pos_label=2
        )[:-1]
    ]
    return dictionary


def compute_and_write(write_dir, source, target, source_labels, target_labels,
                      dataset, pair, classifier_name, method, true_labels,
                      predicted, time, memory, optimal, train_source=True):
    file = 'results.json' if train_source else 'results2.json'
    metric_dict = compute_metrics(
            source, target, source_labels, target_labels,
            classifier_name, method, time, memory, optimal, true_labels,
            predicted, train_source
    )
    write_results(
            write_dir, dataset, pair, classifier_name, method, metric_dict,
            file=file
    )


def write_results(write_dir, dataset, pair, classifier_name, method,
                  metric_dict, file='results.json'):
    results_file = os.path.join(
            write_dir, dataset, '_'.join(pair), classifier_name, method,
            file
    )
    with open(results_file, 'w') as f:
        json.dump(metric_dict, f)


def compute_and_write2(write_dir, source, target, source_labels, target_labels,
                       dataset, pair, classifier_name, method, true_labels,
                       predicted, time, memory, optimal, train_source=True,
                       num_evals=50):
    file = 'results%d_%d.json' % (1 if train_source else 2, num_evals)
    metric_dict = compute_metrics(
            source, target, source_labels, target_labels,
            classifier_name, method, time, memory, optimal, true_labels,
            predicted, train_source
    )
    write_results(
            write_dir, dataset, pair, classifier_name, method, metric_dict,
            file=file
    )


def execute2(source, source_labels, target, target_labels, dataset, pair,
             classifier_name, method, write_dir, classifier_dict, num_evals):
    classifier = classifier_dict[classifier_name]
    print(dataset, pair, classifier_name, method)

    predicted, *target_true, time, memory, optimal = trial(
                source, target, source_labels, target_labels, classifier,
                method, num_evals=num_evals
    )

    target_true = target_true[0] if len(target_true) > 0 else target_labels
    compute_and_write2(
                write_dir, source, target, source_labels, target_labels,
                dataset, pair, classifier_name, method, target_true, predicted,
                time, memory, optimal, num_evals=num_evals
    )

    if method == 'no_transfer_mixed' or 'no_transfer' not in method:
        true, pred = execute_mixed(
                source, source_labels, target, target_labels, dataset, pair,
                classifier, method, write_dir, optimal
        )
        compute_and_write2(
                write_dir, source, target, source_labels, target_labels,
                dataset, pair, classifier_name, method, true,
                pred, time, memory, optimal, train_source=False,
                num_evals=num_evals
        )


def optimization_experiment(dataset_dict=None, classifier_dict=None,
                            method_dict=None, train_majority=True):
    evals_list = [1, 10, 25, 50]
    dataset_dict = dataset_dict or datasets.DATASETS
    classifier_dict = classifier_dict or CLASSIFIERS
    method_dict = method_dict or METHODS
    write_dir = create_base_directory(
            dataset_dict, classifier_dict, method_dict,
            train_majority=train_majority
    )
    data_params, rest_params = construct_iterable(
            dataset_dict, classifier_dict, method_dict,
            train_majority=train_majority
    )
    rest_params = list(rest_params)
    rest_params = list(itertools.product(rest_params, evals_list))
    for ((dataset, load_func), pair) in data_params:
        (source, source_labels), (target, target_labels) = load_func(*pair)
        source, target = data.normalize(source, target)
        adjust_num_features(source.shape[1])
        Parallel(n_jobs=-1, max_nbytes=1000, prefer='processes')(
            delayed(execute2)(
                source, source_labels, target, target_labels, dataset, pair,
                classifier_name, method, write_dir, classifier_dict,
                num_evals=num_evals
            ) for (classifier_name, method), num_evals in rest_params
        )
    return write_dir


def extract_acc_num(subdir, base, numbers):
    li = []
    for number in numbers:
        f = os.path.join(subdir, base % number)
        try:
            with open(f) as json_file:
                d = json.load(json_file)
                li.append((number, d['accuracy'], d['optimal']))
        except FileNotFoundError:
            f = os.path.join(subdir, 'results1_%s.json' % number)
            with open(f) as json_file:
                d = json.load(json_file)
                li.append((number, d['accuracy'], d['optimal']))

    return li


def generate_arrays(subdir):
    # print(subdir)
    jsons = list(filter(lambda x: x.endswith('json'), next(os.walk(subdir))[-1]))

    numbers = set((int(x.split('.')[0].split('_')[-1]) for x in jsons))

    first = extract_acc_num(subdir, 'results1_%d.json', numbers)
    second = extract_acc_num(subdir, 'results2_%d.json', numbers)

    return first, second


def generate_image_optimization(directory, suffix, data):
    image_save_path = os.path.join(directory, 'images')
    try:
        os.mkdir(image_save_path)
    except FileExistsError:
        pass
    method_names = [d[0] for d in data]
    metric_values = list(zip(*[d[-1] for d in data]))
    method_names = shorten_method_names(method_names)
    plots.plot_comparison_roc_curve(
            image_save_path, method_names, metric_values, suffix
    )


def generate_images_optimization(directory):
    subdirs = next(os.walk(directory))[1]
    first = []
    second = []
    for subdir in subdirs:
        a, b = generate_arrays(os.path.join(directory, subdir))
        # method = subdir.split('\\')[-1]
        first.append((subdir, a))
        second.append((subdir, b))

    return first, second


def save(directory, data):
    index = [el[0] for el in data]
    for d in data:
        # print(d)
        for i in range(len(d[1])):
            # print(d[1])
            # print(index)
            d[1][i] = d[1][i][:2]
    data_dict = list(map(lambda x: dict(x[1]), data))
    df = pd.DataFrame(data_dict, index=index)
    string = df.to_csv(
            None,
            sep=',', index=True, header=True, float_format='%.3f',
            line_terminator='\\\\ \n'
    )
    string = string.replace(',', ' & ')
    with open(os.path.join(directory, 'opt.csv'), 'w') as f:
        f.write(string)
    # df.to_csv(os.path.join(directory, 'opt.csv'), index=True, header=True)

def save2(directory, data):
    for d in data:
        # print(d)
        method = d[0]
        data_dict = {}
        if method == 'images':
            continue
        # index = set(list(map(lambda x: x[1][0][-1], d)))
        index = list(functools.reduce(lambda x, y: x.union(y[-1]), d[1], set()))
        for i in range(len(d[1])):
            data_dict[d[1][i][0]] = d[1][i][-1]
        df = pd.DataFrame(data_dict, index=index)
        string = df.to_csv(
            None,
            sep=',', index=True, header=True, float_format='%.3f',
            line_terminator='\\\\ \n'
        )
        string = string.replace(',', ' & ')
        with open(os.path.join(directory, method, 'opti.csv'), 'w') as f:
            f.write(string)


def process_experiment_optimization(directory):
    directories = list_json(directory)
    classifier_level = set(map(lambda x: x.rpartition('\\')[0], directories))
    glob = []

    for directory in classifier_level:
        first, second = generate_images_optimization(directory)
        glob.append((directory, copy.deepcopy(second)))
        save2(directory, second)
        # save(directory, second)

    return dict(glob)





small_dataset = {}
# small_dataset['url'] = datasets.DATASETS['url']
# small_dataset['android'] = datasets.DATASETS['android']
#small_dataset['robots'] = DATASETS['robots']
small_dataset['android'] = DATASETS['android']

small_classifier = {
        'knn': CLASSIFIERS['knn']
}

small_methods = {
    'ada': METHODS['ada']

}


# small_methods = copy.deepcopy(METHODS)
# del small_methods['TIT']


if __name__ == '__main__':
    #write_dir = optimization_experiment(small_dataset, CLASSIFIERS,
    #                                    METHODS, train_majority=False)

     experiment(small_dataset, CLASSIFIERS, small_methods,
                train_majority=False, num_evals=50)
    #write_dir = experiment(small_dataset, small_classifier, small_methods,
    #                       train_majority=False, num_evals=1)
    #generate_images(write_dir)
    #generate_metric_comparisons(write_dir)


    # experiment4(datasets.DATASETS, CLASSIFIERS, METHODS)
    # experiment4(datasets.DATASETS, CLASSIFIERS, METHODS)
