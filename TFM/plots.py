# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 09:23:40 2019

@author: David
"""
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os


def save_confusion_matrix_image(cm, savepath, classes=None,
                                normalize=False, title='Confusion matrix',
                                cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    # print(cm)
    if classes is None:
        # classes = np.arange(len(cm), dtype=np.int32)
        classes = ['normal', 'malicioso']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    cm = np.array(cm)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    # plt.tight_layout()
    # plt.show()
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()


def save_roc(fpr, tpr, savepath):
    plt.figure()
    # plt.title('Curva ROC')
    plt.xlabel("Ratio de Falsos Positivos")
    plt.ylabel("Ratio de Verdaderos Positivos")
    plt.plot(fpr, tpr, 'r--')
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()


def save_learning_curve(training_acc, crossval_acc, path):
    plt.figure()
    # plt.title(title)
    plt.ylim(top=1.01)
    train_sizes = np.linspace(.1, 1.0, 5)
    plt.xlabel("Porcentaje de ejemplos de entrenamiento")
    plt.ylabel("Porcentaje de acierto")
    # train_scores_mean = np.mean(training_acc, axis=1)
    # train_scores_std = np.std(training_acc, axis=1)
    # test_scores_mean = np.mean(crossval_acc, axis=1)
    # test_scores_std = np.std(crossval_acc, axis=1)
    plt.grid()

    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                 train_scores_mean + train_scores_std, alpha=0.1,
    #                 color="r")
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, training_acc, 'o-', color="r",
             label="Porcentaje de acierto en entrenamiento-testeo")
    plt.plot(train_sizes, crossval_acc, 'o-', color="g",
             label="Porcentaje en el Cross-validation")

    plt.legend(loc="best")
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_comparison_roc_curve(save_path, method_names, method_metric_values,
                              suffix=''):
    plt.figure()
    plt.title('Comparación de curva ROC')
    plt.xlabel("Ratio de Falsos Positivos")
    plt.ylabel("Ratio de Verdaderos Positivos")

    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                 train_scores_mean + train_scores_std, alpha=0.1,
    #                 color="r")
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
    for name, data in zip(method_names, method_metric_values):
        plt.plot(*data, label=name)

    plt.legend(loc="best")
    plt.savefig(
            os.path.join(save_path, ('comparison_roc_curve' + suffix)),
            bbox_inches='tight'
    )
    plt.close()


def plot_comparison_metrics(save_path, method_names, method_metric_values,
                            metric, suffix=''):
    plt.figure()
    metric = translate(metric)
    plt.title('Comparación de %s' % metric)
    plt.xlabel("Métodos")
    plt.ylabel(metric)

    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                 train_scores_mean + train_scores_std, alpha=0.1,
    #                 color="r")
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.bar(method_names, method_metric_values, align='center')

    plt.savefig(
            os.path.join(save_path, ('comparison' + metric + suffix)),
            bbox_inches='tight'
    )
    plt.close()


def translate(metric):
    if metric == 'accuracy':
        return 'tasa de acierto'
    elif metric == 'balanced_accuracy':
        return 'tasa de acierto balanceada'
    elif metric == 'time':
        return 'tiempo (s)'
    elif metric == 'max_memory':
        return 'memoria (MB)'
    else:
        return metric
