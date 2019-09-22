from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import graphviz
from joblib import dump, load
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import numpy as np
import plot
import time
import data_funcs

# Only for best fits (if test)
# Not useful for kNN since it's a lazy learner
restore = False
save = False

learning_curve = False
validation_curve = False
test = True

run_fmnist = False
run_chess = True

data_prop = 1.0
test_prop = 0.2

fmnist_trgX, fmnist_tstX, fmnist_trgY, fmnist_tstY =  data_funcs.get_data('fmnist', data_prop, test_prop)
chess_trgX, chess_tstX, chess_trgY, chess_tstY = data_funcs.get_data('chess', data_prop, test_prop)

# Note: 100 seconds w/o early stopping, 18 w/ early stopping
# Takes forever to do this with full set; 5 hours, 2 hours, 6 hours or something like that
if learning_curve:
    if run_fmnist:
        # n_neighbors=5 weight = uniform
        plot.plot_learning_curve(KNeighborsClassifier(n_neighbors=5, weights='uniform'), "fmnist_kNN_learning",
                                 fmnist_trgX, fmnist_trgY, cv=5, n_jobs=-1)
    if run_chess:
        # n_neighbors=10 weight = uniform
        plot.plot_learning_curve(KNeighborsClassifier(n_neighbors=10, weights='uniform'), "chess_kNN_learning",
                                 chess_trgX, chess_trgY, cv=5, n_jobs=-1)


if validation_curve:
    if run_fmnist:
        parameters = {'n_neighbors': [1, 5, 10, 20, 50], 'weights': ('uniform', 'distance')}
        plot.perform_grid_search(KNeighborsClassifier(), type="kNN", dataset="FMNIST", params=parameters,
                                 trg_X=fmnist_trgX, trg_Y=fmnist_trgY, tst_X=fmnist_tstX, tst_Y=fmnist_tstY, cv=5)
    if run_chess:
        parameters = {'n_neighbors': [1, 5, 10, 20, 50], 'weights': ('uniform', 'distance')}
        plot.perform_grid_search(KNeighborsClassifier(), type="kNN", dataset="chess", params=parameters,
                                 trg_X=chess_trgX, trg_Y=chess_trgY, tst_X=chess_tstX, tst_Y=chess_tstY, cv=5)


if test:
    if run_fmnist:
        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        if restore:
            clf = load('kNN_fmnist.joblib')
        else:
            clf = clf.fit(fmnist_trgX, fmnist_trgY)
        test_score = clf.score(fmnist_tstX, fmnist_tstY)
        data_funcs.write_best_results('kNN', 'fmnist', test_score, clf, save)
    if run_chess:
        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        if restore:
            clf = load('kNN_chess.joblib')
        else:
            clf = clf.fit(chess_trgX, chess_trgY)
        test_score = clf.score(chess_tstX, chess_tstY)
        data_funcs.write_best_results('kNN', 'chess', test_score, clf, save)

