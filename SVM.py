from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import SVC
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
restore = False
save = True

learning_curve = False
validation_curve = False
iteration_curve = True
test = False

run_fmnist = True
run_chess = False
data_prop = 1.0
test_prop = 0.2

fmnist_trgX, fmnist_tstX, fmnist_trgY, fmnist_tstY =  data_funcs.get_data('fmnist', data_prop, test_prop)
chess_trgX, chess_tstX, chess_trgY, chess_tstY = data_funcs.get_data('chess', data_prop, test_prop)

# Note: 100 seconds w/o early stopping, 18 w/ early stopping
if learning_curve:
    if run_fmnist:
        # FMNIST
        # alpha = 0.005 with linear (82% and 81.5% ish)
        # C = 5 w/ RBF (~96% and 90%)
        # 2 hours runtime laptop
        plot.plot_learning_curve(SVC(gamma='scale', C=5), "fmnist_SVM_RBF_scale", fmnist_trgX, fmnist_trgY, cv=5, n_jobs=-1)
    if run_chess:
        # Chess
        # alpha = 0.001 with linear (98% and 97%)
        # C = 5 w/ RBF (99.8% and 99.5% ish)
        # 2 seconds laptop
        plot.plot_learning_curve(SVC(gamma='scale', C=5), "chess_SVM_RBF_scale", chess_trgX, chess_trgY, cv=5, n_jobs=-1)

if iteration_curve:
    if run_chess:
        parameters = {'max_iter':[20, 40, 60, 80, 100, 120, 140, 160, 180, 200]}
        plot.perform_grid_search(SVC(gamma='scale', C=5), "SVM_RBF_iteration_curve", "chess", parameters, chess_trgX, chess_trgY, chess_tstX, chess_tstY)
    if run_fmnist:
        parameters = {'max_iter':[200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
        plot.perform_grid_search(SVC(gamma='scale', C=5), "SVM_RBF_iteration_curve_2000", "FMNIST", parameters, fmnist_trgX, fmnist_trgY, fmnist_tstX, fmnist_tstY)

# Took 5 hours on desktop for all (1 hour for RBF, ~3.5 hours for linear)
if validation_curve:
    if run_fmnist:
        parameters = {'C':[0.05, .1, .5, 1, 5, 10]}
        plot.perform_grid_search(SVC(gamma='scale'), "SVM_RBF", "FMNIST", parameters, fmnist_trgX, fmnist_trgY, fmnist_tstX, fmnist_tstY)
        parameters = {'penalty':('l2', 'l1', 'elasticnet'), 'alpha':[10**-5, 10**-4, 10**-3, 5*10**-3, 10**-2]}
        plot.perform_grid_search(SGDClassifier(), "SVM_Linear", "FMNIST", parameters, fmnist_trgX, fmnist_trgY, fmnist_tstX, fmnist_tstY)
    if run_chess:
        parameters = {'C':[0.05, .1, .5, 1, 5, 10]}
        plot.perform_grid_search(SVC(gamma='scale'), "SVM_RBF", "chess", parameters, chess_trgX, chess_trgY, chess_tstX, chess_tstY)
        parameters = {'penalty':('l2', 'l1', 'elasticnet'), 'alpha':[10**-5, 10**-4, 10**-3, 5*10**-3, 10**-2]}
        plot.perform_grid_search(SGDClassifier(), "SVM_Linear", "chess", parameters, chess_trgX, chess_trgY, chess_tstX, chess_tstY)


if test:
    if run_fmnist:
        clf = SVC(gamma='scale', C=5)
        if restore:
            clf = load('SVM_fmnist.joblib')
        else:
            clf = clf.fit(fmnist_trgX, fmnist_trgY)
        test_score = clf.score(fmnist_tstX, fmnist_tstY)
        data_funcs.write_best_results('SVM', 'fmnist', test_score, clf, save)
    if run_chess:
        clf = SVC(gamma='scale', C=5)
        if restore:
            clf = load('SVM_chess.joblib')
        else:
            clf = clf.fit(chess_trgX, chess_trgY)
        test_score = clf.score(chess_tstX, chess_tstY)
        data_funcs.write_best_results('SVM', 'chess', test_score, clf, save)
