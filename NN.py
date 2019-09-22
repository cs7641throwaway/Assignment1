from sklearn.neural_network import MLPClassifier
import pandas as pd
import graphviz
from joblib import dump, load
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import numpy as np
import plot
import data_funcs

restore = False
save = True

learning_curve = False
validation_curve = False
iteration_curve = False
test = True

run_fmnist = False
run_chess = True

data_prop = 1.0
test_prop = 0.2

fmnist_trgX, fmnist_tstX, fmnist_trgY, fmnist_tstY =  data_funcs.get_data('fmnist', data_prop, test_prop)
chess_trgX, chess_tstX, chess_trgY, chess_tstY = data_funcs.get_data('chess', data_prop, test_prop)


if learning_curve:
    if run_fmnist:
        # RELU
        # 50 layers
        # 0.002 tolerance
        plot.plot_learning_curve(MLPClassifier(hidden_layer_sizes=(50,), activation='relu', tol=0.002), "fmnist_MLP_relu_learning", fmnist_trgX, fmnist_trgY, cv=5, n_jobs=-1)
    if run_chess:
        # RELU
        # (50, 10)
        plot.plot_learning_curve(MLPClassifier(max_iter=1000, hidden_layer_sizes=(50,10), activation='relu'), "chess_MLP_relu_learning", chess_trgX, chess_trgY, cv=5, n_jobs=-1)

if iteration_curve:
    if run_chess:
        parameters = {'max_iter':[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
        plot.perform_grid_search(MLPClassifier(hidden_layer_sizes=(50,10), activation='relu'), "NN_iteration_curve", "chess", parameters, chess_trgX, chess_trgY, chess_tstX, chess_tstY)
    if run_fmnist:
        parameters = {'max_iter':[20, 40, 60, 80, 100, 120, 140, 160, 180, 200]}
        plot.perform_grid_search(MLPClassifier(hidden_layer_sizes=(50,), activation='relu'), "NN_iteration_curve", "FMNIST", parameters, fmnist_trgX, fmnist_trgY, fmnist_tstX, fmnist_tstY)


if validation_curve:
    if run_fmnist:
        parameters = {'hidden_layer_sizes':[(10,), (50,), (100,), (200,), (500,)], 'tol':[10**-6, 10**-5, 10**-4, 10**-3, 10**-2]}
        plot.perform_grid_search(MLPClassifier(activation='relu'), "NN", "FMNIST", parameters, fmnist_trgX, fmnist_trgY, fmnist_tstX, fmnist_tstY)
    if run_chess:
        parameters = {'hidden_layer_sizes':[(10,), (50,), (100,), (10,10), (50,10), (100,10)], 'tol':[10**-6, 10**-5, 10**-4, 10**-3, 10**-2]}
        plot.perform_grid_search(MLPClassifier(), "NN", "chess", parameters, chess_trgX, chess_trgY, chess_tstX, chess_tstY)


if test:
    if run_fmnist:
        clf = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', tol=0.002)
        if restore:
            clf = load('NN_fmnist.joblib')
        else:
            clf = clf.fit(fmnist_trgX, fmnist_trgY)
        test_score = clf.score(fmnist_tstX, fmnist_tstY)
        data_funcs.write_best_results('NN', 'fmnist', test_score, clf, save)
    if run_chess:
        clf = MLPClassifier(hidden_layer_sizes=(50,10), activation='relu')
        if restore:
            clf = load('NN_chess.joblib')
        else:
            clf = clf.fit(chess_trgX, chess_trgY)
        test_score = clf.score(chess_tstX, chess_tstY)
        data_funcs.write_best_results('NN', 'chess', test_score, clf, save)
