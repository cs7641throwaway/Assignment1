from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import graphviz
from joblib import dump, load
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import numpy as np
import plot
import data_funcs

# Only for best fits (if test)
restore = False
save = True

learning_curve = False
validation_curve = False
test = True

run_fmnist = True
run_chess = True
data_prop = 1.0
test_prop = 0.2


fmnist_trgX, fmnist_tstX, fmnist_trgY, fmnist_tstY =  data_funcs.get_data('fmnist', data_prop, test_prop)
chess_trgX, chess_tstX, chess_trgY, chess_tstY = data_funcs.get_data('chess', data_prop, test_prop)

# Note:
# With no max depth, get perfect fit on training but validation doesn't look any better than regular DT
# With max depth =1 (default base_estimator) results are trash
# 10 looks pretty good
if learning_curve:
    # FMNIST max depth=20 (10 close but 87.5 vs. 85) and n_estimators=200
    # Chess is 5 max depth 200 estimators with random split
    if run_fmnist:
        plot.plot_learning_curve(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=20, splitter='random'), n_estimators=200),
                                "FMNIST_Boosting_ADA_SAMME.R_learning", fmnist_trgX, fmnist_trgY, cv=5, n_jobs=-1)
    if run_chess:
        plot.plot_learning_curve(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5, splitter='random'), n_estimators=200),
                                 "chess_Boosting_ADA_SAMME.R_learning", chess_trgX, chess_trgY, cv=5, n_jobs=-1)


if validation_curve:
    if run_fmnist:
        parameters = {'base_estimator__max_depth': [1, 5, 10, 20, 50], 'n_estimators': [10, 20, 50, 100, 200, 500]}
        plot.perform_grid_search(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(splitter='random')), type="boosting", dataset="FMNIST", params=parameters, trg_X=fmnist_trgX, trg_Y=fmnist_trgY, tst_X=fmnist_tstX, tst_Y=fmnist_tstY, cv=5)
    if run_chess:
        parameters = {'base_estimator__max_depth': [1, 5, 10, 20, 50], 'base_estimator__splitter': ('best', 'random'),
                      'n_estimators': [10, 20, 50, 100, 200, 500]}
        plot.perform_grid_search(AdaBoostClassifier(base_estimator=DecisionTreeClassifier()), type="boosting", dataset="chess", params=parameters, trg_X=chess_trgX, trg_Y=chess_trgY, tst_X=chess_tstX, tst_Y=chess_tstY, cv=5)

if test:
    if run_fmnist:
        clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=20, splitter='random'), n_estimators=200)
        if restore:
            clf = load('boosting_fmnist.joblib')
        else:
            clf = clf.fit(fmnist_trgX, fmnist_trgY)
        test_score = clf.score(fmnist_tstX, fmnist_tstY)
        data_funcs.write_best_results('boosting', 'fmnist', test_score, clf, save)
    if run_chess:
        clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5, splitter='random'), n_estimators=200)
        if restore:
            clf = load('boosting_chess.joblib')
        else:
            clf = clf.fit(chess_trgX, chess_trgY)
        test_score = clf.score(chess_tstX, chess_tstY)
        data_funcs.write_best_results('boosting', 'chess', test_score, clf, save)
