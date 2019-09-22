from sklearn import tree
import pandas as pd
import graphviz
from joblib import dump, load
import sklearn.model_selection as ms
import plot
import data_funcs
import matplotlib.pyplot as plt
import numpy as np
import sys

# TODO: Pruning

plot_tree = False

restore_tree = False
save = True

learning_curve = False
validation_curve = False
test = True

run_fmnist = True
run_chess = False

data_prop = 1
test_prop = 0.2

fmnist_trgX, fmnist_tstX, fmnist_trgY, fmnist_tstY =  data_funcs.get_data('fmnist', data_prop, test_prop)
chess_trgX, chess_tstX, chess_trgY, chess_tstY = data_funcs.get_data('chess', data_prop, test_prop)

clf = tree.DecisionTreeClassifier()


# ms.validation_curve(tree.DecisionTreeClassifier(), fmnist_trgX, fmnist_trgY, )


if plot_tree:
    tree.plot_tree(clf.fit(fmnistX, fmnistY))
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("fmnist")

# Evaluate
# out = clf.predict(fmnist_tstX)

# Criterion = how to determine best attribute to split on
# splitter = how to choose the split value (best vs. random)
# max_depth = limits tree size (form of pruning by limiting growth)
# min_samples_split =

if learning_curve:
    if run_fmnist:
        # Best attrs: entropy, max depth = 20, min samples leaf = 5, min_impurity_decrease = 0.0005
        plot.plot_learning_curve(tree.DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_leaf=5,
                                                             min_impurity_decrease=0.0005),
                                 "FMNIST_DT_entropy_learning", fmnist_trgX, fmnist_trgY, cv=5, n_jobs=-1)
    if run_chess:
        # Best attrs: entropy, max depth = 100, min samples leaf = 1, min_impurity_decrease = 0.0005
        plot.plot_learning_curve(tree.DecisionTreeClassifier(criterion='entropy', max_depth=100, min_samples_leaf=1,
                                                             min_impurity_decrease = 0.0005),
                                 "chess_DT_entropy_learning", chess_trgX, chess_trgY, cv=5, n_jobs=-1)

if validation_curve:
    #parameters = {'criterion':('gini', 'entropy'), 'min_samples_leaf':[1,5,10,50]}
        # Best was 0.001 and gini
    parameters = {'criterion': ('gini', 'entropy'), 'min_impurity_decrease': [0, 0.0001, 0.0005, 0.001, 0.005, 0.01], 'max_depth':[5, 10, 20, 50, 100], 'min_samples_leaf':[1, 5, 10, 50]}
    #parameters = {'criterion': ('gini', 'entropy'), 'min_impurity_decrease': [0, 0.0001, 0.0005, 0.001, 0.005, 0.01], 'max_depth':[5,10,20,50,100]}
    if run_fmnist:
        plot.perform_grid_search(tree.DecisionTreeClassifier(), type="DT", dataset="FMNIST", params=parameters, trg_X=fmnist_trgX, trg_Y=fmnist_trgY, tst_X=fmnist_tstX, tst_Y=fmnist_tstY, cv=5)
    if run_chess:
        plot.perform_grid_search(tree.DecisionTreeClassifier(), type="DT", dataset="chess", params=parameters, trg_X=chess_trgX, trg_Y=chess_trgY, tst_X=chess_tstX, tst_Y=chess_tstY, cv=5)

if test:
    if run_fmnist:
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_leaf=5, min_impurity_decrease=0.0005)
        if restore_tree:
            # Restore
            clf = load('dt_fmnist.joblib')
        else:
            # Generate tree again
            clf = clf.fit(fmnist_trgX, fmnist_trgY)
        test_score = clf.score(fmnist_tstX, fmnist_tstY)
        data_funcs.write_best_results('DT', 'fmnist', test_score, clf, save)
    if run_chess:
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=100, min_samples_leaf=1, min_impurity_decrease=0.0005)
        if restore_tree:
            # Restore
            clf = load('dt_chess.joblib')
        else:
            # Generate tree again
            clf = clf.fit(chess_trgX, chess_trgY)
        test_score = clf.score(chess_tstX, chess_tstY)
        data_funcs.write_best_results('DT', 'chess', test_score, clf, save)


