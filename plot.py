#from sklearn import tree
import pandas as pd
import graphviz
from joblib import dump, load
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import numpy as np

# From sklearn tutorial

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = ms.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=1, shuffle = True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    #plt.show()
    plt.savefig(title+'.png')
    return plt


# Modified from jontay (also grabs train score)

def perform_grid_search(estimator, type, dataset, params, trg_X, trg_Y, tst_X, tst_Y, cv=5, n_jobs=-1, train_score=True):
    cv = ms.GridSearchCV(estimator, n_jobs=n_jobs, param_grid= params, refit=True, verbose=2, cv=cv, return_train_score=train_score)
    cv.fit(trg_X, trg_Y)
    test_score = cv.score(tst_X, tst_Y)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./results/{}_{}_reg.csv'.format(type,dataset),index=False)
    with open('./results/test_results.csv','a') as f:
        f.write('{},{},{},{}\n'.format(type,dataset,test_score,cv.best_params_))


def plot_distribution():
    data = pd.read_hdf('datasets_full.hdf', 'fmnist')
    d = data['Class'].value_counts()
    d.plot.bar()
    plt.ylabel('Frequency')
    plt.xlabel('Class')
    plt.title('FMNIST Class Distribution')
    plt.savefig('FMNIST_class_dist.png')


def plot_complexity(title, param_str, df, ylim=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(param_str)
    plt.ylabel("Score")
    plt.grid()

    # TODO: Need data in form of param, value, training mean, training std, validation mean, validation std
    param_values = df['param_'+param_str]
    train_scores_mean = df['mean_train_score']
    train_scores_std = df['std_train_score']
    test_scores_mean = df['mean_test_score']
    test_scores_std = df['std_test_score']


    plt.fill_between(param_values, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(param_values, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(param_values, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(param_values, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    #plt.show()
    plt.savefig(title+'.png')
    return plt


def get_model_complexity_data(file, param=None, param_value=None):
    df = pd.read_csv(file)
    if param is None:
        return df
    df2 = df[df['param_'+param] == param_value]
    # Open file
    # Read into df
    # Slice to get df
    # Format accordingly
    return df2

def plot_DT_complexity():
    file = 'results/DT_FMNIST_reg.csv'
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_max_depth']==20] ; # Slice off max depth
    df = df[df['param_min_samples_leaf']==5] ; # Slice off max depth
    plot_complexity("FMNIST_DT_entropy_max_depth=20_min_samples_leaf=5", 'min_impurity_decrease', df)
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_min_impurity_decrease']==0.0005] ; # Slice off max depth
    df = df[df['param_min_samples_leaf']==5] ; # Slice off max depth
    plot_complexity("FMNIST_DT_entropy_min_impurity_decr=0.0005_min_samples_leaf=5", 'max_depth', df)
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_min_impurity_decrease']==0.0005] ; # Slice off max depth
    df = df[df['param_max_depth']==20] ; # Slice off max depth
    plot_complexity("FMNIST_DT_entropy_min_impurity_decr=0.0005_max_depth=20", 'min_samples_leaf', df)
    file = 'results/DT_chess_reg.csv'
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_max_depth']==100] ; # Slice off max depth
    df = df[df['param_min_samples_leaf']==1] ; # Slice off max depth
    plot_complexity("chess_DT_entropy_max_depth=100_min_samples_leaf=1", 'min_impurity_decrease', df)
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_min_impurity_decrease']==0.0005] ; # Slice off max depth
    df = df[df['param_min_samples_leaf']==1] ; # Slice off max depth
    plot_complexity("chess_DT_entropy_min_impurity_decr=0.0005_min_samples_leaf=1", 'max_depth', df)
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_min_impurity_decrease']==0.0005] ; # Slice off max depth
    df = df[df['param_max_depth']==100] ; # Slice off max depth
    plot_complexity("chess_DT_entropy_min_impurity_decr=0.0005_max_depth=100", 'min_samples_leaf', df)

def plot_SVM_complexity():
    file = 'results/SVM_Linear_chess_reg.csv'
    df = get_model_complexity_data(file, 'penalty', 'l1')
    plot_complexity("chess_SVM_Linear_alpha", 'alpha', df)
    file = 'results/SVM_RBF_chess_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("chess_SVM_RBF_C", 'C', df)
    file = 'results/SVM_Linear_FMNIST_reg.csv'
    df = get_model_complexity_data(file, 'penalty', 'l1')
    plot_complexity("FMNIST_SVM_Linear_alpha", 'alpha', df)
    file = 'results/SVM_RBF_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_SVM_RBF_C", 'C', df)

def plot_kNN_complexity():
    file = 'results/kNN_chess_reg.csv'
    df = get_model_complexity_data(file, 'weights', 'uniform')
    plot_complexity("chess_kNN_uniform_n_neighbors", 'n_neighbors', df)
    df = get_model_complexity_data(file, 'weights', 'distance')
    plot_complexity("chess_kNN_distance_n_neighbors", 'n_neighbors', df)
    file = 'results/kNN_FMNIST_reg.csv'
    df = get_model_complexity_data(file, 'weights', 'uniform')
    plot_complexity("FMNIST_kNN_uniform_n_neighbors", 'n_neighbors', df)
    df = get_model_complexity_data(file, 'weights', 'distance')
    plot_complexity("FMNIST_kNN_distance_n_neighbors", 'n_neighbors', df)

def plot_NN_complexity():
    file = 'results/NN_chess_reg.csv'
    df = get_model_complexity_data(file, 'hidden_layer_sizes', "(50, 10)")
    plot_complexity("chess_NN_relu_hidden_layers=(50,10)_tol", 'tol', df)
    file = 'results/NN_FMNIST_reg.csv'
    df = get_model_complexity_data(file, 'tol', 10**-3)
    plot_complexity("FMNIST_NN_tol_10e-3_hidden_layers", 'hidden_layer_sizes', df)
    df = get_model_complexity_data(file, 'hidden_layer_sizes', "(50,)")
    plot_complexity("FMNIST_NN_hidden_layers_50_tol", 'tol', df)

def plot_NN_iteration_curve():
    file = 'results/NN_iteration_curve_chess_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("chess_NN_iteration_curve", 'max_iter', df)
    file = 'results/NN_iteration_curve_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_NN_iteration_curve", 'max_iter', df)

def plot_SVM_iteration_curve():
    file = 'results/SVM_RBF_iteration_curve_chess_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("chess_SVM_RBF_iteration_curve", 'max_iter', df)
    file = 'results/SVM_RBF_iteration_curve_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_SVM_RBF_iteration_curve", 'max_iter', df)
    file = 'results/SVM_RBF_iteration_curve_300_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_SVM_RBF_iteration_curve_300", 'max_iter', df)
    file = 'results/SVM_RBF_iteration_curve_500_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_SVM_RBF_iteration_curve_500", 'max_iter', df)
    file = 'results/SVM_RBF_iteration_curve_1000_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_SVM_RBF_iteration_curve_1000", 'max_iter', df)
    file = 'results/SVM_RBF_iteration_curve_2000_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_SVM_RBF_iteration_curve_2000", 'max_iter', df)

def plot_boosting_complexity():
    file = 'results/boosting_chess_reg.csv'
    df = get_model_complexity_data(file, 'base_estimator__splitter', 'random')
    df2 = df[df['param_n_estimators'] == 200]
    plot_complexity('chess_boosting_splitter_random_n_estimators_200_max_depth', 'base_estimator__max_depth', df2)
    df = get_model_complexity_data(file, 'base_estimator__splitter', 'random')
    df2 = df[df['param_base_estimator__max_depth'] == 5]
    plot_complexity('chess_boosting_splitter_random_max_depth_5_n_estimators', 'n_estimators', df2)
    df = get_model_complexity_data(file, 'base_estimator__splitter', 'best')
    df2 = df[df['param_n_estimators'] == 200]
    plot_complexity('chess_boosting_splitter_best_n_estimators_200_max_depth', 'base_estimator__max_depth', df2)
    df = get_model_complexity_data(file, 'base_estimator__splitter', 'best')
    df2 = df[df['param_base_estimator__max_depth'] == 5]
    plot_complexity('chess_boosting_splitter_best_max_depth_5_n_estimators', 'n_estimators', df2)
    file = 'results/boosting_FMNIST_reg.csv'
    df = get_model_complexity_data(file, 'n_estimators', 100)
    plot_complexity('FMNIST_boosting_splitter_random_100_estimators_max_depth', 'base_estimator__max_depth', df)
    df = get_model_complexity_data(file, 'base_estimator__max_depth', 20)
    plot_complexity('FMNIST_boosting_splitter_random_max_depth_20_n_estimators', 'n_estimators', df)
    df = get_model_complexity_data(file, 'base_estimator__max_depth', 10)
    plot_complexity('FMNIST_boosting_splitter_random_max_depth_10_n_estimators', 'n_estimators', df)
    file = 'results/boosting_FMNIST_best_reg.csv'
    df = get_model_complexity_data(file, 'n_estimators', 100)
    plot_complexity('FMNIST_boosting_splitter_best_100_estimators_max_depth', 'base_estimator__max_depth', df)
    df = get_model_complexity_data(file, 'base_estimator__max_depth', 20)
    plot_complexity('FMNIST_boosting_splitter_best_max_depth_20_n_estimators', 'n_estimators', df)
    df = get_model_complexity_data(file, 'base_estimator__max_depth', 10)
    plot_complexity('FMNIST_boosting_splitter_best_max_depth_10_n_estimators', 'n_estimators', df)

#plot_kNN_complexity()
#plot_DT_complexity()
#plot_NN_complexity()
#plot_boosting_complexity()
#plot_NN_iteration_curve()
#plot_SVM_iteration_curve()
#plot_distribution()


