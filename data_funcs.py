import pandas as pd
import numpy as np
import sklearn.model_selection as ms
from joblib import dump, load

def get_data(name, data_prop, test_prop):
    if name == "fmnist":
        data = pd.read_hdf('datasets_full.hdf', 'fmnist')
        dataX = data.drop('Class', 1).copy().values
        dataY = data['Class'].copy().values
    elif name == "chess":
        data = pd.read_hdf('datasets_full.hdf', 'chess')
        dataX = data.drop('win', 1).copy().values
        dataY = data['win'].copy().values
    else:
        print("ERROR: Unexpected value of name: ", name)
        return

    if data_prop == 1:
        data_dsX = dataX
        data_dsY = dataY
    else:
        data_dsX, data_dummyX, data_dsY, data_dummyY = ms.train_test_split(dataX, dataY, test_size = 1.0 - data_prop,
                                                                               random_state=0, stratify=dataY)
    data_trainX, data_testX, data_trainY, data_testY = ms.train_test_split(data_dsX, data_dsY, test_size = test_prop,
                                                                           random_state=0, stratify=data_dsY)
    return data_trainX, data_testX, data_trainY, data_testY

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


def perform_grid_search(estimator, type, dataset, params, trg_X, trg_Y, tst_X, tst_Y, cv=5, n_jobs=-1):
    cv = ms.GridSearchCV(estimator, n_jobs=n_jobs, param_grid= params, refit=True, verbose=2, cv=cv)
    cv.fit(trg_X, trg_Y)
    test_score = cv.score(tst_X, tst_Y)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./results/{}_{}_reg.csv'.format(type,dataset),index=False)
    with open('./results/test_results.csv','a') as f:
        f.write('{},{},{},{}\n'.format(type,dataset,test_score,cv.best_params_))

def write_best_results(type, dataset, test_score, clf, save=False):
    params = clf.get_params()
    if save:
        dump(clf, type+'_'+dataset+'.joblib')
    with open('./results/best_results.csv','a') as f:
        f.write('{},{},{},{}\n'.format(type,dataset,test_score,params))


