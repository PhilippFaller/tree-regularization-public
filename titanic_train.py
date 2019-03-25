import pandas as pd
from math import isnan
import numpy as np
from train import GRUTree
from train import visualize
import os
import cPickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score


def preprocess(data):
    # complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    # complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    # complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
    # delete the cabin feature/column and others previously stated to exclude in train dataset
    drop_column = ['PassengerId', 'Cabin', 'Ticket']
    dataset.drop(drop_column, axis=1, inplace=True)

    # Discrete variables
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = "Alone"  # initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = "Not alone"  # now update to no/0 if family size is greater than 1
    dataset.drop(["SibSp", "Parch"], axis=1, inplace=True)

    # quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    # Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
    # Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    #dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    # Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    #dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

    # cleanup rare title names
    # print(data1['Title'].value_counts())
    stat_min = 10  # while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
    title_names = (dataset['Title'].value_counts() < stat_min)  # this will create a true false series with title name as index

    # apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    dataset.drop("Name", axis=1, inplace=True)

    return pd.get_dummies(data)


if __name__ == "__main__":
    dataframe = pd.read_csv("titanic_data/train.csv")
    target = dataframe["Survived"]
    data = dataframe.drop("Survived", axis=1)
    data = preprocess(data)
    samples, features = data.shape


    X_train, X_test, y_train, y_test = train_test_split(data.values, target, test_size = 0.33)

    X_train = np.swapaxes(X_train, axis1=0, axis2=1)
    X_test = np.swapaxes(X_test, axis1=0, axis2=1)
    F_train = np.array(range(0, int(round(samples*(0.66)))))
    F_test = np.array(range(0, int(round(samples*(0.33)))))
    y_train = y_train[np.newaxis, :]
    y_test = y_test[np.newaxis, :]

    #Build Model

    gru = GRUTree(features, 1, [25], 1, strength=1000)
    gru.train(X_train, F_train, y_train, iters_retrain=25, num_iters=300,
              batch_size=10, lr=1e-2, param_scale=0.1, log_every=10)

    if not os.path.isdir('./trained_models_titanic'):
        os.mkdir('./trained_models_titanic')

    with open('./trained_models_titanic/trained_weights.pkl', 'wb') as fp:
        cPickle.dump({'gru': gru.gru.weights, 'mlp': gru.mlp.weights}, fp)
        print('saved trained model to ./trained_models_titanic')

    visualize(gru.tree, './trained_models_titanic/tree.pdf')
    print('saved final decision tree to ./trained_models_titanic')

    y_hat = gru.pred_fun(gru.weights, X_test, F_test)
    auc_test = roc_auc_score(y_test.T, y_hat.T)
    print('Test AUC: {:.2f}'.format(auc_test))

    auc_test = mean_squared_error(y_test.T, y_hat.T)
    print('Test MSE: {:.2f}'.format(auc_test))

    auc_test = accuracy_score(y_test.T, np.round(y_hat.T))
    print('Test ACC: {:.2f}'.format(auc_test))
