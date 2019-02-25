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
    # Title info from names
    data["Mr"] = data["Name"].apply(lambda name: 1 if "Mr" in name and not "Mrs" in name else 0)
    data["Miss"] = data["Name"].apply(lambda name: 1 if "Miss" in name else 0)
    data["Mrs"] = data["Name"].apply(lambda name: 1 if "Mrs" in name else 0)
    data["Title"] = data["Name"].apply(
        lambda name: 1 if any([t in name for t in ["Major", "Capt", "Col", "Rev", "Master", "Dr"]]) else 0)

    # Fix format
    data["Sex"] = data["Sex"].apply(lambda sex: 0 if sex == "male" else 1)


    # Drop useless
    data = data.drop("Ticket", axis=1)
    data = data.drop("Cabin", axis=1)
    data = data.drop("PassengerId", axis=1)
    data = data.drop("Name", axis=1)

    # Categorize
    data["1st"] = data["Pclass"].apply(lambda t: 1 if t == 1 else 0)
    data["2nd"] = data["Pclass"].apply(lambda t: 1 if t == 2 else 0)
    data["3rd"] = data["Pclass"].apply(lambda t: 1 if t == 3 else 0)
    data = data.drop("Pclass", axis=1)
    # Just noise
    data["Cherbourg"] = data["Embarked"].apply(lambda s: 1 if s == "C" else 0)
    data["Queenstown"] = data["Embarked"].apply(lambda s: 1 if s == "Q" else 0)
    data["Southampton"] = data["Embarked"].apply(lambda s: 1 if s == "S" else 0)
    data = data.drop("Embarked", axis=1)
    # Sanitize
    data["Age_valid"] = data["Age"].apply(lambda a: 0 if isnan(a) else 1)
    data["Age"] = data["Age"].apply(lambda a: -1 if isnan(a) else a)
    data["Fare"] = data["Fare"].apply(lambda x: data["Fare"].mean() if isnan(x) else x)

    # Truncate Outliers
    data["SibSp"] = data["SibSp"].apply(lambda s: min(4, s))
    data["Parch"] = data["Parch"].apply(lambda s: min(3, s))
    data["Fare"] = data["Fare"].apply(lambda s: np.log(s + 1))

    return data


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