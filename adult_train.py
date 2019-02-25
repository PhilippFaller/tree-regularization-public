import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from train import GRUTree
from train import visualize
import os
import cPickle

if __name__ == "__main__":
    data = pd.read_csv("adult_data/adult.data", names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
                       sep=r'\s*,\s*',
                       engine='python',
                       na_values="?")
    target, _ = pd.Series(data["Target"].factorize())
    data = data.drop(columns=["Target"])

    feature_cats = []
    for feature in data.keys():
        if data[feature].dtype == 'object':
            cats = list(data[feature].astype("category").cat.categories)
        else:
            cats = None
        feature_cats.append(cats)
    data = pd.get_dummies(data)


    #Build Model
    samples, num_features = data.shape


    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(data.values, target, test_size = test_size)

    X_train = np.swapaxes(X_train, axis1=0, axis2=1)
    X_test = np.swapaxes(X_test, axis1=0, axis2=1)
    F_train = np.array(range(0, int(round(samples*(1-test_size)))))
    F_test = np.array(range(0, int(round(samples*(test_size)))))
    y_train = y_train[np.newaxis, :]
    y_test = y_test[np.newaxis, :]


    gru = GRUTree(num_features, 1, [25], 1, strength=1000)
    gru.train(X_train, F_train, y_train, iters_retrain=25, num_iters=300,
              batch_size=10, lr=1e-2, param_scale=0.1, log_every=10)

    if not os.path.isdir('./trained_models_adult'):
        os.mkdir('./trained_models_adult')

    with open('./trained_models_adult/trained_weights.pkl', 'wb') as fp:
        cPickle.dump({'gru': gru.gru.weights, 'mlp': gru.mlp.weights}, fp)
        print('saved trained model to ./trained_models_adult')

    visualize(gru.tree, './trained_models_adult/tree.pdf')
    print('saved final decision tree to ./trained_models_adult')

    y_hat = gru.pred_fun(gru.weights, X_test, F_test)
    auc_test = roc_auc_score(y_test.T, y_hat.T)
    print('Test AUC: {:.2f}'.format(auc_test))

    auc_test = mean_squared_error(y_test.T, y_hat.T)
    print('Test MSE: {:.2f}'.format(auc_test))

    auc_test = accuracy_score(y_test.T, np.round(y_hat.T))
    print('Test ACC: {:.2f}'.format(auc_test))
