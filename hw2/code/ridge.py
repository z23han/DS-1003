#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.model_selection import ParameterGrid

from ridge_regression import RidgeRegression
from setup_problem import load_problem


def get_data():
    lasso_data_fname = "lasso_data.pickle"
    x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

    X_train = featurize(x_train)
    X_val = featurize(x_val)

    return X_train, y_train, X_val, y_val


def test_lambda(X_train, y_train, X_val, y_val):
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.concatenate((y_train, y_val))
    val_fold = [-1]*len(X_train) + [0]*len(X_val)

    param_grid = [{'l2reg': np.unique(np.concatenate((10.**np.arange(-6,1,1), np.arange(1,3,.3) ))) }]
    
    ridge_regression_estimator = RidgeRegression()

    grid = GridSearchCV(ridge_regression_estimator,
        param_grid,
        return_train_score=True,
        cv = PredefinedSplit(test_fold=val_fold),
        refit = True,
        scoring = make_scorer(mean_squared_error, greater_is_better = False)
    )
    grid.fit(X_train_val, y_train_val)

    df = pd.DataFrame(grid.cv_results_)
    df['mean_test_score'] = -df['mean_test_score']
    df['mean_train_score'] = -df['mean_train_score']
    cols_to_keep = ["param_l2reg", "mean_test_score", "mean_train_score"]
    df_toshow = df[cols_to_keep].fillna('-')
    df_toshow = df_toshow.sort_values(by=["param_l2reg"])

    print(df_toshow)

    fig, ax = plt.subplots()
    ax.semilogx(df_toshow["param_l2reg"], df_toshow["mean_test_score"])
    ax.grid()
    ax.set_title("Validation Performance vs L2 Regularization")
    ax.set_xlabel("L2-Penalty Regularization Parameter")
    ax.set_ylabel("Mean Squared Error")
    plt.show()




def main():
    X_train, y_train, X_val, y_val = get_data()
    test_lambda(X_train, y_train, X_val, y_val)



if __name__ == '__main__':
    main()

