from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def fit(X_train,X_test,y_train,y_test):
    """
    Train and evaluate multiple regression models on the given training and test data.

    Parameters
    ----------
    X_train: DataFrame 
             Training data features.
    X_test:  DataFrame 
             Test data features.
    y_train: Series
             Training data target values.
    y_test:  Series 
             Test data target values.

    Raises
    ------
    ValueError: If any of the inputs are empty or None.

    Returns
    -------
    dict 
        A dictionary containing the performance scores for each model and metric.
    """
    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        scores_list = {}

        models = {
            'Linear Regression': LinearRegression(),
            'Linear Regression (L1)': Lasso(),
            'Linear Regression (L2)': Ridge(),
            'Linear Support Vector Machine': LinearSVR(),
            'Support Vector Machine': SVR(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'AdaBoost': AdaBoostRegressor()
        }

        

        for name, model in models.items():
            model.fit(X_train,y_train)
            print(f'{name} trained.')

            scores_list[name] = {}

            metrics = {
                'Mean Absolute Error': mean_absolute_error(y_test, model.predict(X_test)),
                'Mean Absolute Percentage Error': mean_absolute_percentage_error(y_test, model.predict(X_test)),
                'R2 Score': r2_score(y_test, model.predict(X_test)),
                'Mean Squared Error': mean_squared_error(y_test, model.predict(X_test)),
                'Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
                

            }

            for metric_name, metric in metrics.items():

                scores_list[name][metric_name] = metric

    else:
        raise Exception('Please input a valid DataFrame')


    return scores_list


if __name__ == '__main__':
    fit()
