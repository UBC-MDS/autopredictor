
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.datasets import load_diabetes




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

def fit(X_train,X_test,y_train,y_test):
    if X_train and X_test and y_train and y_test:
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

        metrics = {
            'Mean Absolute Error': mean_absolute_error(),
            'Mean Absolute Percentage Error': mean_absolute_percentage_error(),
            'R2 Score': r2_score()
        }

        for name, model in models.items():
            model.fit(X_train,y_train)
            print(f'{name} trained.')

            scores_list[name] = {}

            for metric_name, metric in metrics.items():

                scores_list[name][metric_name] = metric(y_test, model.predict(X_test))

    else:
        raise('Please input a valid DataFrame')
    







