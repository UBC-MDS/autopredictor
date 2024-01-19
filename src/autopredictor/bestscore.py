import pandas as pd
from tabulate import tabulate

def display_best_score(X, scoring_metric):
    """
    This function identifies the best score with respect to a specific scoring metric along with the corresponding model.
    It returns a DataFrame and displays the result in a table format.

    Parameters
    ----------
    X : DataFrame
        A DataFrame containing all scoring metrics results alongside the corresponding model, sorted alphabetically.
    scoring_metric : str
        A string containing the regression scoring metric, which is used to display best model.    

    Returns
    -------
    DataFrame
        If the scoring metric is found, a dataframe containing the best score and the corresponding model is returned.
        If the scoring metric is not found, a ValueError is raised.

    Examples
    --------
    >>> from autopredictor.bestscore import display_best_score
    >>> df = pd.DataFrame({'MAE': [5.6, 3.4],
                                  'MSE': [9.4, 21.4],
                                  'MAPE': [0.34, 0.45]},
                                 index=['Linear Regression', 'Random Forest'])
    >>> display_best_score(df, 'MAE')
                       MAE  
    Linear Regression  5.6
    
    >>> display_best_score(df, 'R2')
    ValueError: Invalid Scoring metric 'R2'.The specified metric is not in the list of available metrics. Available metrics: MAE, MSE, MAPE.
   """
    if X is None or not isinstance(X, pd.DataFrame):
        raise TypeError("Invalid DataFrame provided.")
    
    if X.empty:
        raise TypeError("DataFrame is empty.")

    if scoring_metric not in X.columns:
        available_metrics = X.columns.tolist()
        available_metrics_string = ", ".join(available_metrics)
        raise ValueError (f"Invalid Scoring metric '{scoring_metric}'.The specified metric is not in the list of available metrics. Available metrics: {available_metrics_string}.")
    
    if X[scoring_metric].isnull().any():
        raise ValueError(f"Invalid Scoring metric '{scoring_metric}'. The specified metric contains null values. Please handle or remove null values before using this function.")

    best_model = X[scoring_metric].idxmax()
    best_score = X.loc[best_model, scoring_metric]

    result_table = pd.DataFrame({scoring_metric: [best_score]}, index=[best_model])
    print(tabulate(result_table, headers='keys', tablefmt='github', showindex=True))

    return result_table

if __name__ == '__main__':
    display_best_score()