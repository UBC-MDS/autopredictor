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
        A DataFrame containing the best score with respect to a specific scoring metric alongside the corresponding 
        model.
    """
    if scoring_metric not in X.columns:
        print(f"'{scoring_metric}' not found in the DataFrame.")
        return None

    max_score = X.loc[X[scoring_metric].idxmax()]

    result_table = pd.DataFrame([max_score])
    print(tabulate(result_table, headers='keys', tablefmt='github', showindex=False))

    return max_score