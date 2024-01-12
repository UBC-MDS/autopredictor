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