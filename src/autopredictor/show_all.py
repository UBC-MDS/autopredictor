def show_all(X):
    """
    This function converts the trained regressor model scores stored in a dictionary by 
    the "fit" function into a DataFrame, sorted alphabetically by model. It also 
    outputs the DataFrame in a table format.

    Parameters
    ----------
    X : dict
        A dictionary containing scoring metrics data for each regression model. The keys 
        represent model names where the values represent the scoring results and should 
        be numeric.

    Returns
    -------
    DataFrame
        A DataFrame containing all scoring metrics results alongside the corresponding 
        model, sorted alphabetically.

    Examples
    --------
    >>> from autopredictor.show_all import show_all
    >>> model_scores = {"A":{"MSE":0.568, "MAE":0.789, "MAPE":0.234, "R2":0.863},
                        "B":{"MSE":0.236, "MAE":0.680, "MAPE":0.854, "R2":0.653}}
    >>> result = show_all(model_scores)
    >>> print(result)
    """