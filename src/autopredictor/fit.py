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