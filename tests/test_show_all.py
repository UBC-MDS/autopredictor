from autopredictor.show_all import show_all
import pandas as pd

def test_empty_dict_input():
    """ Test that the function return the correct statement when the input is an empty dictionary
    
    """
    expected = "Input should not be an empty dictionary."
    actual = show_all({})
    assert actual == expected, "Handled empty dictionary incorrectly!"

def test_not_dict_input():
    """ Test that the function return the correct statement when the input is not of correct type
    """
    expected = "Input should be of dictionary type."
    actual_string = show_all('This is a string')
    actual_dataframe = show_all(pd.DataFramne({
        "A":{"MSE":0.5},
        "B":{"MSE":0.1}
    }))
    actual_int = show_all(12)
    assert expected == actual_string, "Handled incorrect input type incorrectly!"
    assert expected == actual_dataframe, "Handled incorrect input type incorrectly!"
    assert expected == actual_int, "Handled incorrect input type incorrectly!"

def test_empty_values():
    """ Test that the function return the correct statement when the input dictionary has empty values
    """
    expected = "No training scores available. Call fit function to train the model."
    actual = show_all({
        "A":{},
        "B":{}
    })
    assert expected == actual, "Handled empty dictionary input incorrectly!"

def test_inconsistent_metric_names():
    """ Test that the function handle inconsistent metrics name well"""
    inconsistent_names_dict = {
        "Logistic Regression": {'Mape':0.345, 'MSE':0.988},
        "Linear Regression": {'MAPE':0.672, 'mse':0.723}
    }
    expected = pd.DataFrame({
        "Logistic Regression":{"MAPE":0.5, 'MSE': 0.988},
        "Linear Regression":{"MAPE":0.672, 'MSE': 0.988}
    })
    actual = show_all(inconsistent_names_dict)
    assert expected == actual, "Handled inconsistent metric names incorrectly!"

def test_incomplete_incorrect_scoring_metrics():
    """ Test that the input have the correct scoring metrics"""
    incomplete_model_scores = {
        "Logistic Regression": {'Mape':0.345, 'MSE':0.988,"R2":0.782}
    }
    actual_incomplete = show_all(incomplete_model_scores)
    incorrect_model_scores = {
        "Logistic Regression": {'MAP':0.345, 'MS':0.988,"R2":0.782, "MSE":0.123, "RMSE":0.526}
    }
    actual_incorrect = show_all(incorrect_model_scores)
    assert actual_incomplete == False, "Handled incomplete scoring metric incorrectly!"
    assert actual_incorrect == False, "Handled incorrect scoring metric incorrectly!"

def test_correct_output():
    """ Test that the function outputs correctly"""
    model_scores = {
        "Logistic Regression": {'MAE':0.345, 'MAPE':0.988,"R2":0.782, "MSE":0.123, "RMSE":0.526},
        "Linear Regression": {'MAP':0.435, 'MS':0.888,"R2":0.282, "MSE":0.523, "RMSE":0.326},
        "Support Vector Machine": {'MAP':0.945, 'MS':0.938,"R2":0.772, "MSE":0.133, "RMSE":0.566}
    }
    expected = pd.DataFrame({
        "Linear Regression": {'MAP':0.435, 'MS':0.888,"R2":0.282, "MSE":0.523, "RMSE":0.326},
        "Logistic Regression": {'MAE':0.345, 'MAPE':0.988,"R2":0.782, "MSE":0.123, "RMSE":0.526},
        "Support Vector Machine": {'MAP':0.945, 'MS':0.938,"R2":0.772, "MSE":0.133, "RMSE":0.566}
    })
    actual = show_all(model_scores)
    assert expected == actual, 'Show_all function is outputting incorrectly!'