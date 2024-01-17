from autopredictor.show_all import show_all
import pandas as pd
import pytest

def test_empty_dict_input():
    """ Test that the function return the correct statement when the input is an empty dictionary
    
    """
    expected = "Input should not be an empty dictionary. No training scores are avaialble. Call fit function to test the model."
    with pytest.raises(ValueError, match=expected):
        show_all({})

def test_not_dict_input():
    """ Test that the function return the correct statement when the input is not of correct type
    """
    expected = "Input should be of dictionary type."
    with pytest.raises(TypeError, match=expected):
        show_all('This is a string')
    with pytest.raises(TypeError, match=expected):
        show_all(pd.DataFrame({
        "A":{"MSE":0.5},
        "B":{"MSE":0.1}
    }))
    with pytest.raises(TypeError, match=expected):
        show_all(12)

def test_case_insensitivity_metric_names():
    """ Test that the function handle case insensitive metrics name well"""
    expected = pd.DataFrame({
        'AdaBoost': {
        'MAE': 0.982,
        'MAPE': 0.872,
        'R2': 0.521,
        'RMSE': 0.123, 
        'MSE': 0.765     
        }
    }).T
    
    inconsistent_names_dict = {
        'AdaBoost': {'Mean absolute error': 0.982,
        'Mean Absolute Percentage Error': 0.872,
        'R2 SCORE': 0.521,
        'Root Mean Squared Error': 0.123,
        'Mean Squared ERROR': 0.765
        }
    }
    actual = show_all (inconsistent_names_dict)
    assert set(actual) == set(expected)

def test_incorrect_scoring_metrics():
    """ Test that the input have the correct scoring metrics"""

    expected_incorrect = "Invalid scoring metrics for model."
    incorrect_model_scores = {
        "Logistic Regression": {'MAP':0.345, 'MS':0.988,"R2":0.782, "MSE":0.123, "RMSE":0.526}
    }
    with pytest.raises(ValueError, match=expected_incorrect):
        show_all(incorrect_model_scores)

def test_incomplete_scoring_metrics():
    """ Test that the input have the correct number scoring metrics"""

    expected_incomplete = "Scoring metrics is incomplete."
    incomplete_model_scores = {
        'AdaBoost': {'Mean absolute error': 0.982,
        'Mean Absolute Percentage Error': 0.872,
        'R2 SCORE': 0.521,
        'Root Mean Squared Error': 0.123
    }
    }
    with pytest.raises(ValueError, match=expected_incomplete):
        show_all(incomplete_model_scores)

def test_correct_output():
    """ Test that the function outputs correctly"""
    model_scores = {
        'Linear Regression': {'Mean Absolute Error': 0.453,
                            'Mean Absolute Percentage Error': 0.346,
                            'R2 Score': 0.512,
                            'Mean Squared Error': 0.567,
                            'Root Mean Squared Error': 0.987},
  'Linear Regression (L1)': {'Mean Absolute Error': 61.2,
                            'Mean Absolute Percentage Error': 0.457,
                            'R2 Score': 0.239,
                            'Mean Squared Error': 0.873,
                            'Root Mean Squared Error': 72.4},
  'Linear Regression (L2)': {'Mean Absolute Error': 55.2,
                            'Mean Absolute Percentage Error': 0.412,
                            'R2 Score': 0.379,
                            'Mean Squared Error': 0.678,
                            'Root Mean Squared Error': 65.3}
    }
    expected = pd.DataFrame({
        "Linear Regression": {'MAE':0.453, 'MAPE':0.346,"R2":0.512, "MSE":0.567, "RMSE":0.987},
        "Linear Regression (L1)": {'MAE':61.2, 'MAPE':0.457,"R2":0.239, "MSE":0.873, "RMSE":72.4},
        "Linear Regression (L2)": {'MAE':55.2, 'MAPE':0.412,"R2":0.379, "MSE":0.678, "RMSE":65.3}
    }).T
    actual = show_all(model_scores)
    pd.testing.assert_frame_equal(actual, expected), 'Show_all function is outputting incorrectly!'