from autopredictor.bestscore import display_best_score
import pandas as pd
import pytest

def test_existing_scoring_metric():
    """Test that the function returns the correct DataFrame for an existing scoring metric."""
    df = pd.DataFrame({'MAE': [5.6, 3.4],
                       'MSE': [9.4, 21.4],
                       'MAPE': [0.34, 0.45]},
                       index=['Linear Regression', 'Random Forest'])

    expected_output = pd.DataFrame({
        'MAE': [5.6]
    }, index=['Linear Regression'])

    result_score, result_model = display_best_score(df, 'MAE')
    
    pd.testing.assert_frame_equal(result_score, expected_output)
    assert result_model == 'Linear Regression'


def test_invalid_scoring_metric():
    """Test that the function raises the correct error message for a invalid scoring metric."""
    df = pd.DataFrame({'MAE': [5.6, 3.4],
                       'MSE': [9.4, 21.4],
                       'MAPE': [0.34, 0.45]},
                       index=['Linear Regression', 'Random Forest'])

    expected_error = "Invalid Scoring metric 'R2'. The specified metric is not in the list of available metrics. Available metrics: MAE, MSE, MAPE."
    
    with pytest.raises(ValueError, match=expected_error):
        display_best_score(df, 'R2')


def test_empty_dataframe():
    """Test that the function raises a ValueError when the DataFrame is empty."""
    df_empty = pd.DataFrame()

    with pytest.raises(ValueError, match="DataFrame is empty."):
        display_best_score(df_empty, 'MAE')


def test_non_dataframe_input():
    """Test that the function raises a TypeError when input is not a DataFrame."""
    not_df = "Not a DataFrame"

    with pytest.raises(TypeError, match="Invalid DataFrame provided."):
        display_best_score(not_df, 'MAE')


def test_non_string_scoring_metric():
    """Test that the function raises a TypeError when scoring_metric is not a string."""
    df = pd.DataFrame({'MAE': [5.6, 3.4],
                       'MSE': [9.4, 21.4],
                       'MAPE': [0.34, 0.45]},
                       index=['Linear Regression', 'Random Forest'])

    with pytest.raises(TypeError, match="scoring_metric must be a string."):
        display_best_score(df, 123)
