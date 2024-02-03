from autopredictor.bestscore import display_best_score
import pandas as pd
import pytest

def test_existing_scoring_metric():
    """Test that the function returns the correct DataFrame for an existing scoring metric."""
    df = pd.DataFrame({'MAE': [5.6, 3.4],
                           'MSE': [9.4, 21.4],
                           'MAPE': [0.34, 0.45],
                           'R2': [0.239, 0.712]},
                           index=['Linear Regression', 'Random Forest'])
    
    expected_output = pd.DataFrame({
        'MAE': [3.4]
    }, index=['Random Forest'])

    result = display_best_score(df, 'MAE')

    pd.testing.assert_frame_equal(result, expected_output)

def test_existing_scoring_metric_R2():
    """Test that the function returns the correct DataFrame for R2 scoring metric."""
    df = pd.DataFrame({'MAE': [5.6, 3.4],
                           'MSE': [9.4, 21.4],
                           'MAPE': [0.34, 0.45],
                           'R2': [0.239, 0.712]},
                           index=['Linear Regression', 'Random Forest'])
    
    expected_output_2 = pd.DataFrame({
        'R2': [0.712]
    }, index=['Random Forest'])

    result_2 = display_best_score(df, 'R2')

    pd.testing.assert_frame_equal(result_2, expected_output_2)

def test_invalid_scoring_metric():
    """Test that the function raises the correct error message for a invalid scoring metric."""
    df = pd.DataFrame({'MAE': [5.6, 3.4],
                       'MSE': [9.4, 21.4],
                       'MAPE': [0.34, 0.45]},
                       index=['Linear Regression', 'Random Forest'])

    invalid_metric = 'R2'
    available_metrics = df.columns.tolist()
    expected_error = f"Invalid Scoring metric '{invalid_metric}'. The specified metric is not in the list of available metrics. Available metrics: {', '.join(available_metrics)}."
    
    with pytest.raises(ValueError, match=expected_error):
        display_best_score(df, 'R2')

def test_empty_dataframe():
    """Test that the function raises a ValueError when the DataFrame is empty."""
    df_empty = pd.DataFrame()

    with pytest.raises(TypeError, match="DataFrame is empty."):
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

    with pytest.raises(ValueError, match="scoring_metric must be a string."):
        display_best_score(df, 123)

def test_null_values_in_scoring_metric():
    """Test that the function raises a ValueError when the specified metric contains null values."""
    df_with_null = pd.DataFrame({'MAE': [5.6, 3.4, None],
                                 'MSE': [9.4, 21.4, 15.0],
                                 'MAPE': [0.34, 0.45, 0.2]},
                                index=['Linear Regression', 'Random Forest', 'XGBoost'])

    scoring_metric_with_null = 'MAE'

    with pytest.raises(ValueError, match=f"Invalid Scoring metric '{scoring_metric_with_null}'. The specified metric contains null values. Please handle or remove null values before using this function."):
        display_best_score(df_with_null, scoring_metric_with_null)   
