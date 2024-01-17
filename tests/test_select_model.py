from autopredictor.select_model import select_model
import pandas as pd
import pytest

def test_select_existing_model():
    """ Test that the function returns the correct DataFrame row for an existing model """
    df_scores = pd.DataFrame({
        'MAE': [2.5, 3.6],
        'MSE': [10.1, 20.3]
    }, index=['Linear Regression', 'Random Forest'])
    
    expected = pd.DataFrame({
        'MAE': [2.5],
        'MSE': [10.1]
    }, index=['Linear Regression'])
    
    result = select_model(df_scores, 'Linear Regression')
    pd.testing.assert_frame_equal(result, expected)


def test_select_non_existing_model():
    """ Test that the function returns the correct error message for a non-existing model """
    df_scores = pd.DataFrame({
        'MAE': [2.5, 3.6],
        'MSE': [10.1, 20.3]
    }, index=['Linear Regression', 'Random Forest'])
    
    expected = "Model 'Support Vector Machine' not found. Here is the list of the models available: Linear Regression, Random Forest."
    result = select_model(df_scores, 'Support Vector Machine')
    assert result == expected


def test_empty_dataframe():
    """Test that the function raises a ValueError when the DataFrame is empty."""
    df_empty = pd.DataFrame()

    with pytest.raises(ValueError, match="df_output DataFrame is empty."):
        select_model(df_empty, 'Linear Regression')


def test_non_dataframe_input():
    """Test that the function raises a TypeError when input is not a DataFrame."""
    not_a_dataframe = "Not a DataFrame"

    with pytest.raises(TypeError, match="df_output must be a pandas DataFrame."):
        select_model(not_a_dataframe, 'Linear Regression')


def test_non_string_model_name():
    """Test that the function raises a TypeError when model_name is not a string."""
    df_scores = pd.DataFrame({
        'MAE': [2.5],
        'MSE': [10.1]
    }, index=['Linear Regression'])

    with pytest.raises(TypeError, match="model_name must be a string."):
        select_model(df_scores, 123)


def test_select_model_case_sensitivity():
    """Test case sensitivity in model name matching."""
    df_scores = pd.DataFrame({
        'MAE': [2.5],
        'MSE': [10.1]
    }, index=['Linear Regression'])

    result = select_model(df_scores, 'linear regression')
    assert result == "Model 'linear regression' not found. Here is the list of the models available: Linear Regression."


def test_select_model_with_whitespace():
    """Test that the function handles model names with leading or trailing whitespace."""
    df_scores = pd.DataFrame({
        'MAE': [2.5],
        'MSE': [10.1]
    }, index=['Linear Regression'])

    expected_output = df_scores.loc[['Linear Regression']]
    actual_output = select_model(df_scores, 'Linear Regression ')
    pd.testing.assert_frame_equal(actual_output, expected_output)




