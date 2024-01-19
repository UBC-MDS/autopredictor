import pandas as pd
import pytest
import sys
import os
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from unittest.mock import Mock, patch



current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
# Append the parent directory to sys.path
sys.path.append(os.path.dirname(current_dir))

from src.autopredictor.fit import fit

X, y = load_diabetes(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)


def test_output_dict():
    result = fit(X_train, X_test, y_train, y_test)
    assert isinstance(result, tuple), f"Expected a dictionary, but got {type(result)}"


def test_scores_test():
    result_test, result_train = fit(X_train, X_test, y_train, y_test)
    assert isinstance(result_test['Linear Regression']['Mean Absolute Error'], float)
    assert isinstance(result_test['Linear Regression (L1)']['Mean Absolute Error'], float)


def test_scores_test():
    result_test, result_train = fit(X_train, X_test, y_train, y_test, return_train=True)
    assert isinstance(result_train['Linear Regression']['Mean Absolute Error'], float)
    assert isinstance(result_train['Linear Regression (L1)']['Mean Absolute Error'], float)


def test_output_tuple():
    result_test, result_train = fit(X_train, X_test, y_train, y_test)
    assert result_train == {}


def test_fit_raises_value_error_for_missing_input():
    with pytest.raises(Exception):
        fit(None, None, None, None)


X_train_mock = pd.DataFrame({
    'feature1': [1, 2, 3],
    'feature2': [4, 5, 6],
    'feature3': [7, 8, 9]
})
y_train_mock = pd.Series([10, 20, 30])

X_test_mock = pd.DataFrame({
    'feature1': [10, 11, 12],
    'feature2': [13, 14, 15],
    'feature3': [16, 17, 18]
})
y_test_mock = pd.Series([40, 50, 60])

class MockModel(Mock):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return [1, 2, 3]  # Replace with appropriate mock data

class MockModel(Mock):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return [1, 2, 3]  # Replace with appropriate mock data

def test_fit_returns_metrics_for_each_model():
    # Mock the models with custom mock class
    with patch('src.autopredictor.fit.LinearRegression', new_callable=MockModel) as mock_linear, \
         patch('src.autopredictor.fit.Lasso', new_callable=MockModel) as mock_lasso, \
         patch('src.autopredictor.fit.Ridge', new_callable=MockModel) as mock_ridge:

        result = fit(X_train_mock, X_test_mock, y_train_mock, y_test_mock, return_train=True)

    # Verify that the result is a tuple containing two dictionaries
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], dict)  # Train metrics dictionary
    assert isinstance(result[1], dict)  # Test metrics dictionary

    # You can add more specific assertions on the metrics if needed
    assert 'Linear Regression' in result[0]
    assert 'Linear Regression (L1)' in result[0]
    assert 'Linear Regression (L2)' in result[0]
    assert 'Linear Regression' in result[1]
    assert 'Linear Regression (L1)' in result[1]
    assert 'Linear Regression (L2)' in result[1]





if __name__ == '__main__':
    pytest.main()
