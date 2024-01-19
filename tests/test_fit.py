import pandas as pd
import pytest
import sys
import os
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from autopredictor.fit import fit

X, y = load_diabetes(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

def test_output_dict():
    """Test if the fit function returns a tuple."""
    result = fit(X_train, X_test, y_train, y_test)
    assert isinstance(result, tuple)"


def test_scores_test():
    """Test if the fit function returns test scores and checks their types."""
    result_test, result_train = fit(X_train, X_test, y_train, y_test)
    assert isinstance(result_test['Linear Regression']['Mean Absolute Error'], float)
    assert isinstance(result_test['Linear Regression (L1)']['Mean Absolute Error'], float)


def test_scores_train():
    """Test if the fit function returns train scores and checks their types."""
    result_test, result_train = fit(X_train, X_test, y_train, y_test, return_train=True)
    assert isinstance(result_train['Linear Regression']['Mean Absolute Error'], float)
    assert isinstance(result_train['Linear Regression (L1)']['Mean Absolute Error'], float)


def test_output_tuple():
    """Test if the fit function returns an empty dictionary for train scores."""
    result_test, result_train = fit(X_train, X_test, y_train, y_test)
    assert result_train == {}


def test_fit_raises_value_error_for_missing_input():
    """Test if fit function raises a ValueError for missing input."""
    with pytest.raises(Exception):
        fit(None, None, None, None)

if __name__ == '__main__':
    pytest.main()
