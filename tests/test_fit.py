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




if __name__ == '__main__':
    pytest.main()
