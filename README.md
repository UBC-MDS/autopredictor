# autopredictor

This Autopredictor Python package streamlines the process of selecting and assessing machine learning models, presenting a simplified approach for evaluating different regression models without intricate manual setup. This tool facilitates the exploration of multiple models on a dataset, minimizing the coding required for selecting and fitting various models. Utilizing preprocessed and trained data, this package evaluates models using default settings, enabling users to swiftly comprehend model performance. By computing and showcasing diverse performance metrics for each model, it offers an efficient means to compare their effectiveness. Overall, Autopredictor provides a convenient and quick framework for initial model assessment and comparison in machine learning workflows.

This package includes four main functions:
- `fit`: Fits a clean, preprocessed training data into eight different regression models. This function returns a dictionary containing four metric scores for each model
- `show_all`: Generates a DataFrame presenting each scoring metric alongside the respective model, while outputting a clear overview of the results in a table format
- `display_best_score`: Identifies the best score with respect to a specific scoring metric along with the corresponding model
- `select_model`: Returns a summary of all the scoring metrics associated with a specific machine learning model

This package focuses on eight widely used regressor models, providing a curated selection that covers a broad range of algorithmic approaches. This package are designed to be user-friendly through automation with default configurations for each model. It is catered for both beginners by eliminating complicated model arguments and for experts by providing baseline results. However, this package may not be suitable for experienced practitioner who requires customized regressor models. Within the Python ecosystem, there is an existing, well developed and maintained library named [lazypredict](https://pypi.org/project/lazypredict/) that offer similar functionality with a wider range of models, including classification models.

## Installation

Since this package is still in the developing process, it has not been published on PyPI yet. Thus, In order to use this package, please run the instructions provided below.

### Initialization
1. Clone this GitHub repository using this command:
```bash
git clone https://github.com/UBC-MDS/autopredictor.git
```

2. Install poetry in your base environment by following these [instructions](https://python-poetry.org/docs/#installation).

3. Run the following commands from the root directory of this project to create a virtual environment for this package and install autopredictor through poetry:
```bash
conda create --name autopredictor python=3.9 -y
conda activate autopredictor
poetry install
```

### Future update

Once this package is published on PyPi, run the following command to intall autopredictor in the chosen environment:
```bash
$ pip install autopredictor
```

## Usage


To use `autopredictor`, follow these simple steps:

1. Import the package:

    ```python
    import autopredictor
    ```

2. Load your preprocessed training data.


3. Once you have your data split into training and testing, you can start by fitting the data to obtain scores for eight different regression models:
    ```python
    results_test, results_train = autopredictor.fit(X_train,
                      X_test,
                      y_train,
                      y_test,
                      return_train=True)
    ```

    The `fit` function returns a dictionary containing four metric scores for each model.

5. Display an overview of the results in a table format:

    ```python
    scores_test = autopredictor.show_all(results_test)
    scores_train = autopredictor.show_all(results_train)
    ```

By calling `autopredictor.show_all(results)`, the function will print a tabulated version of the resulting DataFrame for easier visualization.

6. Identify the best score with respect to a specific metric:

    ```python
    autopredictor.display_best_score(metric='r2')
    ```

7. Get a summary of all scoring metrics associated with a specific model:

    ```python
    predictor.select_model(model='Linear Regression')
    ```

### Example

```python
import autopredictor

# return_train will always default to False assuming the user does not want to see the train scores
scores, _ = autopredictor.fit(X_train, 
                           X_test, 
                           y_train, 
                           y_test)

# Display an overview of the results
test_df = autopredictor.show_all(scores)

# Identify the best score with respect to the R-squared metric
autopredictor.display_best_score(metric='r2')

# Get a summary of scoring metrics for the Linear Regression model
autopredictor.select_model(model='Linear Regression')
```


## Contributing

Interested in contributing? Check out the contributing [guidelines](https://github.com/UBC-MDS/autopredictor/blob/main/CONTRIBUTING.md). Please note that this project is released with a [Code of Conduct](https://github.com/UBC-MDS/autopredictor/blob/main/CONDUCT.md). By contributing to this project, you agree to abide by its terms. Please find the list of contributors [here](https://github.com/UBC-MDS/autopredictor/blob/main/CONTRIBUTORS.md).

## License

`autopredictor` was created by Anu Banga, Arturo Rey, Sharon Voon, Zeily Garcia. It is licensed under the terms of the MIT license.

## Credits

`autopredictor` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
