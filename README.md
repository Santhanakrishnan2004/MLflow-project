

# MLflow Wine Quality Prediction

This project demonstrates using **MLflow** for experiment tracking and model monitoring while building an **ElasticNet regression model** to predict wine quality. It also integrates with **DagsHub** for versioning and logging.


## Project Overview

This project predicts the quality of red wine based on physicochemical tests. The workflow includes:

1. Loading and preparing the dataset
2. Splitting data into train and test sets
3. Training an **ElasticNet** regression model
4. Evaluating the model with **RMSE, MAE, and R2** metrics
5. Logging parameters, metrics, and the trained model to MLflow and DagsHub

---

## Dataset

The dataset used is from the [MLflow GitHub repository](https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv). It contains the following columns:

* `fixed acidity`
* `volatile acidity`
* `citric acid`
* `residual sugar`
* `chlorides`
* `free sulfur dioxide`
* `total sulfur dioxide`
* `density`
* `pH`
* `sulphates`
* `alcohol`
* `quality` (target)

---

## Requirements

Python libraries required:

```bash
pip install mlflow dagshub
```

---

## Usage

Run the script using:

```bash
python app.py [alpha] [l1_ratio]
```

* `alpha` – Regularization strength (default: 0.5)
* `l1_ratio` – Mix between L1 and L2 regularization (default: 0.5)

Example:

```bash
python app.py 0.7 0.3
```

---

## Model Evaluation

The model is evaluated using the following metrics:

* **RMSE (Root Mean Squared Error)** – Measures average magnitude of error
* **MAE (Mean Absolute Error)** – Measures average absolute difference
* **R2 Score** – Measures proportion of variance explained

---

## Logging and Monitoring

* **MLflow ** is enabled to  log parameters, metrics, and models.
* Custom metrics (`RMSE`, `MAE`, `R2`) are manually logged for more control.
* **DagsHub** integration allows you to version your data, code, and ML experiments.

---

## License

This project is open-source and available under the MIT License.

---
