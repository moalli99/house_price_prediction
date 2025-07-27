from sklearn.linear_model import LinearRegression, Ridge, Lasso
from featureEngineering import *
from data import load_train_data
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml
import sys
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
sys.path.append("..")
from utils.loggers import get_logger

# Initialize logger for logging results
logger = get_logger()

# Linear Regression with cross-validation, returns R² score
def simple_regression(X, y, cv):
    pipeline = Pipeline([
        ("scaller", StandardScaler()),
        ("model", LinearRegression())
    ])
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
    logger.info(f"Linear Regression CV R2 score: {scores.mean()}")
    return scores.mean()

# Ridge Regression with GridSearchCV, returns the best trained pipeline
def try_ridge(X, y, cv, param_grid, scoring):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])
    grid = GridSearchCV(estimator=pipeline, cv=cv, param_grid=param_grid, scoring=scoring)
    grid.fit(X, y)
    logger.info(f"Ridge best R2 score: {grid.best_score_}")
    return grid.best_estimator_

# Lasso Regression with GridSearchCV, returns the best trained pipeline
def try_lasso(X, y, cv, param_grid, scoring):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso())
    ])
    grid = GridSearchCV(estimator=pipeline, cv=cv, param_grid=param_grid, scoring=scoring)
    grid.fit(X, y)
    logger.info(f"Lasso best R2 score: {grid.best_score_}")
    return grid.best_estimator_

if __name__ == "__main__":
    # Load YAML config file
    def load_config(path="C:/HousePrice/utils/config.yaml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    config = load_config()

    # Set up cross-validation config
    cv_config = config["cross_validation"]
    cv = KFold(n_splits=cv_config["n_splits"], shuffle=cv_config["shuffle"], random_state=cv_config["random_state"])

    # Load model parameter grids from config
    param_grid_ridge = config["models"]["ridge"]["param_grid"]
    param_grid_lasso = config["models"]["lasso"]["param_grid"]
    scoring = config["scoring"]

    # Load and preprocess training data
    PATH = config["path"]["train"]
    train = load_train_data(PATH)
    df = train.copy()
    df = processor(df)

    # Separate features and target
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    y_log = y.apply(np.log1p)  # Apply log1p to target to reduce skew

    # Feature selection (optional)
    if config["feature_selection"]:
        logger.info("Trying feature selection...")
        selected_feature = feature_selection(X, y_log)
        X = X[selected_feature]

    # Train and evaluate models
    r2_lr = simple_regression(X, y_log, cv)
    lasso_model = try_lasso(X, y_log, cv, param_grid_lasso, scoring)
    ridge_model = try_ridge(X, y_log, cv, param_grid_ridge, scoring)

    # Save the best ridge model
    joblib.dump(ridge_model, "ridge_model.pkl")
