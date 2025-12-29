import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import scratch models
from backend.models.linear_svr import LinearSVR_Scratch
from backend.models.poly_svr import PolySVR_Scratch
from backend.models.rbf_svr import RBFSVRScratch

# -----------------------------
# Preprocessing helper
# -----------------------------
def preprocess_dataframe(df, target_col, sort_by_time=True):
    df = df.dropna(subset=[target_col])
    if sort_by_time and "time" in df.columns:
        df = df.sort_values(by="time").reset_index(drop=True)
    return df

def time_ordered_split(X, y, test_size=0.2):
    n = len(X)
    n_train = int((1 - test_size) * n)
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    return X_train, X_test, y_train, y_test

# -----------------------------
# Unified Training function
# -----------------------------
def train_model(df, target_col, model_type="linear", test_size=0.2, sort_by_time=True, split_type="Time-ordered split"):
    df = preprocess_dataframe(df, target_col, sort_by_time=sort_by_time)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Choose split strategy
    if split_type == "Time-ordered split":
        X_train, X_test, y_train, y_test = time_ordered_split(X, y, test_size=test_size)
    else:  # Random shuffle
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose model
    if model_type == "linear":
        model = SVR(kernel="linear", C=1.0, epsilon=0.1)
    elif model_type == "poly":
        model = SVR(kernel="poly", degree=2, C=1.0, epsilon=0.1)
    elif model_type == "rbf":
        model = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
    elif model_type == "linear_scratch":
        model = LinearSVR_Scratch(lr=0.003, epochs=1500, C=1.0, epsilon=0.02)
    elif model_type == "poly_scratch":
        model = PolySVR_Scratch(lr=0.001, epochs=900, C=1.0, epsilon=0.02, degree=2)
    elif model_type == "rbf_scratch":
        model = RBFSVRScratch(C=1.0, epsilon=0.02, gamma=None, lr=0.001, epochs=2000, mu=100.0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Fit + predict
    model.fit(X_train_scaled, y_train.values)
    y_pred = model.predict(X_test_scaled)

    metrics = {
        "MSE": mean_squared_error(y_test.values, y_pred),
        "MAE": mean_absolute_error(y_test.values, y_pred),
        "R2": r2_score(y_test.values, y_pred),
    }

    return (
        model,
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train.values,
            "y_test": y_test.values,
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
        },
        metrics,
        scaler,
    )

# -----------------------------
# Unified Auto-tune
# -----------------------------
def autotune_model(df, target_col, model_type="linear", k=3, sort_by_time=True, split_type="Time-ordered split"):
    """
    Lightweight auto-tune for all models.
    For scikit-learn: vary C, epsilon, gamma, degree.
    For scratch: vary lr, epochs, C, epsilon.
    """
    df = preprocess_dataframe(df, target_col, sort_by_time=sort_by_time)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Define param grids
    if model_type == "linear":
        param_grid = [
            {"C": 0.5, "epsilon": 0.01},
            {"C": 1.0, "epsilon": 0.05},
            {"C": 2.0, "epsilon": 0.1},
        ]
        kernel = "linear"

    elif model_type == "poly":
        param_grid = [
            {"C": 1.0, "epsilon": 0.05, "degree": 2},
            {"C": 1.0, "epsilon": 0.05, "degree": 3},
            {"C": 2.0, "epsilon": 0.1, "degree": 2},
        ]
        kernel = "poly"

    elif model_type == "rbf":
        param_grid = [
            {"C": 1.0, "epsilon": 0.05, "gamma": "scale"},
            {"C": 2.0, "epsilon": 0.1, "gamma": 0.01},
            {"C": 5.0, "epsilon": 0.1, "gamma": 0.1},
        ]
        kernel = "rbf"

    elif model_type == "linear_scratch":
        param_grid = [
            {"lr": 0.001, "epochs": 1000, "C": 1.0, "epsilon": 0.02},
            {"lr": 0.003, "epochs": 1500, "C": 1.0, "epsilon": 0.02},
            {"lr": 0.005, "epochs": 2000, "C": 2.0, "epsilon": 0.01},
        ]
    elif model_type == "poly_scratch":
        param_grid = [
            {"lr": 0.001, "epochs": 900, "C": 1.0, "epsilon": 0.02, "degree": 2},
            {"lr": 0.002, "epochs": 1200, "C": 1.0, "epsilon": 0.02, "degree": 3},
        ]
    elif model_type == "rbf_scratch":
        param_grid = [
            {"lr": 0.001, "epochs": 1500, "C": 1.0, "epsilon": 0.02, "mu": 100.0},
            {"lr": 0.002, "epochs": 2000, "C": 2.0, "epsilon": 0.01, "mu": 50.0},
        ]
    else:
        return None, None, None

    n_samples = len(X)
    fold_size = n_samples // k
    best_score = -float("inf")
    best_config = None
    best_metrics = None

    # Loop over param combinations
    for params in param_grid:
        fold_scores = []
        fold_metrics = {"MSE": [], "MAE": [], "R2": []}

        for fold in range(k):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k - 1 else n_samples
            val_idx = list(range(val_start, val_end))
            train_idx = list(range(0, val_start)) + list(range(val_end, n_samples))

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Train model depending on type
            if model_type in ["linear","poly","rbf"]:
                model = SVR(kernel=kernel, **params)
            elif model_type == "linear_scratch":
                model = LinearSVR_Scratch(**params)
            elif model_type == "poly_scratch":
                model = PolySVR_Scratch(**params)
            elif model_type == "rbf_scratch":
                model = RBFSVRScratch(**params)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)

            metrics = {
                "MSE": mean_squared_error(y_val, y_pred),
                "MAE": mean_absolute_error(y_val, y_pred),
                "R2": r2_score(y_val, y_pred),
            }
            fold_scores.append(metrics["R2"])
            for key in fold_metrics:
                fold_metrics[key].append(metrics[key])

        avg_score = np.mean(fold_scores)
        avg_metrics = {key: float(np.mean(vals)) for key, vals in fold_metrics.items()}

        if avg_score > best_score:
            best_score = avg_score
            best_config = params
            best_metrics = avg_metrics

    return best_config, best_metrics, best_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings

def train_benchmark_model(df, target_col, model_type="rf", test_size=0.2,
                          sort_by_time=True, split_type="Time-ordered split"):
    df = preprocess_dataframe(df, target_col, sort_by_time=sort_by_time)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Choose split strategy
    if split_type == "Time-ordered split":
        X_train, X_test, y_train, y_test = time_ordered_split(X, y, test_size=test_size)
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    # Scale features (MLP benefits, RF/XGB don’t strictly need it)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose model
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)   # train on arrays for consistency
        y_pred = model.predict(X_test_scaled)

    elif model_type == "mlp":
        # Suppress convergence warnings
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=2000,
                             solver="adam", random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    elif model_type == "xgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)


    else:
        raise ValueError(f"Unknown benchmark model type: {model_type}")

    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }

    return (
        model,
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train.values,
            "y_test": y_test.values,
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
        },
        metrics,
        scaler,
    )

def autotune_benchmark_model(df, target_col, model_type="rf", k=3,
                             sort_by_time=True, split_type="Time-ordered split"):
    df = preprocess_dataframe(df, target_col, sort_by_time=sort_by_time)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Simple param grids
    if model_type == "rf":
        param_grid = [
            {"n_estimators": 100, "max_depth": None},
            {"n_estimators": 200, "max_depth": 10},
            {"n_estimators": 300, "max_depth": 20},
        ]
        ModelClass = RandomForestRegressor
    elif model_type == "mlp":
        param_grid = [
            {"hidden_layer_sizes": (50,), "max_iter": 500},
            {"hidden_layer_sizes": (100,), "max_iter": 500},
            {"hidden_layer_sizes": (100, 50), "max_iter": 1000},
        ]
        ModelClass = MLPRegressor
    elif model_type == "xgb":
        param_grid = [
            {"learning_rate": 0.05, "max_iter": 200},
            {"learning_rate": 0.1, "max_iter": 300},
            {"learning_rate": 0.05, "max_iter": 500},
        ]
        from sklearn.ensemble import HistGradientBoostingRegressor
        ModelClass = HistGradientBoostingRegressor
    else:
        return None, None, None

    # k-fold CV
    n_samples = len(X)
    fold_size = n_samples // k
    best_score = -float("inf")
    best_config, best_metrics = None, None

    for params in param_grid:
        fold_scores, fold_metrics = [], {"MSE": [], "MAE": [], "R2": []}
        for fold in range(k):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k - 1 else n_samples
            val_idx = list(range(val_start, val_end))
            train_idx = list(range(0, val_start)) + list(range(val_end, n_samples))

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            if model_type == "rf" or model_type == "xgb":
                model = ModelClass(**params, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            else:  # mlp
                model = ModelClass(**params, random_state=42)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)

            metrics = {
                "MSE": mean_squared_error(y_val, y_pred),
                "MAE": mean_absolute_error(y_val, y_pred),
                "R2": r2_score(y_val, y_pred),
            }
            fold_scores.append(metrics["R2"])
            for key in fold_metrics:
                fold_metrics[key].append(metrics[key])

        avg_score = np.mean(fold_scores)
        avg_metrics = {key: float(np.mean(vals)) for key, vals in fold_metrics.items()}

        if avg_score > best_score:
            best_score, best_config, best_metrics = avg_score, params, avg_metrics

    return best_config, best_metrics, best_score