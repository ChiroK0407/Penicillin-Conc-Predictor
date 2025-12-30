import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import scratch models
from backend.models.linear_svr import LinearSVR_Scratch
from backend.models.poly_svr import PolySVR_Scratch
from backend.models.rbf_svr import RBFSVRScratch

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings


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
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return X_train, X_test, y_train, y_test

def preprocess_general_testing(df, target_col, sort_by_time=True):
    """
    Preprocess mixed datasets for general testing:
    - Drops rows with missing target values
    - Sorts by 'time' column if present
    - Converts datetime/timestamp strings into numeric features (hour, day, month)
    - Keeps only numeric features
    - Imputes missing values with column means
    - Scales features
    Returns: X_scaled, y, scaler, imputer
    """
    df = df.dropna(subset=[target_col])

    if sort_by_time and "time" in df.columns:
        df = df.sort_values(by="time").reset_index(drop=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle datetime/timestamp columns
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col + "_hour"] = X[col].dt.hour
            X[col + "_day"] = X[col].dt.day
            X[col + "_month"] = X[col].dt.month
            X = X.drop(columns=[col])
        elif X[col].dtype == "object":
            parsed = pd.to_datetime(X[col], errors="coerce")
            if parsed.notna().sum() > 0.5 * len(parsed):
                X[col + "_hour"] = parsed.dt.hour
                X[col + "_day"] = parsed.dt.day
                X[col + "_month"] = parsed.dt.month
            X = X.drop(columns=[col])

    # Keep only numeric columns
    X_num = X.select_dtypes(include=["number"])
    X_num = X_num.dropna(axis=1, how="all")

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_num_imputed = imputer.fit_transform(X_num)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num_imputed)

    return X_scaled, y.values, scaler, imputer

# -----------------------------
# Training function (patched)
# -----------------------------
def train_model(df, target_col, model_type="linear", test_size=0.2,
                sort_by_time=True, split_type="Time-ordered split"):
    # Preprocess with robust helper
    X_scaled, y, scaler, imputer = preprocess_general_testing(df, target_col, sort_by_time=sort_by_time)

    # Split
    n = len(X_scaled)
    n_train = int((1 - test_size) * n)
    if split_type == "Time-ordered split":
        X_train, X_test, y_train, y_test = X_scaled[:n_train], X_scaled[n_train:], y[:n_train], y[n_train:]
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

    # Model choice
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
        model = RBFSVRScratch(C=1.0, epsilon=0.02, gamma=None,
                              lr=0.001, epochs=2000, mu=100.0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Fit + predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }

    # Return both arrays and DataFrames for plotting
    return (
        model,
        {
            "X_train": pd.DataFrame(X_train),   # DF for plotting
            "X_test": pd.DataFrame(X_test),
            "y_train": y_train,
            "y_test": y_test,
            "X_train_scaled": X_train,          # arrays for training
            "X_test_scaled": X_test,
        },
        metrics,
        scaler,
    )


def train_benchmark_model(df, target_col, model_type="rf", test_size=0.2,
                          sort_by_time=True, split_type="Time-ordered split"):
    # Preprocess with robust helper
    X_scaled, y, scaler, imputer = preprocess_general_testing(df, target_col, sort_by_time=sort_by_time)

    # Split
    n = len(X_scaled)
    n_train = int((1 - test_size) * n)
    if split_type == "Time-ordered split":
        X_train, X_test, y_train, y_test = X_scaled[:n_train], X_scaled[n_train:], y[:n_train], y[n_train:]
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

    # Model choice
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_type == "mlp":
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=2000,
                             solver="adam", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_type == "xgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    else:
        raise ValueError(f"Unknown benchmark model type: {model_type}")

    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }

    # Return both arrays and DataFrames for plotting
    return (
        model,
        {
            "X_train": pd.DataFrame(X_train),
            "X_test": pd.DataFrame(X_test),
            "y_train": y_train,
            "y_test": y_test,
            "X_train_scaled": X_train,
            "X_test_scaled": X_test,
        },
        metrics,
        scaler,
    )

def autotune_model(df, target_col, model_type="linear", k=3,
                   sort_by_time=True, split_type="Time-ordered split"):
    # ✅ Preprocess once
    X_scaled, y, scaler, imputer = preprocess_general_testing(
        df, target_col, sort_by_time=sort_by_time
    )

    # Define param grids + kernel
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
        kernel = None

    elif model_type == "poly_scratch":
        param_grid = [
            {"lr": 0.001, "epochs": 900, "C": 1.0, "epsilon": 0.02, "degree": 2},
            {"lr": 0.002, "epochs": 1200, "C": 1.0, "epsilon": 0.02, "degree": 3},
        ]
        kernel = None

    elif model_type == "rbf_scratch":
        param_grid = [
            {"lr": 0.001, "epochs": 1500, "C": 1.0, "epsilon": 0.02, "mu": 100.0},
            {"lr": 0.002, "epochs": 2000, "C": 2.0, "epsilon": 0.01, "mu": 50.0},
        ]
        kernel = None

    else:
        return None, None, None

    # k-fold CV
    n_samples = len(X_scaled)
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

            X_train, y_train = X_scaled[train_idx], y[train_idx]
            X_val, y_val = X_scaled[val_idx], y[val_idx]

            # Train model depending on type
            if model_type in ["linear", "poly", "rbf"]:
                model = SVR(kernel=kernel, **params)
            elif model_type == "linear_scratch":
                model = LinearSVR_Scratch(**params)
            elif model_type == "poly_scratch":
                model = PolySVR_Scratch(**params)
            elif model_type == "rbf_scratch":
                model = RBFSVRScratch(**params)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

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

def autotune_benchmark_model(df, target_col, model_type="rf", k=3,
                             sort_by_time=True, split_type="Time-ordered split"):
    # ✅ Preprocess once
    X_scaled, y, scaler, imputer = preprocess_general_testing(
        df, target_col, sort_by_time=sort_by_time
    )

    # Param grids
    if model_type == "rf":
        param_grid = [
            {"n_estimators": 100, "max_depth": None},
            {"n_estimators": 200, "max_depth": 10},
            {"n_estimators": 300, "max_depth": 20},
        ]
        from sklearn.ensemble import RandomForestRegressor
        ModelClass = RandomForestRegressor

    elif model_type == "mlp":
        param_grid = [
            {"hidden_layer_sizes": (50,), "max_iter": 500},
            {"hidden_layer_sizes": (100,), "max_iter": 500},
            {"hidden_layer_sizes": (100, 50), "max_iter": 1000},
        ]
        from sklearn.neural_network import MLPRegressor
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
    n_samples = len(X_scaled)
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

            X_train, y_train = X_scaled[train_idx], y[train_idx]
            X_val, y_val = X_scaled[val_idx], y[val_idx]

            if model_type in ["rf", "xgb"]:
                model = ModelClass(**params, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            else:  # mlp
                model = ModelClass(**params, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

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