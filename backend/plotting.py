import matplotlib.pyplot as plt

def overlay_plot(X_test, y_true, y_pred_orig, y_pred_tuned, target_col):
    if "time" in X_test.columns:
        x_vals = X_test["time"].values
        x_label = "Time (h)"
    else:
        x_vals = range(len(y_true))
        x_label = "Sample index"

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_true, label="Actual", color="blue")
    ax.plot(x_vals, y_pred_orig, label="Original Predicted", color="red")
    ax.plot(x_vals, y_pred_tuned, label="Auto-tuned Predicted", color="green")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{target_col}")
    ax.set_title("Model Training vs Auto-tuned Predictions")
    ax.legend()
    return fig

import matplotlib.pyplot as plt

def multi_plot(X_test, y_true, predictions_dict, target_col):
    """
    Plot actual vs multiple model predictions.

    Parameters
    ----------
    X_test : pd.DataFrame
        Test features (must contain 'time' column if available).
    y_true : array-like
        Ground truth target values.
    predictions_dict : dict
        Dictionary of {model_name: y_pred_array}.
    target_col : str
        Name of the target column for labeling.
    """
    # Choose x-axis
    if "time" in X_test.columns:
        x_vals = X_test["time"].values
        x_label = "Time (h)"
    else:
        x_vals = range(len(y_true))
        x_label = "Sample index"

    fig, ax = plt.subplots()

    # Plot actual
    ax.plot(x_vals, y_true, label="Actual", color="black", linewidth=2)

    # Plot each model’s predictions
    colors = ["red", "green", "orange", "purple", "brown", "cyan"]
    for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(x_vals, y_pred, label=f"{model_name} Predicted", color=color)

    ax.set_xlabel(x_label)
    ax.set_ylabel(target_col)
    ax.set_title("Comparison of Multiple Models")
    ax.legend()
    return fig

import matplotlib.pyplot as plt

def multi_scatter_plot(y_true, predictions_dict, target_col):
    """
    Scatter plot of actual vs predicted values for multiple models.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    predictions_dict : dict
        Dictionary of {model_name: y_pred_array}.
    target_col : str
        Name of the target column for labeling.
    """
    fig, ax = plt.subplots()

    # Plot actual-actual reference line
    ax.plot(y_true, y_true, color="black", linestyle="--", label="Actual = Predicted")

    # Plot scatter points for each model
    colors = ["red", "green", "orange", "purple", "brown", "cyan"]
    for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        ax.scatter(y_true, y_pred, label=f"{model_name} Predicted", alpha=0.6, s=30, color=color)

    ax.set_xlabel(f"Actual {target_col}")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted Scatter Plot")
    ax.legend()
    return fig