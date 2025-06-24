import os
import joblib

def load_cached_variables(load_path=None):
    """
    Returns a dict with all the variables saved by data_preparation.py.
    """
    if load_path is None:
        load_path = os.path.join(os.path.dirname(__file__), 'cache')

    names = [
        "start_test_day",          
        "end_test_day",
        "clf_loaded",
        "X_csp_features",
        "X_csp_features_scaled",
        "X_csp_features_train",
        "X_csp_features_scaled_train",
        "denoised_signal_train_centered",
        "denoised_signal_test_centered",
        "y_label",
        "y_label_train",
        "days_label",
        "days_label_train",
        "accuracies",
        "inter_variances",
        "intra_variances_idle",
        "intra_variances_motor",
        "X_csp_features_scaled_1d",
        "X_csp_features_scaled_2d",
        "X_pca_features_2D",
        "X_umap_features_2D",
        "X_pca_features_3D",
        "X_umap_features_3D",
        "eigen_values",
        "eigen_vectors",
        "cv_results",
        "auc_scores"
    ]


    vars_dict = {}
    for name in names:
        path = os.path.join(load_path, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cache file missing: {path}")
        vars_dict[name] = joblib.load(path)

    print("✅ Loaded all cached variables.")
    return vars_dict
