import os
import visualizations
import numpy as np
from loader_var import load_cached_variables

# ─── Configuration ────────────────────────────────────────────────────────────
# CHANGE THIS ID TO SWITCH SUBJECTS (201, 205, 206)
CURRENT_SUB = '201'

# Base Paths
BASE_PATH = r"C:/Users/owner/Desktop/Niv"
CACHE_PATH = os.path.join(BASE_PATH, "Motor imagery skill/cache")
FIGURES_PATH = os.path.join(BASE_PATH, "Niv_github/_Figures") 

# Subject-specific settings
SUBJECT_CONFIG = {
    '201': { 
        'window_size': 10, 'overlap': 5, 'window_dir': "10_day_window",
        'window_smooth': 5,
        'label_offset': 30,      # "label_days = 30"
        'delta_offset':30,      # "delta_day = day + 1"
        'start_adj': 0,          # No change to start_test_day
        'end_adj': 0,             # No change to end_test_day
        'smooth': True,
        'start_day':1
    },
    '205': { 
        'window_size': 5,  'overlap': 0, 'window_dir': "10_day_window",
        'window_smooth': 0,
        'label_offset': 3,       # "label_days = unique_days + 3"
        'delta_offset':4,        # "delta_day = day + 1"
        'start_adj': 1,          # "start_test_day += 1"
        'end_adj': -1,            # "end_test_day -= 1"
        'smooth': False,
        'start_day':1
    },
    '206': { 
        'window_size': 6,  'overlap': 0, 'window_dir': "10_day_window",
        'window_smooth': 0,
        'label_offset': 3,       # "label_days = unique_days + 3"
        'delta_offset':4,        # "delta_day = day + 1"
        'start_adj': 1,          # "start_test_day += 1"
        'end_adj': 0,             # No change to end_test_day
        'smooth': False,
        'start_day':1
    }
}

sub_config = SUBJECT_CONFIG[CURRENT_SUB]

def main():
    # ─── 1) Load all cached variables ──────────────────────────────────────────
    print(f"=== Running experiment for Subject {CURRENT_SUB} ===")
    
    sub_config = SUBJECT_CONFIG[CURRENT_SUB]
    load_path = os.path.join(CACHE_PATH, f'sub_{CURRENT_SUB}')
    print(f"Loading cache from: {load_path}")

    vars = load_cached_variables(load_path=load_path)
    globals().update(vars)

# ─── 2) Setup Directories ──────────────────────────────────────────
    # Dynamic Save Paths
    save_dir = os.path.join(FIGURES_PATH, f'sub_{CURRENT_SUB}')
    save_dir_10_day_window = os.path.join(save_dir, sub_config['window_dir'])
    save_dir_centers = os.path.join(save_dir, "centers")
    save_dir_trial_window = os.path.join(save_dir, "trial_window")

    # Ensure they exist
    for path in [save_dir, save_dir_10_day_window, save_dir_centers, save_dir_trial_window]:
        try:
            os.makedirs(path, exist_ok=True)
            print(f"[OK] Directory exists: {path}")
        except OSError as err:
            print(f"[ERROR] Could not create {path}: {err}")

# ─── Dynamic Space & Dimension Setup ──────────────────────────────────
    unique_days = np.unique(days_label)

    # 1. Automatically detect if we are using 6D (Sub 201/205) or 10D (Sub 206)
    n_csp_cols = X_csp_features_scaled.shape[1]
    csp_full_name = f'CSP-{n_csp_cols}D'
    
    print(f"Detected CSP dimensionality: {n_csp_cols} ({csp_full_name})")

    # 2. Build the spaces dictionary using the dynamic key
    spaces = {
        csp_full_name: X_csp_features_scaled,  # Dynamic Key (e.g., 'CSP-6D' or 'CSP-10D')
        'CSP-2D':      X_csp_features_scaled_2d,
        'PCA-2D':      X_pca_features_2D,
        'UMAP-2D':     X_umap_features_2D,
        'PCA-3D':      X_pca_features_3D,
        'UMAP-3D':     X_umap_features_3D
    }
    
    # 3. Build the dimensions dictionary
    dim_dict = {
        csp_full_name: n_csp_cols, # Dynamic Value
        'UMAP-3D': 3, 'PCA-3D': 3, 
        'UMAP-2D': 2, 'PCA-2D': 2, 'CSP-2D': 2
    }


    # ─── 3) Cluster‐separation plots ────────────────────────────────────────────
    plot_configs = [
        (X_csp_features_scaled_2d, 2, 'CSP'),
        (X_pca_features_2D,        2, 'PCA'),
        (X_umap_features_2D,       2, 'UMAP'),
        (X_pca_features_3D,        3, 'PCA'),
        (X_umap_features_3D,       3, 'UMAP'),
        (X_csp_features_scaled,    n_csp_cols, 'CSP')
    ]

    # print("Generating Cluster Separation Plots...")
    # for features, dim, reducer in plot_configs:
    #     visualizations.plot_auc_vs_cluster_separation(
    #         X_csp=X_csp_features_scaled,  # Pass full CSP for AUC
    #         X_reduced=features,           # Pass reduced for Separation
    #         y_label=y_label, 
    #         days_labels=days_label, 
    #         clf_loaded=clf_loaded,        # Needed for LDA
    #         start_test_day=start_test_day, 
    #         end_test_day=end_test_day,
    #         dim=dim, 
    #         reducer=reducer, 
    #         # Pass adjustments from config
    #         label_offset=sub_config['label_offset'],
    #         start_adj=sub_config['start_adj'],
    #         end_adj=sub_config['end_adj'],
    #         directory=save_dir
    #     )

    # # ─── 4) Variance‐vs‐accuracy smoothing ──────────────────────────────────────
    # print("Generating SMOOTHED Cluster Separation Plots...")
    # for features, dim, reducer in plot_configs:
    #         visualizations.plot_auc_vs_cluster_separation(
    #             X_csp=X_csp_features_scaled,
    #             X_reduced=features,
    #             y_label=y_label, 
    #             days_labels=days_label, 
    #             clf_loaded=clf_loaded,
    #             start_test_day=start_test_day, 
    #             end_test_day=end_test_day,
    #             dim=dim, 
    #             reducer=reducer,
    #             # Adjustments
    #             label_offset=sub_config['label_offset'],
    #             start_adj=sub_config['start_adj'],
    #             end_adj=sub_config['end_adj'],
    #             # SMOOTHING ENABLED
    #             smooth=sub_config['smooth'],
    #             window=sub_config['overlap'], 
    #             directory=save_dir
    #         )


    # ─── 5) Variance‐vs‐accuracy smoothing ──────────────────────────────────────
    # print("Running Smoothed Variance Analysis...")

    # # A) Search for best window (2-15 results: (5,14,15), inter/idle/motor, in original space
    # visualizations.plot_auc_vs_variances_smoothed(
    #     unique_days, auc_scores,
    #     inter_variances, intra_variances_idle, intra_variances_motor,
    #     start_test_day, end_test_day, 
    #     directory=save_dir, 
    #     min_window=2, max_window=15,
    #     label_offset=sub_config['label_offset'],
    #     start_adj=sub_config['start_adj'], 
    #     end_adj=sub_config['end_adj']      
    # )

    # # B) Fixed window from Config
    # fixed_win = sub_config['window_smooth'] #decideed by inter
    
    # visualizations.plot_auc_vs_variances_smoothed(
    #     unique_days, auc_scores,
    #     inter_variances, intra_variances_idle, intra_variances_motor,
    #     start_test_day, end_test_day, 
    #     directory=save_dir, 
    #     min_window=fixed_win, max_window=fixed_win,
    #     label_offset=sub_config['label_offset'],
    #     start_adj=sub_config['start_adj'], 
    #     end_adj=sub_config['end_adj']      
    # )

    # # ─── 5) Delta‐matrix analyses ─────────────────────────
    # print("Generating Delta Matrix analysis...")
    
    # delta_auc_matrix, delta_inter_var_matrix, delta_intra_var_matrix_idle, delta_intra_var_matrix_motor = visualizations.delta_auc_var(
    #     X_csp_features_scaled,
    #     y_label,
    #     days_label,
    #     clf_loaded,
    #     start_test_day,
    #     end_test_day,
    #     smooth=sub_config['smooth'],
    #     directory=save_dir,
    #     delta_offset=sub_config['delta_offset'] 
    # ) 
    
    # visualizations.delta_vs_raw(
    #     X_csp_features_scaled,
    #     y_label,
    #     days_label,
    #     clf_loaded,
    #     start_test_day,
    #     end_test_day,
    #     smooth=sub_config['smooth'], 
    #     directory=save_dir,
    #     delta_offset=sub_config['delta_offset']
    # )

    # # ─── 6) 10-day window projection ─────────────────────────

    # visualizations.plot_sliding_windows(
    #     days_label,y_label, clf_loaded,
    #     X_csp_features_scaled, X_csp_features_scaled_2d,
    #     X_pca_features_2D,X_umap_features_2D,
    #     X_pca_features_3D,X_umap_features_3D,
    #     save_dir_win=save_dir_10_day_window,
    #     start_day=1,window_size=sub_config['window_size'],overlap_size=sub_config['overlap'])


if __name__ == '__main__':
    main()
