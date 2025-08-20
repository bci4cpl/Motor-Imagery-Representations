import os
import visualizations
import utils
import numpy as np
from loader_var import load_cached_variables

def main():
    # ─── 1) Load all cached variables ──────────────────────────────────────────

    #C:\Users\owner\Desktop\Niv\Motor imagery skill\cache

    vars = load_cached_variables(load_path='d:\\Niv\\Motor imagery skill\\cache\\sub_201')
    # vars = load_cached_variables(load_path='d:\\Niv\\Motor imagery skill\\cache\\sub_205')
    # vars = load_cached_variables(load_path='d:\\Niv\\Motor imagery skill\\cache\\sub_206')


    globals().update(vars)
    

    # ───2) Setup ──────────────────────────────────────────


    save_dir = r"D:/Niv/Motor imagery skill/Figures/sub_201"
    save_dir_10_day_window = r'D:/Niv/Motor imagery skill/Figures/sub_201/10_day_window'
    save_dir_centers = r"D:/Niv/Motor imagery skill/Figures/sub_201/centers"
    save_dir_trial_window = r'D:/Niv/Motor imagery skill/Figures/sub_206/trial_window'


    # save_dir = r"D:/Niv/Motor imagery skill/Figures/sub_205"
    # save_dir_10_day_window = r'D:/Niv/Motor imagery skill/Figures/sub_205/10_day_window'
    # save_dir_centers = r"D:/Niv/Motor imagery skill/Figures/sub_205/centers"
    # save_dir_trial_window = r'D:/Niv/Motor imagery skill/Figures/sub_206/trial_window'


    # save_dir = r"D:/Niv/Motor imagery skill/Figures/sub_206"
    # save_dir_10_day_window = r'D:/Niv/Motor imagery skill/Figures/sub_206/10_day_window'
    # save_dir_centers = r"D:/Niv/Motor imagery skill/Figures/sub_206/centers"
    # save_dir_trial_window = r'D:/Niv/Motor imagery skill/Figures/sub_206/trial_window'
    # unique_days = np.unique(days_label)

    spaces = {
        # 'CSP-6D': X_csp_features_scaled, # for sub 201,205
        'CSP-2D': X_csp_features_scaled_2d,
        # "CSP-1D": X_csp_features_scaled_1d,
        'PCA-2D': X_pca_features_2D,
        'UMAP-2D': X_umap_features_2D,
        'PCA-3D': X_pca_features_3D,
        'UMAP-3D': X_umap_features_3D
        ,'CSP-10D': X_csp_features_scaled # for sub 206 the full space had 10 Components

    }
    dim_dict= {'CSP-10D': 10, 'UMAP-3D': 3, 'PCA-3D': 3, 'UMAP-2D':2, 'PCA-2D':2, 'CSP-2D': 2}


    # spaces_demo = { "CSP-1D": X_csp_features_scaled_1d}


    # ─── 3) Cluster‐separation plots ────────────────────────────────────────────
    visualizations.plot_auc_vs_cluster_separation(
          X_csp_features_scaled,X_csp_features_scaled_2d,
          y_label, days_label, clf_loaded,
          start_test_day,end_test_day,dim=2,directory=save_dir)

    visualizations.plot_auc_vs_cluster_separation(
        X_csp_features_scaled, X_pca_features_2D,
        y_label, days_label, clf_loaded,
        start_test_day, end_test_day,
        dim=2, reducer='PCA',directory=save_dir
    )

    visualizations.plot_auc_vs_cluster_separation(
        X_csp_features_scaled,X_umap_features_2D,
        y_label, days_label, clf_loaded,
        start_test_day,end_test_day,
        dim=2, reducer='UMAP',directory=save_dir
    )

    visualizations.plot_auc_vs_cluster_separation(
        X_csp_features_scaled,X_pca_features_3D,
        y_label, days_label, clf_loaded,
        start_test_day,end_test_day,
        dim=3, reducer='PCA',directory=save_dir
    )

    visualizations.plot_auc_vs_cluster_separation(
        X_csp_features_scaled,X_umap_features_3D,
        y_label, days_label, clf_loaded,
        start_test_day,end_test_day,
        dim=3, reducer='UMAP',directory=save_dir
    )

    visualizations.plot_auc_vs_cluster_separation(
        X_csp_features_scaled,X_csp_features_scaled,
        y_label, days_label, clf_loaded,
        start_test_day,end_test_day,
        directory=save_dir
    )

    # ─── 4) Variance‐vs‐accuracy smoothing ──────────────────────────────────────

    # # smoothed over windows of varying size CSP6D trying to find the best window (5,14,15)
    # smoothed_auc_inter, smoothed_inter_variances, smoothed_auc_idle, smoothed_intra_variances_idle,  smoothed_auc_motor ,smoothed_intra_variances_motor =  visualizations.plot_auc_vs_variances_smoothed(
    #     unique_days, auc_scores,
    #     inter_variances, intra_variances_idle, intra_variances_motor,
    #     start_test_day, end_test_day, save_dir, min_window=2, max_window=15) # smoothed
    


    # # fixed window (e.g. best inter‐correlation)
    # smoothed_auc_inter, smoothed_inter_variances, smoothed_auc_idle, smoothed_intra_variances_idle,  smoothed_auc_motor ,smoothed_intra_variances_motor = visualizations.plot_auc_vs_variances_smoothed(
    #     unique_days, auc_scores,
    #     inter_variances, intra_variances_idle, intra_variances_motor,
    #     start_test_day, end_test_day, save_dir, min_window=5, max_window=5) 

    # visualizations.main_graph(unique_days, auc_scores, inter_variances,start_test_day, end_test_day, save_dir, min_window=5, max_window=5)

    # # ─── 5) Delta‐matrix analyses ─────────────────────────

    # delta_auc_matrix, delta_inter_var_matrix, delta_intra_var_matrix_idle, delta_intra_var_matrix_motor = visualizations.delta_auc_var(X_csp_features_scaled,y_label,days_label,clf_loaded,start_test_day,end_test_day,directory=save_dir) 

    # # ─── 6) 10-day window projection
    # SUB201
    visualizations.plot_sliding_windows(
        days_label,y_label, clf_loaded,
        X_csp_features_scaled, X_csp_features_scaled_2d,
        X_pca_features_2D,X_umap_features_2D,
        X_pca_features_3D,X_umap_features_3D,
        save_dir_win=save_dir_10_day_window,
        start_day=1,window_size=10,overlap_size=5)
    
    # #SUB205
    # visualizations.plot_sliding_windows(
    #     days_label,y_label, clf_loaded,
    #     X_csp_features_scaled, X_csp_features_scaled_2d,
    #     X_pca_features_2D,X_umap_features_2D,
    #     X_pca_features_3D,X_umap_features_3D,
    #     save_dir_win=save_dir_10_day_window,
    #     start_day=1,window_size=5,overlap_size=0)

    # #SUB206
    # visualizations.plot_sliding_windows(
    #     days_label,y_label, clf_loaded,
    #     X_csp_features_scaled, X_csp_features_scaled_2d,
    #     X_pca_features_2D,X_umap_features_2D,
    #     X_pca_features_3D,X_umap_features_3D,
    #     save_dir_win=save_dir_10_day_window,
    #     start_day=1,window_size=6,overlap_size=0)



    # # ─── 7) track trajectories of centers

    # visualizations.track_centers(spaces,y_label,days_label, save_dir_centers)

    # # ─── 8) plot sliding window per trials
    
    visualizations.sliding_window_metric_analysis(X_csp_features_scaled, spaces, y_label, days_label, clf_loaded,
                                   metric='AUC', window_size=10, dim_dict=dim_dict, directory=save_dir_trial_window)
    a=5
















    # # ───  Analysis that I didnt use ──────────────────────────────────────
    # original_accuracy, t_max, p_value,significant_days = utils.t_max_permutation_test(X_csp_features_scaled,y_label,days_label,clf_loaded,save_dir)


    # avg_mu_power, avg_beta_power = utils.analyze_psd_over_time(denoised_signal_test_centered, days_label, unique_days, save_dir)
    # metrics = {'Accuracy': accuracies,'Inter-Dist': inter_variances, 'Intra-Idle': intra_variances_idle,'Intra-Motor': intra_variances_motor}
    # bands = {'Mu Power': avg_mu_power,  'Beta Power': avg_beta_power }
    
    # results_t = utils.compare_early_vs_late_all(metrics_data=metrics, band_data=bands, early_end=34, late_start=-34,directory=save_dir) # compare first 34 days vs last 34 days
    # results_anova = utils.compare_phases_all( metrics_data=metrics, band_data=bands, early_end=34, late_start=-34,  directory=save_dir) # compare 3 phases (early, intermediate, late)
    # results_anova_cluster = utils.anova_cluster_distances(inter_variances, intra_variances_idle, intra_variances_motor,num_groups=10, directory=save_dir)

    # visualizations.consistency_of_signals(X_csp_features_scaled, days_label)







    # visualizations.plot_multiple_day_2D_projection(X_pca_features_2D,y_label,days_label,start_test_day,end_test_day,reducer='PCA',directory=save_directory_fig) # relevant (visually) just for couple of days and not the entire dataset
    # visualizations.plot_multiple_day_2D_projection(X_umap_features_2D,y_label,days_label,start_test_day,end_test_day,reducer='UMAP',directory=save_directory_fig)
    # visualizations.plot_multiple_day_3D_projection(X_pca_features_3D, y_label, days_label,start_test_day,end_test_day, reducer='PCA',directory=save_directory_fig)
    # visualizations.plot_multiple_day_3D_projection(X_umap_features_3D, y_label, days_label,start_test_day,end_test_day, reducer='UMAP',directory=save_directory_fig)
    # best_filter_pair_indices, pair_scores = utils.evaluate_csp_filters(denoised_signal_test_centered,y_label,filters)






    # # ───  Similar Analysis for accuracy ──────────────────────────────────────


    visualizations.plot_accuracy_vs_cluster_separation(
          X_csp_features_scaled,X_csp_features_scaled_2d,
          y_label, days_label, clf_loaded,
          start_test_day,end_test_day,dim=2,directory=save_dir)


    visualizations.plot_accuracy_vs_cluster_separation(
        X_csp_features_scaled, X_pca_features_2D,
        y_label, days_label, clf_loaded,
        start_test_day, end_test_day,
        dim=2, reducer='PCA',directory=save_dir
    )



    visualizations.plot_accuracy_vs_cluster_separation(
        X_csp_features_scaled,X_umap_features_2D,
        y_label, days_label, clf_loaded,
        start_test_day,end_test_day,
        dim=2, reducer='UMAP',directory=save_dir
    )
    
    visualizations.plot_accuracy_vs_cluster_separation(
        X_csp_features_scaled,X_pca_features_3D,
        y_label, days_label, clf_loaded,
        start_test_day,end_test_day,
        dim=3, reducer='PCA',directory=save_dir
    )
    
    visualizations.plot_accuracy_vs_cluster_separation(
        X_csp_features_scaled,X_umap_features_3D,
        y_label, days_label, clf_loaded,
        start_test_day,end_test_day,
        dim=3, reducer='UMAP',directory=save_dir
    )

    # also plot CSP-space separation (no dimensionality reduction)
    visualizations.plot_accuracy_vs_cluster_separation(
        X_csp_features_scaled,X_csp_features_scaled,
        y_label, days_label, clf_loaded,
        start_test_day,end_test_day,dim=10,
        directory=save_dir
    )

    # visualizations.main_graph(unique_days, accuracies, inter_variances,start_test_day, end_test_day, save_dir, min_window=2, max_window=15)

    # # smoothed over windows of varying size CSP6D trying to find the best window (5,14,15)
    # smoothed_accuracies_inter, smoothed_inter_variances, smoothed_accuracies_intra_idle, smoothed_intra_variances_idle,  smoothed_accuracies_intra_motor,smoothed_intra_variances_motor  =  visualizations.plot_accuracy_vs_variances_smoothed(
    #     unique_days, accuracies,
    #     inter_variances, intra_variances_idle, intra_variances_motor,
    #     start_test_day, end_test_day, save_dir, min_window=2, max_window=15) # smoothed
    
    # # fixed window (e.g. best inter‐correlation)
    # smoothed_accuracies_inter, smoothed_inter_variances, smoothed_accuracies_intra_idle, smoothed_intra_variances_idle,  smoothed_accuracies_intra_motor,smoothed_intra_variances_motor  = visualizations.plot_accuracy_vs_variances_smoothed(
    #     unique_days, accuracies,
    #     inter_variances, intra_variances_idle, intra_variances_motor,
    #     start_test_day, end_test_day, save_dir, min_window=5, max_window=5) 

    # for i in range(2,16):
    #     smoothed_auc_inter, smoothed_inter_variances, smoothed_auc_idle, smoothed_intra_variances_idle,  smoothed_auc_motor ,smoothed_intra_variances_motor =  visualizations.plot_auc_vs_variances_smoothed(
    #     unique_days, auc_scores,
    #     inter_variances, intra_variances_idle, intra_variances_motor,
    #     start_test_day, end_test_day, save_dir, min_window=i, max_window=i) # smoothed

    # delta_acc_matrix, delta_inter_var_matrix, delta_intra_var_matrix_idle, delta_intra_var_matrix_motor = visualizations.delta_acc_var(X_csp_features_scaled,y_label,days_label,clf_loaded,start_test_day,end_test_day,acc_smoothed, inter_smoothed, intra_idle_smoothed, intra_mi_smoothed,directory=save_dir) 
    
    # utils.analyze_csp_component_usage(clf_loaded, denoised_signal_test_centered, y_label, class_names=['Idle', 'MI'], plot_expected_vs_empirical=True)

    visualizations.track_centers(spaces,y_label,days_label, save_dir_centers)

if __name__ == '__main__':
    main()
