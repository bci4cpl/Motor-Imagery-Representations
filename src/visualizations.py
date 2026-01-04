## visualizations.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm

from matplotlib.ticker import MaxNLocator
import os
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import math

import pandas as pd
import utils
import visualizations

import warnings
import umap
import joblib
import itertools
import mne
from mne.time_frequency import psd_array_multitaper
from scipy.stats import ttest_ind
from scipy.ndimage import uniform_filter1d
from scipy.stats import f_oneway
from sklearn.metrics import pairwise_distances
from scipy import stats
from scipy.stats import linregress, t
from scipy.spatial.distance import cdist,euclidean
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score




def y_lim_cluster_scores(scores):
    ymin, ymax = min(scores), max(scores)
    pad = 0.3*(np.abs(ymax-ymin))
    return ymin-pad, ymax+pad

def graph_limits_2D(X):
    # Calculate global axis limits based on the data
    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    y_min = np.min(X[:, 1])
    y_max = np.max(X[:, 1])

    # Add padding to make plots look nicer
    padding = 2
    x_limits = (x_min - padding, x_max + padding)
    y_limits = (y_min - padding, y_max + padding)
    return x_limits,y_limits

def plot_2d_projection(X, y_label, title,reducer, ax,x_limits, y_limits, days=None):
    # Plot trials for label 0 (Idle)
    ax.scatter(X[y_label == 0, 0], X[y_label == 0, 1], label='Idle', c='blue', marker='o', alpha=0.8)

    # Plot trials for label 1 (Motor Imagery - MI)
    ax.scatter(X[y_label == 1, 0], X[y_label == 1, 1], label='MI', c='red', marker='x', alpha=0.8)

    ax.set_title(title)
    if reducer == 'PCA':
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    if reducer == 'UMAP':
        ax.set_xlabel('UMAP_1')
        ax.set_ylabel('UMAP_2')
    
    if reducer == 'CSP':
        ax.set_xlabel('CSP_1')
        ax.set_ylabel('CSP_2')
    
    ax.legend()

    # Apply the same limits to each plot
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.grid(True)

def plot_metric_cluster_scores(metric_scores, start_test_day, end_test_day,reducer, directory,dim=None, window_size=None, metric=None):

    if metric == 'Davies-Bouldin':
         # higher is better
        metric_scores = np.array(metric_scores, dtype=float)
        metric_scores = 1.0 / metric_scores
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(metric_scores, marker='o', linestyle='-', color='b')  # Line plot with circle markers

    title = f'{dim}D_{reducer} {metric} Scores with Test_Day_{start_test_day}_to_{end_test_day}'
    plt.title(title)

    xticks = np.arange(start_test_day, end_test_day)
    positions = np.arange(len(metric_scores))

    try:
        plt.xticks(ticks=positions, labels=xticks)
    except ValueError:
        # fallback: build a matching labels array of the same length as positions
        fallback_labels = np.arange(start_test_day,
                                    start_test_day+ len(metric_scores))
        # update title if you want
        plt.title(
            f'{dim}D {reducer} {metric} Scores — '
            f'Days {start_test_day} to {end_test_day}'
        )
        plt.xticks(ticks=positions, labels=fallback_labels)   

    plt.xlabel('Days')
    plt.ylabel(f'{metric} Score')

    ymin, ymax = y_lim_cluster_scores(metric_scores)
    plt.ylim([ymin,ymax])
    if metric== "Silhouette":
        plt.ylim([-0.5,0.5])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if not directory == None:
        full_path = os.path.join(directory, title)
        plt.savefig(full_path)
        plt.close()
    plt.close()

def plot_multiple_day_2D_projection(X_features, y_label, days_labels, start_test_day, end_test_day, reducer, directory, window_size=None, sub205=False, sub206=False, show_grid=True, box_off=False):
    unique_days = np.unique(days_labels)
    nrows = 2
    ncols = math.ceil(len(unique_days) / nrows)

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 6))
    axs = axs.flatten()  # Flatten for consistent 1D indexing
    x_lim, y_lim = graph_limits_2D(X_features)

    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []

    reducer_label = 'PC' if reducer == 'PCA' else reducer


    for i, day in enumerate(unique_days):
        day_mask = (days_labels == day)
        X_day = X_features[day_mask]
        y_day = y_label[day_mask]

        # Label logic
        if sub205 or sub206:
            day_label = f"Day {day+3}"
        else:
            day_label = f"Day {start_test_day + i + 30}"

        ax = axs[i]
        ax.scatter(X_day[y_day == 0, 0], X_day[y_day == 0, 1], label='Idle', c='blue', marker='o', alpha=0.8)
        ax.scatter(X_day[y_day == 1, 0], X_day[y_day == 1, 1], label='MI', c='red', marker='x', alpha=0.8)
        ax.set_title(day_label)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        # --- Robust axis label logic ---
        row, col = divmod(i, ncols)
        # Only leftmost
        if col == 0:
            ax.set_ylabel('CSP_2' if reducer == 'CSP' else f'{reducer_label}_2')
        else:
            ax.set_ylabel('')
        # Only bottom row
        if row == nrows - 1:
            ax.set_xlabel('CSP_1' if reducer == 'CSP' else f'{reducer_label}_1')
        else:
            ax.set_xlabel('')
        # --- Grid/Box logic ---
        if show_grid:
            ax.grid(True)
        if box_off:
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Legend (only first subplot)
        if i == 0:
            ax.legend(loc='upper right', fontsize=9, frameon=True)
        else:
            ax.legend().set_visible(False)

        # Cluster quality metrics
        sil_score = silhouette_score(X_day, y_day, metric='euclidean')
        ch_score = calinski_harabasz_score(X_day, y_day)
        db_score = davies_bouldin_score(X_day, y_day)

        silhouette_scores.append(sil_score)
        calinski_harabasz_scores.append(ch_score)
        davies_bouldin_scores.append(db_score)

        # right side text
        ax.text(
                0.98, 0.05,              # position (x, y) in axis coordinates
                f"Sil = {sil_score:.3f}",    # formatted text
                transform=ax.transAxes,  # make it relative to the axes
                fontsize=10, style='italic',
                ha='right', va='bottom', 
                color='dimgray')
        
        # # left side text
        # ax.text(
        #         0.02, 0.05,              # position (x, y) in axis coordinates
        #         f"CH = {ch_score:.3f}",    # formatted text
        #         transform=ax.transAxes,  # make it relative to the axes
        #         fontsize=10, style='italic',
        #         ha='left', va='bottom', 
        #         color='dimgray')

    # Hide any extra/unused subplots
    for j in range(len(unique_days), len(axs)):
        axs[j].axis('off')

    # Label logic
    if sub205 or sub206:
        start_test_day = start_test_day + 3
        end_test_day = end_test_day + 3
    else:
        start_test_day = start_test_day + 30
        end_test_day = end_test_day + 30
        
    fig.suptitle(f"{reducer} 2D Feature Space Across Days {start_test_day}–{end_test_day}", fontsize=15)

    plt.tight_layout()
    prefix = "2D"
    filename = f'{prefix}_{reducer}_Test_Day_{start_test_day}_to_{end_test_day}.jpg'

    subdir = os.path.join(directory, prefix)
    os.makedirs(subdir, exist_ok=True)

    full_path = os.path.join(subdir, filename)
    plt.savefig(full_path, dpi=300)
    plt.close()

    # Plot metrics separately
    plot_metric_cluster_scores(silhouette_scores, start_test_day, end_test_day, reducer, subdir, dim=2, window_size=window_size, metric="Silhouette")
    plot_metric_cluster_scores(calinski_harabasz_scores, start_test_day, end_test_day, reducer, subdir, dim=2, window_size=window_size, metric="Calinski-Harabasz")
    plot_metric_cluster_scores(davies_bouldin_scores, start_test_day, end_test_day, reducer, subdir, dim=2, window_size=window_size, metric="Davies-Bouldin")


def graph_limits_3D(X):
    # Calculate global axis limits based on the 3D data
    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    y_min = np.min(X[:, 1])
    y_max = np.max(X[:, 1])
    z_min = np.min(X[:, 2])
    z_max = np.max(X[:, 2])

    # Add padding to make plots look nicer
    padding = 2
    x_limits = (x_min - padding, x_max + padding)
    y_limits = (y_min - padding, y_max + padding)
    z_limits = (z_min - padding, z_max + padding)
    return x_limits, y_limits, z_limits

def plot_3d_projection(X, y_label, title, ax, reducer, x_limits, y_limits, z_limits, days=None):

    # Plot trials for label 0 (Idle)
    ax.scatter(X[y_label == 0, 0], X[y_label == 0, 1], X[y_label == 0, 2], label='Idle', c='blue', marker='o', alpha=0.8)

    # Plot trials for label 1 (Motor Imagery - MI)
    ax.scatter(X[y_label == 1, 0], X[y_label == 1, 1], X[y_label == 1, 2], label='MI', c='red', marker='x', alpha=0.8)

    ax.set_title(title)
    if reducer == 'PCA':
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

    if reducer == 'UMAP':
        ax.set_xlabel('UMAP_1')
        ax.set_ylabel('UMAP_2')
        ax.set_zlabel('UMAP_3')

    # Apply the same limits to each plot
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_zlim(z_limits)

    ax.legend()
    ax.grid(True)

    
def _rot_z(a):
    c,s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def _rot_x(e):
    c,s = np.cos(e), np.sin(e)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

import numpy as np
from sklearn.metrics import silhouette_score  # you already import the others

def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def _rot_x(e):
    c, s = np.cos(e), np.sin(e)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])

def best_view_angles(X3, y, elevs=None, azims=None):
    if elevs is None:
        elevs = np.deg2rad([10, 20, 30])
    if azims is None:
        azims = np.deg2rad(np.arange(-180, 181, 15))

    best_e, best_a, best_score = None, None, -np.inf

    # (Optional) speed-up: uniformly sample if very large
    if len(X3) > 5000:
        idx = np.random.choice(len(X3), 5000, replace=False)
        X3s, ys = X3[idx], y[idx]
    else:
        X3s, ys = X3, y

    for e in elevs:
        Rx = _rot_x(e)
        for a in azims:
            R = Rx @ _rot_z(a)
            XY = (X3s @ R.T)[:, :2]
            # Silhouette requires at least 2 labels; guard just in case
            try:
                sil = silhouette_score(XY, ys, metric='euclidean')
            except Exception:
                sil = -np.inf
            if sil > best_score:
                best_e, best_a, best_score = e, a, sil

    return np.rad2deg(best_e), np.rad2deg(best_a)


def plot_multiple_day_3D_projection(X_features, y_label, days_labels, start_test_day, end_test_day, reducer, directory, window_size=None, sub205=False, sub206=False, show_grid=True, box_off=False, auto_view=False):

    unique_days = np.unique(days_labels)
    nrows = 2
    ncols = math.ceil(len(unique_days) / nrows)   # <-- use ceil so grid has enough slots
    fig = plt.figure(figsize=(15, 10))

    # Calculate the global axis limits for all days
    x_lim, y_lim, z_lim = graph_limits_3D(X_features)

    silhouette_scores = []
    calinski_harabasz_scores =[]
    davies_bouldin_scores = []

    reducer_label = 'PC' if reducer == 'PCA' else reducer
    # -------- Auto view selection --------
    default_view = (20, -60)  # elev, azim fallback
    global_view = None
    if auto_view is True or auto_view == 'global':
        mask = np.isin(days_labels, unique_days)
        elev_deg, azim_deg = best_view_angles(X_features[mask], y_label[mask])
        global_view = (elev_deg, azim_deg)

    # Iterate over each day
    for i, day in enumerate(unique_days):
        day_mask = (days_labels == day)
        X_day = X_features[day_mask]
        y_day = y_label[day_mask]

        # Create 3D subplot for each day
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        ax.scatter(X_day[y_day == 0, 0], X_day[y_day == 0, 1], X_day[y_day == 0, 2],
                   label='Idle', c='blue', marker='o', alpha=0.8)
        ax.scatter(X_day[y_day == 1, 0], X_day[y_day == 1, 1], X_day[y_day == 1, 2],
                   label='MI', c='red', marker='x', alpha=0.8)

        # Title: just Day N
        if sub205 or sub206:
            day_label = f"Day {day+3}"
        else:
            day_label = f"Day {start_test_day + i + 30}"

        ax.set_title(day_label)
    
        # Limits
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)

        # Axis labels: only bottom row for x, leftmost col for y and z
        if row == nrows - 1:
            ax.set_xlabel(f"{reducer_label}_1")
        else:
            ax.set_xlabel("")
        if col == 0:
            ax.set_ylabel(f"{reducer_label}_2")
            ax.set_zlabel(f"{reducer_label}_3")
        else:
            ax.set_ylabel("")
            ax.set_zlabel("")

        # Grid/box options
        if show_grid:
            ax.grid(True)
        if box_off:
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Legend only in first subplot
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.legend().set_visible(False)

        # Calculate silhouette score for the current day
        sil_score = silhouette_score(X_day, y_day, metric='euclidean')
        ch_score = calinski_harabasz_score(X_day, y_day)
        db_score = davies_bouldin_score(X_day, y_day)

        silhouette_scores.append(sil_score)
        calinski_harabasz_scores.append(ch_score)
        davies_bouldin_scores.append(db_score)

        
                # ---- View ----
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type('ortho')
        if auto_view == 'per_day':
            elev_deg, azim_deg = best_view_angles(X_day, y_day)
            ax.view_init(elev=elev_deg, azim=azim_deg)
        elif global_view is not None:
            ax.view_init(*global_view)
        else:
            ax.view_init(*default_view)
        
        # Label logic
    if sub205 or sub206:
        start_test_day = start_test_day + 3
        end_test_day = end_test_day + 3
    else:
        start_test_day = start_test_day + 30
        end_test_day = end_test_day + 30

    plt.tight_layout()
    fig.suptitle(f"{reducer} 3D Feature Space Across Days {start_test_day}–{end_test_day}", fontsize=15)

    prefix = "3D"

    filename = f'{prefix}_{reducer}_Test_Day_{start_test_day}_to_{end_test_day}.jpg'
    subdir = os.path.join(directory, prefix)
    os.makedirs(subdir, exist_ok=True)
    full_path = os.path.join(subdir, filename)
    plt.savefig(full_path)
    plt.close()  # Close the figure after saving

    # Plot silhouette scores
    plot_metric_cluster_scores(silhouette_scores, start_test_day, end_test_day, reducer, subdir,dim=3, window_size=window_size, metric="Silhouette")
    plot_metric_cluster_scores(calinski_harabasz_scores, start_test_day, end_test_day, reducer, subdir,dim=3,window_size=window_size,metric = "Calinski-Harabasz")
    plot_metric_cluster_scores(davies_bouldin_scores, start_test_day, end_test_day, reducer, subdir,dim=3,window_size=window_size,metric = "Davies-Bouldin")


def plot_sliding_windows(days_label, y_label, clf, X_csp_scaled, X_csp_2d, X_pca2d, X_umap2d, X_pca3d, X_umap3d, save_dir_win, start_day=1, window_size=10, overlap_size=5):
    """
    For each sliding window of `window_size` days (with `overlap_size` overlap),
    slice your precomputed feature arrays and call both:
      - plot_multiple_day_2D_projection
      - plot_accuracy_vs_cluster_separation

    Each window’s plots go into: save_dir/10_day_window_{start}_{end-1}/
    """
    max_day = int(days_label.max())
    start_day = 1 
    step = window_size - overlap_size

    for start_day in range(start_day, max_day, step):
        end_day = min(start_day + window_size -1 , max_day)
        print(start_day,end_day)
    
        if end_day - start_day < overlap_size:
            print(start_day,end_day)
            print("the loop is over")
            break

        mask = (days_label >= start_day) & (days_label <= end_day)  
        print(mask)
        if not mask.any():
            continue

        # slice
        X_win      = X_csp_scaled[mask]
        X2d_win    = X_csp_2d[mask]
        P2d_win    = X_pca2d[mask]
        U2d_win    = X_umap2d[mask]
        P3d_win    = X_pca3d[mask]
        U3d_win    = X_umap3d[mask]
        y_win      = y_label[mask]
        days_win   = days_label[mask]

        # make subfolder
        day_folder_name = f'Test_Day_{start_day+3}_to_{end_day+3}'
        day_folder_path = os.path.join(save_dir_win, day_folder_name)
        os.makedirs(day_folder_path, exist_ok=True)

        # # 1) 2D projections
        plot_multiple_day_2D_projection(
            X2d_win, y_win, days_win,
            start_day, end_day, reducer='CSP', directory=day_folder_path,
            window_size=window_size, sub206=True
        )
        plot_multiple_day_2D_projection(
            P2d_win, y_win, days_win,
            start_day, end_day, reducer='PCA', directory=day_folder_path,
            window_size=window_size, sub206=True

        )
        plot_multiple_day_2D_projection(
            U2d_win, y_win, days_win,
            start_day, end_day, reducer='UMAP', directory=day_folder_path,
            window_size=window_size, sub206=True

        )

        # # 2) 3D projections
        plot_multiple_day_3D_projection(
            P3d_win, y_win, days_win,
            start_day, end_day, reducer='PCA', directory=day_folder_path,
            window_size=window_size, sub206=True, auto_view='global'

        )
        plot_multiple_day_3D_projection(
            U3d_win, y_win, days_win,
            start_day, end_day, reducer='UMAP', directory=day_folder_path,
            window_size=window_size, sub206=True, auto_view='global'

        )

        # # 3) Accuracy vs cluster separation
        # for (Xr, dim, red) in [ 
        #     (X_win, 2, 'CSP'),
        #     (X_win, 2, 'PCA'),
        #     (X_win, 3, 'PCA'),
        #     (X_win, 2, 'UMAP'),
        #     (X_win, 3, 'UMAP'),
        # ]:
        #     # select the matching reduced array
        #     Xred = {
        #         ('CSP',2): X2d_win,
        #         ('PCA',2): P2d_win,
        #         ('PCA',3): P3d_win,
        #         ('UMAP',2): U2d_win,
        #         ('UMAP',3): U3d_win
        #     }[(red, dim)]

        #     corr_accuracy_inter, corr_accuracy_intra_idle, corr_accuracy_intra_motor, corr_accuracy_silhouette = plot_accuracy_vs_cluster_separation(
        #         Xr, Xred,
        #         y_win, days_win, clf,
        #         start_day, end_day,
        #         dim=dim, reducer=red,
        #         directory=day_folder_path
        #     )

        #     utils.log_metric_correlations(
        #         metric_name='accuracy',
        #         directory=save_dir_win,
        #         start_test_day=start_day,
        #         end_test_day=end_day,
        #         reducer=red,
        #         dim=dim,
        #         corr_inter=corr_accuracy_inter,
        #         corr_intra_idle=corr_accuracy_intra_idle,
        #         corr_intra_motor=corr_accuracy_intra_motor
        #     )
            
        #     corr_auc_inter, corr_auc_intra_idle, corr_auc_intra_motor = plot_auc_vs_cluster_separation(
        #         Xr, Xred,
        #         y_win, days_win, clf,
        #         start_day, end_day,
        #         dim=dim, reducer=red,
        #         directory=day_folder_path
        #     )

        #     utils.log_metric_correlations(
        #         metric_name='AUC',
        #         directory=save_dir_win,
        #         start_test_day=start_day,
        #         end_test_day=end_day,
        #         reducer=red,
        #         dim=dim,
        #         corr_inter=corr_auc_inter,
        #         corr_intra_idle=corr_auc_intra_idle,
        #         corr_intra_motor=corr_auc_intra_motor
        #     )



        # corr_accuracy_inter, corr_accuracy_intra_idle, corr_accuracy_intra_motor= plot_accuracy_vs_cluster_separation(
        #         X_win,X_win, y_win, days_win, clf,
        #         start_day, end_day,
        #         directory=day_folder_path)
        
        # utils.log_metric_correlations(
        #         metric_name='accuracy',
        #         directory=save_dir_win,
        #         start_test_day=start_day,
        #         end_test_day=end_day,
        #         reducer='CSP',
        #         dim=6,
        #         corr_inter=corr_accuracy_inter,
        #         corr_intra_idle=corr_accuracy_intra_idle,
        #         corr_intra_motor=corr_accuracy_intra_motor
        #     )


        # corr_auc_inter, corr_auc_intra_idle, corr_auc_intra_motor = plot_auc_vs_cluster_separation(X_win,X_win, y_win, days_win, clf,
        #         start_day, end_day,
        #         directory=day_folder_path)
        
        # utils.log_metric_correlations(
        #     metric_name='AUC',
        #     directory=save_dir_win,
        #     start_test_day=start_day,
        #     end_test_day=end_day,
        #     reducer='CSP',
        #     dim=6,
        #     corr_inter=corr_auc_inter,
        #     corr_intra_idle=corr_auc_intra_idle,
        #     corr_intra_motor=corr_auc_intra_motor
        # )

        # print(f"Finished window {start_day+30}–{end_day+30}")

def main_graph(unique_days, accuracies, inter_variances,start_test_day, end_test_day, directory, min_window=2, max_window=15):

    # Find the best window size for inter- and intra-cluster variances separately
    best_window_size_inter, best_corr_accuracy_inter, p_value_inter = utils.find_best_window_size(accuracies, inter_variances, min_window=min_window, max_window=max_window)
    smoothed_accuracies_inter = np.convolve(accuracies, np.ones(best_window_size_inter) / best_window_size_inter, mode='valid')
    smoothed_inter_variances = np.convolve(inter_variances, np.ones(best_window_size_inter) / best_window_size_inter, mode='valid')

    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.set_xlabel('Days', fontsize=14, labelpad=-10)
    ax1.set_ylabel('AUC', color='tab:blue',fontsize=14)
    ax1.plot(unique_days[:len(smoothed_accuracies_inter)], smoothed_accuracies_inter, label='Smoothed Classifier Accuracy', marker='o', color='tab:blue', linewidth=2, markersize=6)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(visible=True, linestyle='--', linewidth=0.5) 

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel("Inter-Cluster [Cohen's d]", color='tab:red')
    ax1_twin.plot(unique_days[:len(smoothed_inter_variances)], smoothed_inter_variances, label='Smoothed Inter-Cluster Distance', marker='x', color='tab:red', linewidth=2, markersize=6)
    ax1_twin.tick_params(axis='y', labelcolor='tab:red')


    ax1_twin.legend([f"Inter-Cluster (corr: {best_corr_accuracy_inter:.2f}, window: {best_window_size_inter})"], loc='upper right', fontsize=12)
    ax1.set_title(f"AUC vs Inter-Cluster Distance Over Days {start_test_day}-{end_test_day-1}", fontsize=16)
    xticks = ax1.get_xticks().astype(int)
    ax1.set_xticklabels(xticks + 30, fontsize=12)

    fig.tight_layout()
    filename = os.path.join(directory, f'Enhanced_Main_Graph_{start_test_day}_to_{end_test_day - 1}.jpg')
    full_path = os.path.join(directory, filename)
    plt.savefig(full_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_accuracy_vs_cluster_separation_old(X_csp, X_reduced, y_label, days_labels, clf_loaded, start_test_day, end_test_day, dim=6, reducer='CSP', directory='None', layout= '4x1'):
    unique_days = np.unique(days_labels)
    accuracies = []
    inter_variances = []
    intra_variances_idle = []
    intra_variances_motor = []
    silhouette_scores = []

    lda_loaded = clf_loaded.named_steps['LDA']

    for day in unique_days:
        day_mask = (days_labels == day)
        X_day = X_csp[day_mask]
        y_day = y_label[day_mask]

        accuracy = utils.evaluate_classifier(lda_loaded, X_day, y_day)
        accuracies.append(accuracy)

        cluster_variance = utils.calculate_cluster_variance(X_reduced[day_mask], y_day)
        intra_variances_idle.append(cluster_variance['intra_distances']['idle'])
        intra_variances_motor.append(cluster_variance['intra_distances']['motor_imagery'])
        inter_variances.append(cluster_variance['inter_distance'])
        silhouette_scores.append(silhouette_score(X_reduced[day_mask], y_day))

    # Calculate correlation coefficients
    corr_accuracy_inter, p_val_inter = pearsonr(accuracies, inter_variances)
    corr_accuracy_intra_idle, p_val_intra_idle = pearsonr(accuracies, intra_variances_idle)
    corr_accuracy_intra_motor, p_val_intra_motor = pearsonr(accuracies, intra_variances_motor)
    corr_accuracy_silhouette, pval_silhouette = pearsonr(accuracies, silhouette_scores)

    # Layout
    if layout == '4x1':
        fig, axs = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
        axs = axs.flatten()
    elif layout == '2x2':
        fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex='col')
        axs = axs.flatten()
    else:
        raise ValueError("layout must be '4x1' or '2x2'")
    
    # adjustments for sub201
    start_test_day += 30
    end_test_day += 30
    label_days = 30  # offset labels by +30


    # adjustments for sub205
    # start_test_day += 1
    # end_test_day -= 1
    # label_days = unique_days + 3  # offset labels by +3 


    # adjustments for sub206
    # start_test_day += 1
    # label_days = unique_days + 3  # offset labels by +3 
    # # dim=10 for csp

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)  # Tighter height of 12

    plt.subplots_adjust(hspace=0.25)

    # First subplot: Accuracy vs Inter-Cluster Distance
    ax1.set_xlabel('Days', labelpad=-10)
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(unique_days, accuracies, label='Classifier Accuracy', marker='o', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel("Inter-Cluster Distance [Cohen's d]", color='tab:red')
    ax1_twin.plot(unique_days, inter_variances, label="Inter-Cluster Distance [Cohen's d]", marker='x', color='tab:red')
    ax1_twin.tick_params(axis='y', labelcolor='tab:red')

    ax1_twin.legend([f"Inter-Cluster Distance (corr: {corr_accuracy_inter:.2f})"], loc='upper right')
    ax1.set_title(f"Accuracy vs Inter-Cluster Distance in {dim}D {reducer} space Over Days {start_test_day}-{end_test_day}")

    # Second subplot: Accuracy vs Intra-Cluster Distance (Idle)
    ax2.set_xlabel('Days', labelpad=-10)
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(unique_days, accuracies, label='Classifier Accuracy', marker='o', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('Intra-Cluster Distance Idle [std]', color='tab:green')
    ax2_twin.plot(unique_days, intra_variances_idle, label='Intra-Cluster Distance Idle [std]', marker='x', color='tab:green')
    ax2_twin.tick_params(axis='y', labelcolor='tab:green')

    ax2_twin.legend([f"Intra-Cluster Variance Idle (corr: {corr_accuracy_intra_idle:.2f})"], loc='upper right')
    ax2.set_title(f'Accuracy vs Intra-Cluster Variance (Idle) in {dim}D {reducer} space Over Days {start_test_day}-{end_test_day}')

    # Third subplot: Accuracy vs Intra-Cluster Distance (Motor Imagery)
    ax3.set_xlabel('Days', labelpad=-10)
    ax3.set_ylabel('Accuracy', color='tab:blue')
    ax3.plot(unique_days, accuracies, label='Classifier Accuracy', marker='o', color='tab:blue')
    ax3.tick_params(axis='y', labelcolor='tab:blue')

    ax3_twin = ax3.twinx()
    ax3_twin.set_ylabel('Intra-Cluster Variance Motor Imagery [std]', color='tab:purple')
    ax3_twin.plot(unique_days, intra_variances_motor, label='Intra-Cluster Distance Motor Imagery [std]', marker='x', color='tab:purple')
    ax3_twin.tick_params(axis='y', labelcolor='tab:purple')

    ax3_twin.legend([f"Intra-Cluster Variance Motor Imagery (corr: {corr_accuracy_intra_motor:.2f})"], loc='upper right')
    ax3.set_title(f'Accuracy vs Intra-Cluster Variance (Motor Imagery) in {dim}D {reducer} space Over Days {start_test_day}-{end_test_day}')



    # Adjust x-axis ticks for all subplots
    for ax in [ax1, ax2, ax3]:
        xticks = ax.get_xticks().astype(int) #for sub 201
        ax.set_xticklabels(xticks+30) #for multiple graphs 10 days FOE SUB 201

        # ax.set_xticks(unique_days)# for multiple graphs 10 days sub 205
        # ax.set_xticklabels([str(day) for day in label_days]) # for multiple graphs 10 days sub 205

    # Adjust layout and save
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the tight_layout to leave space for title
    filename = f'Accuracy vs Variance in {dim}D {reducer} space_Test_Day_{start_test_day}_to_{end_test_day}.jpg'
    full_path = os.path.join(directory, filename)
    plt.savefig(full_path, bbox_inches='tight', dpi=300)

    plt.close()

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # First subplot: Inter-cluster distance vs accuracy
    ax1.scatter(inter_variances, accuracies, c='r', label=f"Inter-Cluster Distance (r={corr_accuracy_inter:.2f})", marker='o')
    ax1.set_title('Accuracy vs. Inter-Cluster Distance')
    ax1.set_xlabel('Inter-Cluster Distance')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True)

    # Second subplot: Intra-cluster idle distance vs accuracy
    ax2.scatter(intra_variances_idle, accuracies, c='g', label=f'Idle (r={corr_accuracy_intra_idle:.2f})', marker='o')
    ax2.set_title('Accuracy vs. Intra-Cluster Idle Variance')
    ax2.set_xlabel('Intra-Cluster Variance (Idle)')
    ax2.legend(loc='lower right')
    ax2.grid(True)

    # Third subplot: Intra-cluster motor imagery distance vs accuracy
    ax3.scatter(intra_variances_motor, accuracies, c='purple', label=f'Motor Imagery (r={corr_accuracy_intra_motor:.2f})', marker='o')
    ax3.set_title('Accuracy vs. Intra-Cluster Motor Imagery Variance')
    ax3.set_xlabel('Intra-Cluster Variance (Motor Imagery)')
    ax3.legend(loc='lower right')
    ax3.grid(True)

    # Adjust layout to ensure everything fits well
    plt.tight_layout()
    new_filename = f'Correlation Analysis_{filename}'
    full_path = os.path.join(directory, new_filename)
    plt.savefig(full_path)
    plt.close()

    return corr_accuracy_inter, corr_accuracy_intra_idle, corr_accuracy_intra_motor





def plot_accuracy_vs_cluster_separation(
    X_csp, X_reduced, y_label, days_labels, clf_loaded, 
    start_test_day, end_test_day, 
    dim=6, reducer='CSP', directory='None'
):
 
    unique_days = np.unique(days_labels)
    accuracies = []
    inter_variances = []
    intra_variances_idle = []
    intra_variances_motor = []
    silhouette_scores = []

    lda_loaded = clf_loaded.named_steps['LDA']

    for day in unique_days:
        day_mask = (days_labels == day)
        X_day = X_csp[day_mask]
        y_day = y_label[day_mask]

        accuracy = utils.evaluate_classifier(lda_loaded, X_day, y_day)
        accuracies.append(accuracy)

        cluster_variance = utils.calculate_cluster_variance(X_reduced[day_mask], y_day)
        intra_variances_idle.append(cluster_variance['intra_distances']['idle'])
        intra_variances_motor.append(cluster_variance['intra_distances']['motor_imagery'])
        inter_variances.append(cluster_variance['inter_distance'])

        silhouette_scores.append(silhouette_score(X_reduced[day_mask], y_day))

    # Calculate correlation coefficients and p-values
    corr_accuracy_inter, pval_inter = pearsonr(accuracies, inter_variances)
    corr_accuracy_intra_idle, pval_intra_idle = pearsonr(accuracies, intra_variances_idle)
    corr_accuracy_intra_motor, pval_intra_motor = pearsonr(accuracies, intra_variances_motor)
    corr_accuracy_silhouette, pval_silhouette = pearsonr(accuracies, silhouette_scores)

    # --- FIRST PANEL: Line Plots ---
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), constrained_layout=True)
    plt.subplots_adjust(hspace=0.25)

    metrics = [
        (inter_variances, 'ICD', 'tab:red', corr_accuracy_inter, pval_inter),
        (intra_variances_idle, 'ICV Idle', 'tab:green', corr_accuracy_intra_idle, pval_intra_idle),
        (intra_variances_motor, 'ICV MI', 'tab:purple', corr_accuracy_intra_motor, pval_intra_motor),
        (silhouette_scores, 'Silhouette', 'orange', corr_accuracy_silhouette, pval_silhouette),
    ]

    for idx, (y2, label, color, corr, pval) in enumerate(metrics):
        ax = axs[idx]
        ax.set_ylabel('Accuracy', color='tab:blue')
        ax.plot(unique_days, accuracies, marker='o', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax.twinx()
        ax2.set_ylabel(label, color=color)
        ax2.plot(unique_days, y2, marker='x', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend([f"{label} (r={corr:.2f}, p={pval:.3f})"], loc='upper right')

        # Only add x-label to the bottom plot
        if idx == 3:
            ax.set_xlabel('Days')
        else:
            ax.set_xlabel('')
            ax2.set_xlabel('')

        # No individual subplot titles

        # X-tick label shift (your offset)
        xticks = ax.get_xticks().astype(int)
        ax.set_xticklabels(xticks + 30)

    # Main figure title
    fig.suptitle(f"Accuracy and Cluster Metrics in {dim}D {reducer} space\nDays {start_test_day}-{end_test_day}", fontsize=16)

    filename = f'Accuracy_vs_Variance_in_{dim}D_{reducer}_space_Test_Day_{start_test_day}_to_{end_test_day}.jpg'
    full_path = os.path.join(directory, filename)
    plt.savefig(full_path, bbox_inches='tight', dpi=300)
    plt.close()

    # --- SECOND PANEL: Scatter Plots ---
    fig, axs = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
    metrics_data = [
        (inter_variances, 'ICD', 'r', corr_accuracy_inter, pval_inter),
        (intra_variances_idle, 'ICV Idle', 'g', corr_accuracy_intra_idle, pval_intra_idle),
        (intra_variances_motor, 'ICV MI', 'purple', corr_accuracy_intra_motor, pval_intra_motor),
        (silhouette_scores, 'Silhouette', 'orange', corr_accuracy_silhouette, pval_silhouette),
    ]

    for idx, (xvals, label, color, corr, pval) in enumerate(metrics_data):
        axs[idx].scatter(xvals, accuracies, c=color, label=f"{label} (r={corr:.2f}, p={pval:.3f})", marker='o')
        axs[idx].set_xlabel(label)
        axs[idx].legend(loc='lower right')
        axs[idx].grid(True)
        if idx == 0:
            axs[idx].set_ylabel('Accuracy')
        else:
            axs[idx].set_ylabel('')

    fig.suptitle(f"Accuracy vs. Cluster Metrics in {dim}D {reducer} space\nDays {start_test_day + 30}-{end_test_day + 30}", fontsize=16)
    new_filename = f'Correlation Analysis_scatter{filename}'
    full_path = os.path.join(directory, new_filename)
    plt.savefig(full_path, bbox_inches='tight', dpi=300)
    plt.close()

    return corr_accuracy_inter, corr_accuracy_intra_idle, corr_accuracy_intra_motor, corr_accuracy_silhouette



def plot_auc_vs_cluster_separation(X_csp, X_reduced, y_label, days_labels, clf_loaded, start_test_day, end_test_day, dim=6, reducer='CSP', label_offset=0, start_adj=0, end_adj=0,smooth=False, window=5, directory='None'):
    unique_days = np.unique(days_labels)
    adjusted_start_day = start_test_day + start_adj
    adjusted_end_day = end_test_day + end_adj
    aucs = []
    inter_variances = []
    intra_variances_idle = []
    intra_variances_motor = []
    lda_loaded = clf_loaded.named_steps['LDA']

    for day in unique_days:
        day_mask = (days_labels == day)
        X_day = X_csp[day_mask]
        y_day = y_label[day_mask]

        scores_day = lda_loaded.decision_function(X_day)
        auc = roc_auc_score(y_day, scores_day)
        aucs.append(auc)

        cluster_variance = utils.calculate_cluster_variance(X_reduced[day_mask], y_day)
        intra_variances_idle.append(cluster_variance['intra_distances']['idle'])
        intra_variances_motor.append(cluster_variance['intra_distances']['motor_imagery'])
        inter_variances.append(cluster_variance['inter_distance'])



    # Convert to numpy arrays for easier handling
    aucs = np.array(aucs)
    inter_variances = np.array(inter_variances)
    intra_variances_idle = np.array(intra_variances_idle)
    intra_variances_motor = np.array(intra_variances_motor)

    # Apply smoothing only if meaningful
    if smooth and window >= 1:
        kernel = np.ones(window) / window
        aucs = np.convolve(aucs, kernel, mode='valid')
        inter_variances = np.convolve(inter_variances, kernel, mode='valid')
        intra_variances_idle = np.convolve(intra_variances_idle, kernel, mode='valid')
        intra_variances_motor = np.convolve(intra_variances_motor, kernel, mode='valid')

        plot_days = unique_days[:len(aucs)]
        title_prefix = "Smoothed "
        filename_prefix = "Smoothed_"
    else:
        plot_days = unique_days
        title_prefix = ""
        filename_prefix = ""



    # Correlations
    corr_auc_inter, p_auc_inter = pearsonr(aucs, inter_variances)
    corr_auc_intra_idle, p_auc_intra_idle = pearsonr(aucs, intra_variances_idle)
    corr_auc_intra_motor, p_auc_intra_motor = pearsonr(aucs, intra_variances_motor)

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)  # Tighter height of 12
    plt.subplots_adjust(hspace=0.25)

    fig.suptitle(
            f"{title_prefix}AUC vs. Cluster-Separation Metrics in {dim}D {reducer} Space",
            fontsize=16
        )

    # Helper function to plot each metric
    def plot_metric(ax, metric_data, metric_name, color, corr, p_val):
        ax.set_xlabel('Days', labelpad=3)
        ax.set_ylabel('AUC', color='tab:blue')
        ax.plot(plot_days, aucs, label='AUC', marker='o', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')

        ax_twin = ax.twinx()
        ax_twin.set_ylabel(metric_name, color=color)
        ax_twin.plot(plot_days, metric_data, label=metric_name, marker='x', color=color)
        ax_twin.tick_params(axis='y', labelcolor=color)
        
        # P-value text {format_p(p_auc_inter)}, {p_to_stars(p_auc_inter)
        txt = f"r = {corr:.2f}, p = {format_p(p_val)}, {p_to_stars(p_val)}"
        if smooth and window > 1:
            txt += f", w = {window}"
        ax_twin.text(0.98, 0.04, txt, transform=ax_twin.transAxes, ha='right', va='bottom',
                     bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'), fontsize=10)
        ax.set_title(f"AUC vs {metric_name.split('[')[0]}", fontsize=12)
        return ax, ax_twin

    # Plot the 3 graphs
    ax1, ax1_twin = plot_metric(ax1, inter_variances, "Inter-Cluster Distance [Cohen's d]", 'tab:red', corr_auc_inter, p_auc_inter)
    ax2, ax2_twin = plot_metric(ax2, intra_variances_idle, 'Intra-Cluster Variance Idle [std]', 'tab:green', corr_auc_intra_idle, p_auc_intra_idle)
    ax3, ax3_twin = plot_metric(ax3, intra_variances_motor, 'Intra-Cluster Variance Motor Imagery [std]', 'tab:purple', corr_auc_intra_motor, p_auc_intra_motor)

# --- HANDLE OFFSETS ---
    # Sub 201 Logic: Use Index + 30
    if label_offset == 30: 
        for ax in [ax1, ax2, ax3]:
             xticks = ax.get_xticks().astype(int)
             ax.set_xticklabels(xticks + label_offset)
    # Sub 205/206 Logic: Use Value + 3
    else: 
        for ax in [ax1, ax2, ax3]:
             ax.set_xticks(unique_days)
             ax.set_xticklabels(unique_days + label_offset) 

    # Optional: nicer grid (light majors + faint minors)        
    for ax in [ax1, ax1_twin, ax2, ax2_twin, ax3, ax3_twin]:
        ax.grid(True, which='major', axis='both', linestyle='-', linewidth=0.6, alpha=0.25)
        ax.grid(True, which='minor', axis='both', linestyle=':', linewidth=0.4, alpha=0.15)
        ax.minorticks_on()

    panel_letters = ['A', 'B', 'C']
    for ax, letter in zip([ax1, ax2, ax3], panel_letters):
        ax.text(-0.03, 1.02, letter,            # left of x=0, above y=1
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=13, fontweight='bold',
                clip_on=False)   

    if directory != 'None':
        filename = f'{filename_prefix}AUC vs Variance in {dim}D {reducer} space_Test_Day_{adjusted_start_day}_to_{adjusted_end_day}.jpg'
        full_path = os.path.join(directory, filename)
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Inter-cluster distance vs aucs
    scatter_with_fit(
        ax1, inter_variances, aucs,
        color='tab:red',
        title='ROC AUC vs Inter-Cluster Distance',
        xlabel="Inter-Cluster Distance [Cohen's d]",
        ylabel='ROC AUC',
        alpha_sig=0.05
    )

    # Intra-cluster idle distance vs aucs
    scatter_with_fit(
        ax2, intra_variances_idle, aucs,
        color='tab:green',
        title='ROC AUC vs Intra-Cluster Idle Variance',
        xlabel='Intra-Cluster Variance Idle [std]',
        ylabel='ROC AUC',
        alpha_sig=0.05
    )

    # Intra-cluster motor imagery distance vs aucs
    scatter_with_fit(
        ax3, intra_variances_motor, aucs,
        color='tab:purple',
        title='ROC AUC vs Intra-Cluster MI Variance',
        xlabel='Intra-Cluster Variance MI [std]',
        ylabel='ROC AUC',
        alpha_sig=0.05
    )

    # Override symmetric limits (scatter_with_fit centers around 0; use data-driven bounds here)
    for axi, (xv, yv) in zip(
        [ax1, ax2, ax3],
        [(inter_variances, aucs),
        (intra_variances_idle, aucs),
        (intra_variances_motor, aucs)]
    ):
        x_min, x_max = np.nanmin(xv), np.nanmax(xv)
        y_min, y_max = np.nanmin(yv), np.nanmax(yv)
        xr = (x_max - x_min) or 1.0
        yr = (y_max - y_min) or 1.0
        axi.set_xlim(x_min - 0.05*xr, x_max + 0.05*xr)
        axi.set_ylim(y_min - 0.05*yr, y_max + 0.05*yr)

    # Optional: nicer grid (light majors + faint minors)
    for axi in [ax1, ax2, ax3]:
        axi.grid(True, which='major', axis='both', linestyle='-', linewidth=0.6, alpha=0.25)
        axi.grid(True, which='minor', axis='both', linestyle=':', linewidth=0.4, alpha=0.15)
        axi.minorticks_on()

    plt.tight_layout()
    if directory != 'None':
                new_filename = f'{filename_prefix}Correlation_Analysis_{filename}'
                full_path = os.path.join(directory, new_filename)
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()         
    return corr_auc_inter, corr_auc_intra_idle, corr_auc_intra_motor


def sliding_window_metric_analysis(X_csp, X_reduced_dict, y_label, days_labels, clf_loaded,
                                   metric='AUC', window_size=10, dim_dict=None, directory=None):
    """
    Perform sliding-window analysis with flexibility for metric (AUC or accuracy)
    and multiple reduced spaces.

    X_reduced_dict: Dict of reduced spaces, e.g. {
        'CSP': X_csp_scaled,
        'PCA2D': X_pca2d,
        'PCA3D': X_pca3d,
        'UMAP2D': X_umap2d,
        'UMAP3D': X_umap3d
    }
    dim_dict:     Dict of dims per reducer, e.g. {'CSP':6, 'PCA2D':2, ...}
    metric:       'AUC' or 'accuracy'
    window_size:  number of trials per sliding window
    directory:    path to save figures and logs
    """
    if dim_dict is None:
        dim_dict = {key: X.shape[1] for key, X in X_reduced_dict.items()}

    unique_days = np.unique(days_labels)
    lda_loaded  = clf_loaded.named_steps['LDA']

    for reducer, X_reduced in X_reduced_dict.items():
        # ─ Reset trial counter for this reducer ───────────────
        trial_counter = 0

        # Prepare storage
        aucs, accuracies = [], []
        inter_vars, intra_idle, intra_motor = [], [], []
        trial_indices = []

        # Loop over days
        for day in unique_days:
            mask = (days_labels == day)
            X_day          = X_csp[mask]
            X_reduced_day  = X_reduced[mask]
            y_day          = y_label[mask]
            n_trials_day   = X_day.shape[0]

            # Slide window within this day only
            for start in range(0, n_trials_day - window_size + 1):
                end = start + window_size

                # Get data for this window
                X_win     = X_day[start:end]
                y_win     = y_day[start:end]
                Xred_win  = X_reduced_day[start:end]

                # Compute metric
                scores = lda_loaded.decision_function(X_win)
                auc   = roc_auc_score(y_win, scores)
                pred  = lda_loaded.predict(X_win)
                acc   = accuracy_score(y_win, pred)

                aucs.append(auc)
                accuracies.append(acc)

                # Compute cluster variances
                cluster_var = utils.calculate_cluster_variance_window(Xred_win, y_win)
                inter_vars.append(cluster_var['inter_distance'])
                intra_idle.append(cluster_var['intra_distances']['idle'])
                intra_motor.append(cluster_var['intra_distances']['motor_imagery'])

                # Central trial index for plotting
                midpoint = start + window_size // 2
                trial_indices.append(trial_counter + midpoint)

            # Increment counter after finishing this day
            trial_counter += n_trials_day

        # Choose which metric to plot/log
        chosen = aucs if metric == 'AUC' else accuracies
        m_name  = metric

        # Compute correlations
        # Build a mask of windows where both values are finite
        valid_inter = (~np.isnan(chosen)) & (~np.isnan(inter_vars))
        corr_inter, _ = pearsonr(
            np.array(chosen)[valid_inter],
            np.array(inter_vars)[valid_inter])


        valid_intra_idle = (~np.isnan(chosen)) & (~np.isnan(intra_idle))
        corr_idle, _ = pearsonr(np.array(chosen)[valid_intra_idle],
                                np.array(intra_idle)[valid_intra_idle])

        valid_intra_motor = (~np.isnan(chosen)) & (~np.isnan(intra_motor))
        corr_motor, _ =pearsonr(np.array(chosen)[valid_intra_motor],
                                np.array(intra_motor)[valid_intra_motor])
        # Plotting
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)

        # 1) Inter-cluster
        axes[0].set_title(f'{metric} vs Inter-Cluster ({dim_dict[reducer]}D {reducer})')
        axes[0].plot(trial_indices, chosen, label=m_name, color='blue')
        twin0 = axes[0].twinx()
        twin0.plot(trial_indices, inter_vars, label=f'Inter (r={corr_inter:.2f})', color='red')
        twin0.set_ylabel('Inter Distance', color='red')

        # 2) Intra-cluster idle
        axes[1].set_title(f'{metric} vs Intra-Cluster Idle ({dim_dict[reducer]}D {reducer})')
        axes[1].plot(trial_indices, chosen, label=m_name, color='blue')
        twin1 = axes[1].twinx()
        twin1.plot(trial_indices, intra_idle, label=f'Idle (r={corr_idle:.2f})', color='green')
        twin1.set_ylabel('Idle Variance', color='green')

        # 3) Intra-cluster motor
        axes[2].set_title(f'{metric} vs Intra-Cluster Motor ({dim_dict[reducer]}D {reducer})')
        axes[2].plot(trial_indices, chosen, label=m_name, color='blue')
        twin2 = axes[2].twinx()
        twin2.plot(trial_indices, intra_motor, label=f'Motor (r={corr_motor:.2f})', color='purple')
        twin2.set_ylabel('Motor Variance', color='purple')

        for ax in axes:
            ax.set_xlabel('Trial Index (Sliding Window Midpoint)')
            ax.set_ylabel(m_name, color='blue')
            ax.grid(True)

        # Save if directory given
        if directory:
            fname = f'Sliding_{m_name}_Variance_{dim_dict[reducer]}D_{reducer}_w{window_size}.jpg'
            plt.savefig(f'{directory}/{fname}', dpi=300, bbox_inches='tight')
        plt.close()

        # Log correlations
        utils.log_metric_correlations(
            metric_name=m_name,
            directory=directory,
            reducer=reducer,
            dim=dim_dict[reducer],
            corr_inter=corr_inter,
            corr_intra_idle=corr_idle,
            corr_intra_motor=corr_motor
        )































def delta_acc_var(X_csp, y_label, days_labels, clf_loaded, start_test_day, end_test_day, directory, smooth=False, window=5):

    
    unique_days = np.unique(days_labels)
    num_days = len(unique_days)
    accuracies, inter_variances = [], []
    intra_variances_idle, intra_variances_motor = [], []
    lda_loaded = clf_loaded.named_steps['LDA']

    # Loop over each day to compute accuracy and variances
    for day in unique_days:
        day_mask = (days_labels == day)
        X_day = X_csp[day_mask]
        y_day = y_label[day_mask]

        # Calculate accuracy and variances
        accuracy = utils.evaluate_classifier(lda_loaded, X_day, y_day)
        cluster_variance = utils.calculate_cluster_variance(X_day, y_day)
        accuracies.append(accuracy)
        inter_variances.append(cluster_variance['inter_distance'])
        intra_variances_idle.append(cluster_variance['intra_distances']['idle'])
        intra_variances_motor.append(cluster_variance['intra_distances']['motor_imagery'])

    # Convert lists to numpy arrays
    accuracies, inter_variances, intra_variances_idle, intra_variances_motor = \
        map(np.array, (accuracies, inter_variances, intra_variances_idle, intra_variances_motor))

    if smooth:
        accuracies = np.convolve(accuracies, np.ones(window) / window, mode='valid')
        inter_variances = np.convolve(inter_variances, np.ones(window) / window, mode='valid')
        intra_variances_idle = np.convolve(intra_variances_idle, np.ones(window) / window, mode='valid')
        intra_variances_motor = np.convolve(intra_variances_motor, np.ones(window) / window, mode='valid')
        

    # Calculate delta matrices
    delta_acc_matrix = accuracies[:, np.newaxis] - accuracies[np.newaxis, :]
    delta_inter_var_matrix = inter_variances[:, np.newaxis] - inter_variances[np.newaxis, :]
    delta_intra_var_matrix_idle = intra_variances_idle[:, np.newaxis] - intra_variances_idle[np.newaxis, :]
    delta_intra_var_matrix_motor = intra_variances_motor[:, np.newaxis] - intra_variances_motor[np.newaxis, :]

    # Create a mask for the upper triangle, excluding the diagonal
    # mask = np.triu(np.ones_like(delta_acc_matrix, dtype=bool), k=1)

    # Create a mask for the lower triangle, excluding the diagonal
    mask = np.tril(np.ones_like(delta_acc_matrix, dtype=bool), k=-1)
    tick_positions = np.arange(0, num_days, 20)
    tick_labels = tick_positions + 30

    ### Option 1: Four subplots in a single figure ###
    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()
    fig1.suptitle('Delta Matrices for Accuracy and Cluster Variance', fontsize=16)

    # Plot delta accuracy matrix (masked upper triangle)
    cax1 = ax1.matshow(np.where(mask, delta_acc_matrix, np.nan), cmap='coolwarm')
    fig1.colorbar(cax1, ax=ax1)
    ax1.set_title('Delta Accuracy Matrix')
    ax1.set_xlabel('Day Index')
    ax1.set_ylabel('Day Index')
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels)

    # Plot delta inter-cluster variance matrix (masked upper triangle)
    cax2 = ax2.matshow(np.where(mask, delta_inter_var_matrix, np.nan), cmap='coolwarm')
    fig1.colorbar(cax2, ax=ax2)
    ax2.set_title('Delta Inter-Cluster Distance Matrix')
    ax2.set_xlabel('Day Index')
    ax2.set_ylabel('Day Index')
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels)

    # Plot delta intra-cluster variance (Idle) matrix (masked upper triangle)
    cax3 = ax3.matshow(np.where(mask, delta_intra_var_matrix_idle, np.nan), cmap='coolwarm')
    fig1.colorbar(cax3, ax=ax3)
    ax3.set_title('Delta Intra-Cluster Variance (Idle) Matrix')
    ax3.set_xlabel('Day Index')
    ax3.set_ylabel('Day Index')
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels)
    ax3.set_yticks(tick_positions)
    ax3.set_yticklabels(tick_labels)

    # Plot delta intra-cluster variance (Motor Imagery) matrix (masked upper triangle)
    cax4 = ax4.matshow(np.where(mask, delta_intra_var_matrix_motor, np.nan), cmap='coolwarm')
    fig1.colorbar(cax4, ax=ax4)
    ax4.set_title('Delta Intra-Cluster Variance (Motor Imagery) Matrix')
    ax4.set_xlabel('Day Index')
    ax4.set_ylabel('Day Index')
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(tick_labels)
    ax4.set_yticks(tick_positions)
    ax4.set_yticklabels(tick_labels)

    # Save the combined 4-panel plot
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig(os.path.join(directory, f'Delta_Matrices_Upper_Triangle_Test_Day_{start_test_day}_to_{end_test_day - 1}.jpg'))
    plt.close(fig1)

    ### Option 2: Two separate figures ###
    # First figure: Delta Accuracy and Delta Inter-Cluster Variance
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Delta Accuracy and Delta Inter-Cluster Distance', fontsize=14)

    # Plot delta accuracy matrix (masked upper triangle)
    cax1 = ax1.matshow(np.where(mask, delta_acc_matrix, np.nan), cmap='coolwarm')
    fig2.colorbar(cax1, ax=ax1)
    ax1.set_title('Delta Accuracy Matrix')
    ax1.set_xlabel('Day Index')
    ax1.set_ylabel('Day Index')
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels)

    # Plot delta inter-cluster variance matrix (masked upper triangle)
    cax2 = ax2.matshow(np.where(mask, delta_inter_var_matrix, np.nan), cmap='coolwarm')
    fig2.colorbar(cax2, ax=ax2)
    ax2.set_title('Delta Inter-Cluster Distance Matrix')
    ax2.set_xlabel('Day Index')
    ax2.set_ylabel('Day Index')
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels)

    fig2.tight_layout()
    fig2.savefig(os.path.join(directory, f'Delta_Accuracy_and_Inter_Variance_Upper_Triangle_Test_Day_{start_test_day}_to_{end_test_day - 1}.jpg'))
    plt.close(fig2)

    # Second figure: Delta Intra-Cluster Variance (Idle) and (Motor Imagery)
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle('Delta Intra-Cluster Distance (Idle and Motor)', fontsize=14)

    # Plot delta intra-cluster variance (Idle) matrix (masked upper triangle)
    cax3 = ax3.matshow(np.where(mask, delta_intra_var_matrix_idle, np.nan), cmap='coolwarm')
    fig3.colorbar(cax3, ax=ax3)
    ax3.set_title('Delta Intra-Cluster Distance (Idle) Matrix')
    ax3.set_xlabel('Day Index')
    ax3.set_ylabel('Day Index')
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels)
    ax3.set_yticks(tick_positions)
    ax3.set_yticklabels(tick_labels)

    # Plot delta intra-cluster variance (Motor Imagery) matrix (masked upper triangle)
    cax4 = ax4.matshow(np.where(mask, delta_intra_var_matrix_motor, np.nan), cmap='coolwarm')
    fig3.colorbar(cax4, ax=ax4)
    ax4.set_title('Delta Intra-Cluster Distance (Motor Imagery) Matrix')
    ax4.set_xlabel('Day Index')
    ax4.set_ylabel('Day Index')
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(tick_labels)
    ax4.set_yticks(tick_positions)
    ax4.set_yticklabels(tick_labels)

    fig3.tight_layout()
    fig3.savefig(os.path.join(directory, f'Delta_Intra_Variances_Upper_Triangle_Test_Day_{start_test_day}_to_{end_test_day - 1}.jpg'))
    plt.close(fig3)


    tril_indices = np.tril_indices_from(delta_acc_matrix,k=-1)
    delta_acc_flat = delta_acc_matrix[tril_indices]
    delta_inter_var_flat = delta_inter_var_matrix[tril_indices]
    delta_intra_var_flat_idle = delta_intra_var_matrix_idle[tril_indices]
    delta_intra_var_flat_motor = delta_intra_var_matrix_motor[tril_indices]

# Plot scatter graphs
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Scatter plot for Delta Accuracy vs Delta Inter-Cluster Variance
    r_inter, p_inter = pearsonr(delta_acc_flat, delta_inter_var_flat)
    ax[0].scatter(delta_acc_flat, delta_inter_var_flat, color='blue', alpha=0.6)
    ax[0].set_title('Delta Accuracy vs Delta Inter-Cluster Distance')
    ax[0].set_xlabel('Delta Accuracy')
    ax[0].set_ylabel('Delta Inter-Cluster Distance')
    ax[0].grid(True)
    ax[0].legend([f'Pearson r: {r_inter:.2f}'])

    # Scatter plot for Delta Accuracy vs Delta Intra-Cluster Variance (Idle)
    r_intra_idle, p_intra_idle = pearsonr(delta_acc_flat, delta_intra_var_flat_idle)
    ax[1].scatter(delta_acc_flat, delta_intra_var_flat_idle, color='green', alpha=0.6)
    ax[1].set_title('Delta Accuracy vs Delta Intra-Cluster Distance Idle')
    ax[1].set_xlabel('Delta Accuracy')
    ax[1].set_ylabel('Delta Intra-Cluster Distance Idle')
    ax[1].grid(True)
    ax[1].legend([f'Pearson r: {r_intra_idle:.2f}'])

    # Scatter plot for Delta Accuracy vs Delta Intra-Cluster Variance (Motor Imagery)
    r_intra_motor, p_intra_motor = pearsonr(delta_acc_flat, delta_intra_var_flat_motor)
    ax[2].scatter(delta_acc_flat, delta_intra_var_flat_motor, color='purple', alpha=0.6)
    ax[2].set_title('Delta Accuracy vs Delta Intra-Cluster Distance Motor Imagery')
    ax[2].set_xlabel('Delta Accuracy')
    ax[2].set_ylabel('Delta Intra-Cluster Distance Motor Imagery')
    ax[2].grid(True)
    ax[2].legend([f'Pearson r: {r_intra_motor:.2f}'])

    # Adjust layout and save figure
    fig.tight_layout()
    fig.savefig(os.path.join(directory, f'Delta_Accuracy_vs_Cluster_Variances_Test_Day_{start_test_day}_to_{end_test_day - 1}.jpg'))
    plt.show()
    plt.close()


    # Return delta matrices for further analysis if needed
    return delta_acc_matrix, delta_inter_var_matrix, delta_intra_var_matrix_idle, delta_intra_var_matrix_motor

def delta_auc_var(X_csp, y_label, days_labels, clf_loaded, start_test_day, end_test_day, directory, smooth=False, window=5):

    unique_days = np.unique(days_labels)
    num_days = len(unique_days)
    aucs, inter_variances = [], []
    intra_variances_idle, intra_variances_motor = [], []
    lda_loaded = clf_loaded.named_steps['LDA']

    for day in unique_days:
        day_mask = (days_labels == day)
        X_day = X_csp[day_mask]
        y_day = y_label[day_mask]


        scores_day = lda_loaded.decision_function(X_day)
        auc = roc_auc_score(y_day, scores_day)
        aucs.append(auc)
        cluster_variance = utils.calculate_cluster_variance(X_day, y_day)
        inter_variances.append(cluster_variance['inter_distance'])
        intra_variances_idle.append(cluster_variance['intra_distances']['idle'])
        intra_variances_motor.append(cluster_variance['intra_distances']['motor_imagery'])

    aucs, inter_variances, intra_variances_idle, intra_variances_motor = \
        map(np.array, (aucs, inter_variances, intra_variances_idle, intra_variances_motor))
    
    if smooth:
        aucs = np.convolve(aucs, np.ones(window) / window, mode='valid')
        inter_variances = np.convolve(inter_variances, np.ones(window) / window, mode='valid')
        intra_variances_idle = np.convolve(intra_variances_idle, np.ones(window) / window, mode='valid')
        intra_variances_motor = np.convolve(intra_variances_motor, np.ones(window) / window, mode='valid')

    delta_auc_matrix = aucs[:, np.newaxis] - aucs[np.newaxis, :]
    delta_inter_var_matrix = inter_variances[:, np.newaxis] - inter_variances[np.newaxis, :]
    delta_intra_var_matrix_idle = intra_variances_idle[:, np.newaxis] - intra_variances_idle[np.newaxis, :]
    delta_intra_var_matrix_motor = intra_variances_motor[:, np.newaxis] - intra_variances_motor[np.newaxis, :]

    # Create a mask for the upper triangle, excluding the diagonal
    # mask = np.triu(np.ones_like(delta_auc_matrix, dtype=bool), k=1)

    mask = np.tril(np.ones_like(delta_auc_matrix, dtype=bool), k=-1)
    #adjustmnets for sub201
    tick_positions = np.arange(0, num_days, 20)
    tick_labels = tick_positions + 30

    #adjustmnets for sub205
    # tick_positions = np.arange(0, num_days)
    # tick_labels = tick_positions + 3

    #adjustmnets for sub206
    # tick_positions = np.arange(0, num_days)
    # tick_labels = tick_positions + 4


    # Masked versions for clean plotting
    masked_auc         = np.where(mask, delta_auc_matrix, np.nan)
    masked_inter       = np.where(mask, delta_inter_var_matrix, np.nan)
    masked_intra_idle  = np.where(mask, delta_intra_var_matrix_idle, np.nan)
    masked_intra_motor = np.where(mask, delta_intra_var_matrix_motor, np.nan)

    # Colormap with defined NaN (masked) color
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color='white')  # show masked triangle as white

    # Symmetric norms around 0 (99th percentile caps outliers)
    vmax_auc   = np.nanpercentile(np.abs(masked_auc), 99)
    vmax_inter = np.nanpercentile(np.abs(masked_inter), 99)
    vmax_i_idle  = np.nanpercentile(np.abs(masked_intra_idle), 99)
    vmax_i_motor = np.nanpercentile(np.abs(masked_intra_motor), 99)

    norm_auc   = TwoSlopeNorm(vmin=-vmax_auc,   vcenter=0.0, vmax=vmax_auc)
    norm_inter = TwoSlopeNorm(vmin=-vmax_inter, vcenter=0.0, vmax=vmax_inter)
    norm_i_idle  = TwoSlopeNorm(vmin=-vmax_i_idle,  vcenter=0.0, vmax=vmax_i_idle)
    norm_i_motor = TwoSlopeNorm(vmin=-vmax_i_motor, vcenter=0.0, vmax=vmax_i_motor)


    fig1, axes = plt.subplots(2, 2, figsize=(12, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()
    fig1.suptitle('Delta Matrices for AUC and Cluster Variance', fontsize=16)

    cax1 = ax1.matshow(masked_auc, cmap=cmap, norm=norm_auc, interpolation='none')
    fig1.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04, label='Δ AUC')
    ax1.set_title('Δ AUC', fontsize=12, pad=8)
    ax1.set_aspect('auto')
    ax1.tick_params(labelsize=9)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_xlabel('Day Index')
    ax1.set_ylabel('Day Index')
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels)

    cax2 = ax2.matshow(masked_inter, cmap=cmap, norm=norm_inter, interpolation='none')
    fig1.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04, label='Δ Inter-Cluster Distance')
    ax2.set_title('Δ Inter-Cluster Distance', fontsize=12, pad=8)
    ax2.set_aspect('auto')
    ax2.tick_params(labelsize=9)
    ax2.set_xticks(tick_positions)
    ax2.set_xlabel('Day Index')
    ax2.set_ylabel('Day Index')
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels)

    cax3 = ax3.matshow(masked_intra_idle, cmap=cmap, norm=norm_i_idle, interpolation='none')
    fig1.colorbar(cax3, ax=ax3, fraction=0.046, pad=0.04, label='Δ Intra-Cluster Variance Idle')
    ax3.set_title('Δ Intra-Cluster Variance Idle', fontsize=12, pad=8)
    ax3.set_aspect('auto')
    ax3.tick_params(labelsize=9)
    ax3.set_xticks(tick_positions)
    ax3.set_xlabel('Day Index')
    ax3.set_ylabel('Day Index')
    ax3.set_xticklabels(tick_labels)
    ax3.set_yticks(tick_positions)
    ax3.set_yticklabels(tick_labels)

    cax4 = ax4.matshow(masked_intra_motor, cmap=cmap, norm=norm_i_motor, interpolation='none')
    fig1.colorbar(cax4, ax=ax4, fraction=0.046, pad=0.04, label='Δ Intra-Cluster Variance MI')
    ax4.set_title('Δ Intra-Cluster Variance MI', fontsize=12, pad=8)
    ax4.set_aspect('auto')
    ax4.tick_params(labelsize=9)
    ax4.set_xticks(tick_positions)
    ax4.set_xlabel('Day Index')
    ax4.set_ylabel('Day Index')
    ax4.set_xticklabels(tick_labels)
    ax4.set_yticks(tick_positions)
    ax4.set_yticklabels(tick_labels)

    fig1.tight_layout()
    fig1.subplots_adjust(hspace=0.4, wspace=0.3)
    fig1.savefig(os.path.join(directory, f'Delta_AUC_Matrices_Upper_Triangle_Test_Day_{start_test_day}_to_{end_test_day - 1}.jpg'), dpi=300, bbox_inches='tight')
    plt.close(fig1)


    ### Option 2: Two separate figures ###
    # First figure: Delta Accuracy and Delta Inter-Cluster Variance
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Delta AUC and Delta Inter-Cluster Distance', fontsize=14)

    # Plot delta accuracy matrix (masked upper triangle)
    cax1 = ax1.matshow(masked_auc, cmap=cmap, norm=norm_auc, interpolation='none')
    fig2.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04, label='Δ AUC')
    ax1.set_title('Δ AUC', fontsize=12, pad=8)
    ax1.set_aspect('auto')
    ax1.tick_params(labelsize=9)
    ax1.set_xlabel('Day Index')
    ax1.set_ylabel('Day Index')
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels)

    # Plot delta inter-cluster variance matrix (masked upper triangle)
    cax2 = ax2.matshow(masked_inter, cmap=cmap, norm=norm_inter, interpolation='none')
    fig2.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04, label='Δ Inter-Cluster Distance')
    ax2.set_title('Δ Inter-Cluster Distance', fontsize=12, pad=8)
    ax2.set_aspect('auto')
    ax2.tick_params(labelsize=9)
    ax2.set_xlabel('Day Index')
    ax2.set_ylabel('Day Index')
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels)

    fig2.tight_layout()
    fig2.savefig(os.path.join(directory, f'Delta_AUC_and_Inter_Distance_Upper_Triangle_Test_Day_{start_test_day}_to_{end_test_day - 1}.jpg'))
    plt.close(fig2)

    # Second figure: Delta Intra-Cluster Variance (Idle) and (Motor Imagery)
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle('Delta Intra-Cluster Variance (Idle and MI)', fontsize=14)

    # Plot delta intra-cluster variance (Idle) matrix (masked upper triangle)
    cax3 = ax3.matshow(masked_intra_idle, cmap=cmap, norm=norm_i_idle, interpolation='none')
    fig3.colorbar(cax3, ax=ax3, fraction=0.046, pad=0.04, label='Δ Intra-Cluster Variance Idle')
    ax3.set_title('Δ Intra-Cluster Variance Idle', fontsize=12, pad=8)
    ax3.set_aspect('auto')
    ax3.tick_params(labelsize=9)
    ax3.set_xlabel('Day Index')
    ax3.set_ylabel('Day Index')
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels)
    ax3.set_yticks(tick_positions)
    ax3.set_yticklabels(tick_labels)

    # Plot delta intra-cluster variance (Motor Imagery) matrix (masked upper triangle)
    cax4 = ax4.matshow(masked_intra_motor, cmap=cmap, norm=norm_i_motor, interpolation='none')
    fig3.colorbar(cax4, ax=ax4, fraction=0.046, pad=0.04, label='Δ Intra-Cluster Variance MI')
    ax4.set_title('Δ Intra-Cluster Variance MI', fontsize=12, pad=8)
    ax4.set_aspect('auto')
    ax4.tick_params(labelsize=9)
    ax4.set_xlabel('Day Index')
    ax4.set_ylabel('Day Index')
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(tick_labels)
    ax4.set_yticks(tick_positions)
    ax4.set_yticklabels(tick_labels)

    fig3.tight_layout()
    fig3.savefig(os.path.join(directory, f'Delta_Intra_Variances_Upper_Triangle_Test_Day_{start_test_day}_to_{end_test_day - 1}.jpg'))
    plt.close(fig3)

    # Not smoothed
    tril_indices = np.tril_indices_from(delta_auc_matrix, k=-1)
    delta_auc_flat = delta_auc_matrix[tril_indices]
    delta_inter_var_flat = delta_inter_var_matrix[tril_indices]
    delta_intra_var_flat_idle = delta_intra_var_matrix_idle[tril_indices]
    delta_intra_var_flat_motor = delta_intra_var_matrix_motor[tril_indices]

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    scatter_with_fit(ax[0], delta_auc_flat, delta_inter_var_flat,
                    color='tab:blue',
                    title='ΔAUC vs Δ Inter-Cluster Distance',
                    xlabel='ΔAUC', ylabel='Δ Inter-Cluster Distance',
                    alpha_sig=0.05)

    scatter_with_fit(ax[1], delta_auc_flat, delta_intra_var_flat_idle,
                    color='tab:green',
                    title='ΔAUC vs Δ Intra-Cluster Variance Idle',
                    xlabel='ΔAUC', ylabel='Δ Intra-Cluster Variance Idle',
                    alpha_sig=0.05)

    scatter_with_fit(ax[2], delta_auc_flat, delta_intra_var_flat_motor,
                    color='tab:purple',
                    title='ΔAUC vs Δ Intra-Cluster Variance MI',
                    xlabel='ΔAUC', ylabel='Δ Intra-Cluster Variance MI',
                    alpha_sig=0.05)

    fig.tight_layout()
    fig.savefig(os.path.join(directory, f'Delta_AUC_vs_Cluster_Variances_Test_Day_{start_test_day}_to_{end_test_day - 1}.jpg'))
    plt.close(fig)


    return delta_auc_matrix, delta_inter_var_matrix,delta_intra_var_matrix_idle, delta_intra_var_matrix_motor

def plot_accuracy_vs_variances_smoothed(unique_days, accuracies, inter_variances, intra_variances_idle, intra_variances_motor, start_test_day, end_test_day, directory, min_window=2, max_window=4):
    # Find the best window size for inter- and intra-cluster variances separately
    best_window_size_inter, best_corr_accuracy_inter, p_value_inter = utils.find_best_window_size(accuracies, inter_variances, min_window=min_window, max_window=max_window)
    best_window_size_intra_motor, best_corr_accuracy_intra_motor, p_value_intra_motor = utils.find_best_window_size(accuracies, intra_variances_motor, min_window=min_window, max_window=max_window)
    best_window_size_intra_idle, best_corr_accuracy_intra_idle, p_value_intra_idle = utils.find_best_window_size(accuracies, intra_variances_idle, min_window=min_window, max_window=max_window)

    print(f"Best window size for inter-cluster distance: {best_window_size_inter}, Correlation: {best_corr_accuracy_inter:.4f}")
    print(f"Best window size for intra-cluster Motor Imagery varince: {best_window_size_intra_motor}, Correlation: {best_corr_accuracy_intra_motor:.4f}")
    print(f"Best window size for intra-cluster IDLE variance: {best_window_size_intra_idle}, Correlation: {best_corr_accuracy_intra_idle:.4f}")

    # Apply moving average smoothing
    smoothed_accuracies_inter = np.convolve(accuracies, np.ones(best_window_size_inter) / best_window_size_inter, mode='valid')
    smoothed_inter_variances = np.convolve(inter_variances, np.ones(best_window_size_inter) / best_window_size_inter, mode='valid')

    smoothed_accuracies_intra_idle = np.convolve(accuracies, np.ones(best_window_size_intra_idle) / best_window_size_intra_idle, mode='valid')
    smoothed_intra_variances_idle = np.convolve(intra_variances_idle, np.ones(best_window_size_intra_idle) / best_window_size_intra_idle, mode='valid')
    
    smoothed_accuracies_intra_motor = np.convolve(accuracies, np.ones(best_window_size_intra_motor) / best_window_size_intra_motor, mode='valid')
    smoothed_intra_variances_motor = np.convolve(intra_variances_motor, np.ones(best_window_size_intra_motor) / best_window_size_intra_motor, mode='valid')
    # return  smoothed_accuracies_inter, smoothed_inter_variances, smoothed_intra_variances_idle, smoothed_intra_variances_motor
    # Create 3 subplots with shared layout
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)

    plt.subplots_adjust(hspace=0.25)

    # First subplot: Accuracy vs Inter-Cluster Variance
    ax1.set_xlabel('Days', labelpad=-10)
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(unique_days[:len(smoothed_accuracies_inter)], smoothed_accuracies_inter, label='Smoothed Classifier Accuracy', marker='o', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel("Inter-Cluster Distance (Cohen's d)", color='tab:red')
    ax1_twin.plot(unique_days[:len(smoothed_inter_variances)], smoothed_inter_variances, label='Smoothed Inter-Cluster Distance', marker='x', color='tab:red')
    ax1_twin.tick_params(axis='y', labelcolor='tab:red')

    ax1_twin.legend([f"Inter-Cluster Distance (corr: {best_corr_accuracy_inter:.2f}), window: {best_window_size_inter})"], loc='upper right')
    ax1.set_title(f"Accuracy vs Inter-Cluster Distance Over Days {start_test_day}-{end_test_day-1}")

    # Second subplot: Accuracy vs Intra-Cluster Variance (Idle)
    ax2.set_xlabel('Days', labelpad=-10)
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(unique_days[:len(smoothed_accuracies_intra_idle)], smoothed_accuracies_intra_idle, label='Smoothed Classifier Accuracy', marker='o', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('Intra-Cluster Variance (Idle)', color='tab:green')
    ax2_twin.plot(unique_days[:len(smoothed_intra_variances_idle)], smoothed_intra_variances_idle, label='Smoothed Intra-Cluster Distance Idle [std]', marker='x', color='tab:green')
    ax2_twin.tick_params(axis='y', labelcolor='tab:green')

    ax2_twin.legend([f"Intra-Cluster Variance Idle (corr: {best_corr_accuracy_intra_idle:.2f}, window: {best_window_size_intra_idle}))"], loc='upper right')
    ax2.set_title(f'Accuracy vs Intra-Cluster Variance (Idle) Over Days {start_test_day}-{end_test_day-1}')

    # Third subplot: Accuracy vs Intra-Cluster Variance (Motor Imagery)
    ax3.set_xlabel('Days', labelpad=-10)
    ax3.set_ylabel('Accuracy', color='tab:blue')
    ax3.plot(unique_days[:len(smoothed_accuracies_intra_motor)], smoothed_accuracies_intra_motor, label='Smoothed Classifier Accuracy', marker='o', color='tab:blue')
    ax3.tick_params(axis='y', labelcolor='tab:blue')

    ax3_twin = ax3.twinx()
    ax3_twin.set_ylabel('Intra-Cluster Variance (Motor Imagery)', color='tab:purple')
    ax3_twin.plot(unique_days[:len(smoothed_intra_variances_motor)], smoothed_intra_variances_motor, label='Smoothed Intra-Cluster Distance Motor Imagery [std]', marker='x', color='tab:purple')
    ax3_twin.tick_params(axis='y', labelcolor='tab:purple')

    ax3_twin.legend([f"Intra-Cluster Variance Motor Imagery (corr: {best_corr_accuracy_intra_motor:.2f}, window: {best_window_size_intra_motor}))"], loc='upper right')
    ax3.set_title(f'Accuracy vs Intra-Cluster Variance (Motor Imagery) Over Days {start_test_day}-{end_test_day-1}')

    # Adjust x-axis ticks for all subplots
    for ax in [ax1, ax2, ax3]:
        xticks = ax.get_xticks().astype(int)
        ax.set_xticklabels(xticks + 30)

    # Adjust layout and save
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f'Accuracy_vs_Variances_Test_Day_{start_test_day}_to_{end_test_day - 1}_inter{best_window_size_inter}_intraID{best_window_size_intra_idle}_intraMI{best_window_size_intra_motor}.jpg'
    full_path = os.path.join(directory, filename)
    plt.savefig(full_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Create 1x3 subplots with shared y-axis for accuracy
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # First subplot: Accuracy vs Inter-Cluster Variance
    ax1.scatter(smoothed_inter_variances, smoothed_accuracies_inter, c='b', label=f'Inter-Cluster Variance (r={best_corr_accuracy_inter:.2f})', marker='o')
    ax1.set_title('Accuracy vs. Inter-Cluster Variance')
    ax1.set_xlabel("Inter-Cluster Variance [Cohen's d]")
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True)

    # Second subplot: Accuracy vs Intra-Cluster Variance (Idle)
    ax2.scatter(smoothed_intra_variances_idle, smoothed_accuracies_intra_idle, c='g', label=f'Idle (r={best_corr_accuracy_intra_idle:.2f})', marker='o')
    ax2.set_title('Accuracy vs. Intra-Cluster Idle Variance')
    ax2.set_xlabel('Intra-Cluster Variance Idle [std]')
    ax2.legend(loc='lower right')
    ax2.grid(True)

    # Third subplot: Accuracy vs Intra-Cluster Variance (Motor Imagery)
    ax3.scatter(smoothed_intra_variances_motor, smoothed_accuracies_intra_motor, c='purple', label=f'Motor (r={best_corr_accuracy_intra_motor:.2f})', marker='o')
    ax3.set_title('Accuracy vs. Intra-Cluster Motor Imagery Variance')
    ax3.set_xlabel('Intra-Cluster Variance Motor Imagery [std]')
    ax3.legend(loc='lower right')
    ax3.grid(True)

    # Adjust layout to ensure everything fits well
    plt.tight_layout()
    filename=f'correlation_analysis_between_Accuracy_and_variances_inter{best_window_size_inter}_intraID{best_window_size_intra_idle}_intraMI{best_window_size_intra_motor}.jpg'
    full_path = os.path.join(directory, filename)
    plt.savefig(full_path)
    plt.close()

    return  smoothed_accuracies_inter,  smoothed_inter_variances, smoothed_accuracies_intra_idle, smoothed_intra_variances_idle,  smoothed_accuracies_intra_motor,smoothed_intra_variances_motor

    


def plot_auc_vs_variances_smoothed(unique_days, auc_scores, inter_variances, intra_variances_idle, intra_variances_motor,
                                    start_test_day, end_test_day, directory, min_window=2, max_window=15, label_offset=0, start_adj=0, end_adj=0):
    
    # Calculate adjusted days for the title
    adj_start = start_test_day + start_adj
    adj_end = end_test_day + end_adj
    adj_unique_days = unique_days[(unique_days >= adj_start) & (unique_days < adj_end)]
    # Find the best window size for inter- and intra-cluster variances
    best_window_size_inter, best_corr_auc_inter, p_value_inter = utils.find_best_window_size(auc_scores, inter_variances, min_window, max_window)
    best_window_size_intra_idle, best_corr_auc_intra_idle, p_value_intra_idle = utils.find_best_window_size(auc_scores, intra_variances_idle, min_window, max_window)
    best_window_size_intra_motor, best_corr_auc_intra_motor, p_value_intra_motor = utils.find_best_window_size(auc_scores, intra_variances_motor, min_window, max_window)

    print(f"Best window for inter-cluster distance: {best_window_size_inter}, Corr: {best_corr_auc_inter:.4f}")
    print(f"Best window for intra-cluster MI: {best_window_size_intra_motor}, Corr: {best_corr_auc_intra_motor:.4f}")
    print(f"Best window for intra-cluster IDLE: {best_window_size_intra_idle}, Corr: {best_corr_auc_intra_idle:.4f}")


    def smooth(data, win):
        return np.convolve(data, np.ones(win)/win, mode='valid')
    
    smoothed_auc_inter = smooth(auc_scores, best_window_size_inter)
    smoothed_inter_variances = smooth(inter_variances, best_window_size_inter)

    smoothed_auc_idle = smooth(auc_scores, best_window_size_intra_idle)
    smoothed_intra_variances_idle = smooth(intra_variances_idle, best_window_size_intra_idle)

    smoothed_auc_motor = smooth(auc_scores, best_window_size_intra_motor)
    smoothed_intra_variances_motor = smooth(intra_variances_motor, best_window_size_intra_motor)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
# Helper for plots
    def plot_panel(ax, auc_data, metric_data, metric_name, color, corr, win):
        # Adjust days to match the valid convolution length
        plot_days = unique_days[:len(auc_data)]
        
        ax.set_xlabel('Days')
        ax.set_ylabel('AUC', color='tab:blue')
        ax.plot(plot_days, auc_data, marker='o', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        
        ax_twin = ax.twinx()
        ax_twin.set_ylabel(metric_name, color=color)
        ax_twin.plot(plot_days, metric_data, marker='x', color=color)
        ax_twin.tick_params(axis='y', labelcolor=color)
        metric_clean = metric_name.split('[')[0]
        ax.set_title(f"AUC vs {metric_clean} (Days {adj_start}-{adj_end})")
        ax_twin.legend([f"{metric_clean} (r={corr:.2f}, w={win})"], loc='upper right')
        
        # --- Handle Offsets ---
        if label_offset == 30: # Sub 201
             xticks = ax.get_xticks().astype(int)
             ax.set_xticklabels(xticks + label_offset)
        else: # Sub 205/206
             ax.set_xticks(plot_days) 
             ax.set_xticklabels(plot_days + label_offset)



    # Generate Panels
    plot_panel(ax1, smoothed_auc_inter, smoothed_inter_variances, "Inter-Cluster [Cohen's d]", 'tab:red', best_corr_auc_inter, best_window_size_inter)
    plot_panel(ax2, smoothed_auc_idle, smoothed_intra_variances_idle, 'Intra-Cluster Idle [std]', 'tab:green', best_corr_auc_intra_idle, best_window_size_intra_idle)
    plot_panel(ax3, smoothed_auc_motor, smoothed_intra_variances_motor, 'Intra-Cluster Motor Imagery [std]', 'tab:purple', best_corr_auc_intra_motor, best_window_size_intra_motor)
    plt.tight_layout()

    if directory:
        fname = f'AUC_vs_Variances_Days_{adj_start}_to_{adj_end}_inter{best_window_size_inter}_intraID{best_window_size_intra_idle}_intraMI{best_window_size_intra_motor}.jpg'
        plt.savefig(os.path.join(directory, fname), dpi=300)
    plt.close()



    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    ax1.scatter(smoothed_inter_variances, smoothed_auc_inter, c='b', label=f'Inter (r={best_corr_auc_inter:.2f})', marker='o')
    ax1.set_title('AUC vs. Inter-Cluster Variance')
    ax1.set_xlabel("Inter-Cluster Distance [Cohen's d]")
    ax1.set_ylabel('AUC')
    ax1.grid(True)
    ax1.legend()

    ax2.scatter(smoothed_intra_variances_idle, smoothed_auc_idle, c='g', label=f'Idle (r={best_corr_auc_intra_idle:.2f})', marker='o')
    ax2.set_title('AUC vs. Intra-Cluster Idle Variance')
    ax2.set_xlabel('Intra-Cluster Variance Idle [std]')
    ax2.grid(True)
    ax2.legend()

    ax3.scatter(smoothed_intra_variances_motor, smoothed_auc_motor, c='purple', label=f'Motor (r={best_corr_auc_intra_motor:.2f})', marker='o')
    ax3.set_title('AUC vs. Intra-Cluster Motor Imagery Variance')
    ax3.set_xlabel('Intra-Cluster Variance Motor Imagery [std]')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    if directory:
        fname = f'AUC_correlation_analysis_inter{best_window_size_inter}_idle{best_window_size_intra_idle}_motor{best_window_size_intra_motor}.jpg'
        plt.savefig(os.path.join(directory, fname), dpi=300)
    plt.close()

    return  smoothed_auc_inter, smoothed_inter_variances, smoothed_auc_idle, smoothed_intra_variances_idle,  smoothed_auc_motor ,smoothed_intra_variances_motor 
 
def consistency_of_signals(csp_features, days_labels):
    unique_days = np.unique(days_labels)
    signal_variances = []

    for day in unique_days:
        day_mask = (days_labels == day)
        signal_variance = np.var(csp_features[day_mask], axis=0).mean()
        signal_variances.append(signal_variance)
    
    # Plot variance of signals over days
    plt.figure(figsize=(10, 5))
    plt.plot(unique_days + 30, signal_variances, 'bo-', label='Signal Variance')
    plt.title('Variance of CSP Features Over Time')
    plt.xlabel('Days')
    plt.ylabel('Mean Signal Variance')
    plt.grid(True)
    plt.savefig(os.path.join(r"D:/Niv/Motor imagery skill/Figures2",'Variance of CSP Features Over Time.jpg'))
    plt.close

    return signal_variances


def delta_vs_raw(X_csp, y_label, days_labels, clf_loaded, start_test_day, end_test_day, directory, smooth=False, window=5):

    unique_days = np.unique(days_labels)
    num_days = len(unique_days)
    aucs, inter_variances = [], []
    intra_variances_idle, intra_variances_motor = [], []
    lda_loaded = clf_loaded.named_steps['LDA']

    for day in unique_days:
        day_mask = (days_labels == day)
        X_day = X_csp[day_mask]
        y_day = y_label[day_mask]


        scores_day = lda_loaded.decision_function(X_day)
        auc = roc_auc_score(y_day, scores_day)
        aucs.append(auc)
        cluster_variance = utils.calculate_cluster_variance(X_day, y_day)
        inter_variances.append(cluster_variance['inter_distance'])
        intra_variances_idle.append(cluster_variance['intra_distances']['idle'])
        intra_variances_motor.append(cluster_variance['intra_distances']['motor_imagery'])

    aucs, inter_variances, intra_variances_idle, intra_variances_motor = \
        map(np.array, (aucs, inter_variances, intra_variances_idle, intra_variances_motor))
    
    if smooth:
        aucs = np.convolve(aucs, np.ones(window) / window, mode='valid')
        inter_variances = np.convolve(inter_variances, np.ones(window) / window, mode='valid')
        intra_variances_idle = np.convolve(intra_variances_idle, np.ones(window) / window, mode='valid')
        intra_variances_motor = np.convolve(intra_variances_motor, np.ones(window) / window, mode='valid')

    delta_auc_matrix = aucs[:, np.newaxis] - aucs[np.newaxis, :]
    delta_inter_var_matrix = inter_variances[:, np.newaxis] - inter_variances[np.newaxis, :]
    delta_intra_var_matrix_idle = intra_variances_idle[:, np.newaxis] - intra_variances_idle[np.newaxis, :]
    delta_intra_var_matrix_motor = intra_variances_motor[:, np.newaxis] - intra_variances_motor[np.newaxis, :]

    # Create a mask for the upper triangle, excluding the diagonal
    # mask = np.triu(np.ones_like(delta_auc_matrix, dtype=bool), k=1)

    mask = np.tril(np.ones_like(delta_auc_matrix, dtype=bool), k=-1)
    #adjustmnets for sub201
    offset = 30
    tick_positions = np.arange(0, num_days, 20) + offset
    tick_labels = tick_positions
    tick_positions_delta = np.arange(0, num_days, 20)

    #adjustmnets for sub205
    # offset = 3
    # tick_positions = np.arange(0, num_days) + offset
    # tick_labels = tick_positions 
    # tick_positions_delta = np.arange(0, num_days)

    #adjustmnets for sub206
    # offset = 4
    # tick_positions = np.arange(0, num_days) + offset
    # tick_labels = tick_positions 
    # tick_positions_delta =np.arange(0, num_days)

    # Masked versions for clean plotting
    masked_auc         = np.where(mask, delta_auc_matrix, np.nan)
    masked_inter       = np.where(mask, delta_inter_var_matrix, np.nan)
    masked_intra_idle  = np.where(mask, delta_intra_var_matrix_idle, np.nan)
    masked_intra_motor = np.where(mask, delta_intra_var_matrix_motor, np.nan)

    # Colormap with defined NaN (masked) color
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color='white')  # show masked triangle as white

    # Symmetric norms around 0 (99th percentile caps outliers)
    vmax_auc   = np.nanpercentile(np.abs(masked_auc), 99)
    vmax_inter = np.nanpercentile(np.abs(masked_inter), 99)
    vmax_i_idle  = np.nanpercentile(np.abs(masked_intra_idle), 99)
    vmax_i_motor = np.nanpercentile(np.abs(masked_intra_motor), 99)

    norm_auc   = TwoSlopeNorm(vmin=-vmax_auc,   vcenter=0.0, vmax=vmax_auc)
    norm_inter = TwoSlopeNorm(vmin=-vmax_inter, vcenter=0.0, vmax=vmax_inter)
    norm_i_idle  = TwoSlopeNorm(vmin=-vmax_i_idle,  vcenter=0.0, vmax=vmax_i_idle)
    norm_i_motor = TwoSlopeNorm(vmin=-vmax_i_motor, vcenter=0.0, vmax=vmax_i_motor)

    # -----------------------------
    # 4 rows × 2 cols: Raw | Δ
    # -----------------------------
    # If smoothed, the series length may have shrunk; take size from the Δ matrices
    num_days_plot = masked_auc.shape[0]

    # Day labels (set base_offset per subject)
    day_idx    = np.arange(num_days_plot)
    day_labels = day_idx + offset

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    (ax_auc_raw, ax_auc_delta,
     ax_inter_raw, ax_inter_delta,
     ax_idle_raw, ax_idle_delta,
     ax_mi_raw,   ax_mi_delta) = axes.flatten()
    
    panel_letters = ['A', 'B', 'C', 'D']
    for ax, letter in zip([ax_auc_raw, ax_inter_raw, ax_idle_raw, ax_mi_raw], panel_letters):
        ax.text(-0.08, 1.02, letter,            # left of x=0, above y=1
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=13, fontweight='bold',
                clip_on=False)   


    # ---------- Row 1: AUC ----------
    ax_auc_raw.plot(day_labels, aucs, lw=1.6)
    ax_auc_raw.set_title("AUC", fontsize=12, pad=6)
    ax_auc_raw.set_ylabel("AUC")
    ax_auc_raw.set_xticks(tick_positions)
    ax_auc_raw.set_xticklabels(tick_labels)
    ax_auc_raw.grid(True, alpha=0.25)

    im = ax_auc_delta.matshow(masked_auc, cmap=cmap, norm=norm_auc, interpolation='none')
    fig.colorbar(im, ax=ax_auc_delta, fraction=0.046, pad=0.04, label="ΔAUC")
    ax_auc_delta.set_title("ΔAUC", fontsize=12, pad=6)
    ax_auc_delta.set_aspect('auto')
    ax_auc_delta.set_xlim(-0.5, num_days_plot - 0.5)
    ax_auc_delta.set_ylim(num_days_plot - 0.5, -0.5)
    ax_auc_delta.set_xticks(tick_positions_delta); ax_auc_delta.set_xticklabels(tick_labels)
    ax_auc_delta.set_yticks(tick_positions_delta); ax_auc_delta.set_yticklabels(tick_labels)
    ax_auc_delta.set_xlabel("Day Index");     ax_auc_delta.set_ylabel("Day Index")

    # ---------- Row 2: Inter-cluster distance ----------
    ax_inter_raw.plot(day_labels, inter_variances, lw=1.6)
    ax_inter_raw.set_title("Inter-cluster distance", fontsize=12, pad=6)
    ax_inter_raw.set_ylabel("Inter-cluster distance")
    ax_inter_raw.set_xticks(tick_positions)
    ax_inter_raw.set_xticklabels(tick_labels)
    ax_inter_raw.grid(True, alpha=0.25)

    im = ax_inter_delta.matshow(masked_inter, cmap=cmap, norm=norm_inter, interpolation='none')
    fig.colorbar(im, ax=ax_inter_delta, fraction=0.046, pad=0.04, label="Δ Inter-cluster")
    ax_inter_delta.set_title("Δ Inter-cluster distance", fontsize=12, pad=6)
    ax_inter_delta.set_aspect('auto')
    ax_inter_delta.set_xlim(-0.5, num_days_plot - 0.5)
    ax_inter_delta.set_ylim(num_days_plot - 0.5, -0.5)
    ax_inter_delta.set_xticks(tick_positions_delta); ax_inter_delta.set_xticklabels(tick_labels)
    ax_inter_delta.set_yticks(tick_positions_delta); ax_inter_delta.set_yticklabels(tick_labels)
    ax_inter_delta.set_xlabel("Day Index");     ax_inter_delta.set_ylabel("Day Index")

    # ---------- Row 3: Intra (Idle) ----------
    ax_idle_raw.plot(day_labels, intra_variances_idle, lw=1.6)
    ax_idle_raw.set_title("Intra-cluster variance Idle", fontsize=12, pad=6)
    ax_idle_raw.set_ylabel("Intra Idle")
    ax_idle_raw.set_xticks(tick_positions)
    ax_idle_raw.set_xticklabels(tick_labels)
    ax_idle_raw.grid(True, alpha=0.25)

    im = ax_idle_delta.matshow(masked_intra_idle, cmap=cmap, norm=norm_i_idle, interpolation='none')
    fig.colorbar(im, ax=ax_idle_delta, fraction=0.046, pad=0.04, label="Δ Intra Idle")
    ax_idle_delta.set_title("Δ Intra-cluster variance Idle", fontsize=12, pad=6)
    ax_idle_delta.set_aspect('auto')
    ax_idle_delta.set_xlim(-0.5, num_days_plot - 0.5)
    ax_idle_delta.set_ylim(num_days_plot - 0.5, -0.5)
    ax_idle_delta.set_xticks(tick_positions_delta); ax_idle_delta.set_xticklabels(tick_labels)
    ax_idle_delta.set_yticks(tick_positions_delta); ax_idle_delta.set_yticklabels(tick_labels)
    ax_idle_delta.set_xlabel("Day Index");     ax_idle_delta.set_ylabel("Day Index")

    # ---------- Row 4: Intra (MI) ----------
    ax_mi_raw.plot(day_labels, intra_variances_motor, lw=1.6)
    ax_mi_raw.set_title("Intra-cluster variance MI", fontsize=12, pad=6)
    ax_mi_raw.set_ylabel("Intra MI")
    ax_mi_raw.set_xlabel("Day")   # bottom row shows the x-label
    ax_mi_raw.set_xticks(tick_positions)
    ax_mi_raw.set_xticklabels(tick_labels)
    ax_mi_raw.grid(True, alpha=0.25)

    im = ax_mi_delta.matshow(masked_intra_motor, cmap=cmap, norm=norm_i_motor, interpolation='none')
    fig.colorbar(im, ax=ax_mi_delta, fraction=0.046, pad=0.04, label="Δ Intra MI")
    ax_mi_delta.set_title("Δ Intra-cluster variance MI", fontsize=12, pad=6)
    ax_mi_delta.set_aspect('auto')
    ax_mi_delta.set_xlim(-0.5, num_days_plot - 0.5)
    ax_mi_delta.set_ylim(num_days_plot - 0.5, -0.5)
    ax_mi_delta.set_xticks(tick_positions_delta); ax_mi_delta.set_xticklabels(tick_labels)
    ax_mi_delta.set_yticks(tick_positions_delta); ax_mi_delta.set_yticklabels(tick_labels)
    ax_mi_delta.set_xlabel("Day Index");       ax_mi_delta.set_ylabel("Day Index")

    annotate_past_future(ax_auc_delta)   # Row 1, right column only


    fig.suptitle(f"Across-Day Dynamics: Raw versus Pairwise Δ", fontsize=16)
    fig.tight_layout()
    out = os.path.join(directory, f"Raw_vs_Delta_{day_labels[0]}_to_{day_labels[-1]}.jpg")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)




def scatter_with_fit(ax, x, y, color, title, xlabel, ylabel, star_scheme='standard', alpha_sig=0.05):

    ax.scatter(x, y, s=36, alpha=0.75, color=color, edgecolor='k', linewidth=0.5)

    # zero lines
    ax.axhline(0, color='k', lw=0.6, alpha=0.3)
    ax.axvline(0, color='k', lw=0.6, alpha=0.3)

    # regression
    if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
        res = linregress(x, y)  # slope, intercept, rvalue, pvalue, stderr
        xs = np.linspace(np.min(x), np.max(x), 200)
        ys = res.intercept + res.slope * xs

        
        stars = p_to_stars(res.pvalue, scheme=star_scheme)
        line_style = '-' if res.pvalue < alpha_sig else (0, (4, 2))

        ax.plot(xs, ys, color=color, lw=1.6, alpha=0.95, linestyle=line_style)

        # 95% CI band
        n = len(x)
        yhat = res.intercept + res.slope * np.asarray(x, float)
        s_yx = np.sqrt(np.sum((y - yhat)**2) / (n - 2))
        xbar = np.mean(x)
        Sxx = np.sum((x - xbar)**2)
        tval = t.ppf(0.975, df=n-2) 
        se = s_yx * np.sqrt(1/n + (xs - xbar)**2 / Sxx)
        ax.fill_between(xs, ys - tval*se, ys + tval*se, alpha=0.18, color=color, linewidth=0)

        txt = f"r = {res.rvalue:.2f}, p {format_p(res.pvalue)}, {stars}"
    else:
        txt = "n<3 or zero variance"

    # symmetric limits around 0 (helps compare panels)
    def sym_lim(arr):
        m = np.nanmax(np.abs(arr))
        return (-1.05*m, 1.05*m) if m > 0 else (-1, 1)

    ax.set_xlim(*sym_lim(x))
    ax.set_ylim(*sym_lim(y))

    # titles / labels / grid
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)

    # stats badge
    ax.text(0.98, 0.02, txt, transform=ax.transAxes,
            ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'), fontsize=10)



def p_to_stars(p, scheme='standard'):
    """
    Return significance stars for a p-value.
    scheme='standard' : * p<0.05, ** p<0.01, *** p<0.001, **** p<1e-4
    scheme='tight'    : ** p<0.005, *** p<0.001, **** p<1e-4  (no single-star)
    """
    if scheme == 'tight':   # if you want 0.005 threshold emphasized
        if p < 1e-4: return '****'
        if p < 1e-3: return '***'
        if p < 5e-3: return '**'
        return 'ns'
    # standard scheme
    if p < 1e-4: return '****'
    if p < 1e-3: return '***'
    if p < 1e-2: return '**'
    if p < 5e-2: return '*'
    return 'ns'

def format_p(p):
    return f"< 1e-3" if p < 1e-3 else f" = {p:.3f}"


def annotate_past_future(ax,
                         corner=(0.86, 0.84),   # where the L-corner sits (x,y) in axes fraction
                         hlen=0.16, vlen=0.22,  # horizontal (left) and vertical (down) lengths
                         color='0.25', fs=11,   # text color & size
                         head=6, lw=1.2,        # arrowhead size & line width
                         ygap=0.035, xgap=0.02  # text offsets (above/beside arrows)
                         ):
    """
    Draw 'Past' (←) and 'Future' (↓) arrows in the upper-right white triangle
    of a Δ heatmap. Past is horizontal leftward; Future is vertical downward.
    """
    xC, yC = corner
    xL = xC - hlen            # left end of the horizontal arrow
    yB = yC - vlen            # bottom end of the vertical arrow

    # ---- Past: horizontal, head on the LEFT
    ax.annotate('', xy=(xL, yC), xytext=(xC, yC),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', mutation_scale=head,
                                lw=lw, color=color))
    ax.text((xL + xC)/2.0, yC + ygap, 'Past',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=fs, fontweight='bold', color=color)

    # ---- Future: vertical, head at the BOTTOM (downward)
    ax.annotate('', xy=(xC, yB), xytext=(xC, yC),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', mutation_scale=head,
                                lw=lw, color=color))
    ax.text(xC + xgap, (yB + yC)/2.0, 'Future',
            transform=ax.transAxes, ha='left', va='center',
            fontsize=fs, fontweight='bold', color=color)































def plot_center_coordinates(days, idle_centers, mi_centers, reducer, directory):
    """
    Plot each component (dimension) of idle_centers vs. mi_centers over time.

    Parameters
    ----------
    days : array-like, shape (n_days,)
        Sequence of day identifiers (could be ints, datetimes, etc.).
    idle_centers : array-like, shape (n_days, n_dims)
        Center coordinates for the 'Idle' condition.
    mi_centers : array-like, shape (n_days, n_dims)
        Center coordinates for the 'MI' condition.
    reducer : str
        Name of the dimensionality reduction method (for the title / filename).
    directory : str
        Path to the folder where the figure will be saved.
    """
    n_dims = idle_centers.shape[1]
    fig, axes = plt.subplots(n_dims, 1,
                             figsize=(8, 2.5 * n_dims),
                             sharex=True)

    # Make sure `axes` is always a list of Axes
    if n_dims == 1:
        axes = [axes]

    for d, ax in enumerate(axes):
        # compute Pearson r between Idle and MI on this component
        r, p = pearsonr(idle_centers[:, d], mi_centers[:, d])

        ax.plot(days, idle_centers[:, d], '-o', label='Idle')
        ax.plot(days, mi_centers[:, d],   '-o', label='MI')
        ax.legend(loc='upper right', title=f"r={r:.2f}")

        ax.set_ylabel(f'Axis {d+1}')
        ax.grid(True, linestyle='--', alpha=0.5)

        if d == 0:
            ax.set_title(f'{reducer} Center Coordinates by Component')

    axes[-1].set_xlabel('Day')
    plt.tight_layout()

    # ensure output directory exists
    os.makedirs(directory, exist_ok=True)
    fname = os.path.join(directory, f'center_coords_{reducer}.png')
    plt.tight_layout()

    plt.savefig(fname, dpi=300)
    plt.close(fig)

def plot_center_distance_curve(days, idle_ctr, mi_ctr, reducer, directory):

    """""
        1) Between‐centers: Idle vs MI distance over days
    2) Within‐centers: day‐to‐day drift for each class (Idle, MI)
    """
    # --- 1) Between‐centers ---
    distances = np.linalg.norm(idle_ctr - mi_ctr, axis=1)
    days_dd = days[1:]   # corresponds to each step

    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(days, distances, '-o', label='Idle–MI distance')
    # optional smoothing
    sm = pd.Series(distances).rolling(5, center=True).mean()
    ax1.plot(days, sm, '--', label='5-day rolling mean', alpha=0.7)

    ax1.set_title(f'{reducer} Center Separation Over Days')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Euclidean distance')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    fig1.tight_layout()
    fn_between = os.path.join(directory,
                              f'center_dist_{reducer}_between_centers.png')
    fig1.savefig(fn_between, dpi=300)
    plt.close(fig1)


    # --- 2) Within‐centers: day‐to‐day drift for each class ---
    # compute day‐to‐day step distances
    idle_dd = np.linalg.norm(idle_ctr[1:] - idle_ctr[:-1], axis=1)
    mi_dd   = np.linalg.norm(mi_ctr[1:]   - mi_ctr[:-1],   axis=1)

    fig2, (ax_idle, ax_mi) = plt.subplots(1, 2, figsize=(12,4), sharey=True)

    # Idle subplot
    ax_idle.plot(days_dd, idle_dd, '-o', color='C0', label='Idle drift')
    sm_idle = pd.Series(idle_dd).rolling(5, center=True).mean()
    ax_idle.plot(days_dd, sm_idle, '--', color='C0', alpha=0.7,
                 label='Idle 5-day mean')
    ax_idle.set_title(f'{reducer} Idle day-to-day drift')
    ax_idle.set_xlabel('Day')
    ax_idle.set_ylabel('ΔEuclidean distance')
    ax_idle.legend(fontsize=10)
    ax_idle.grid(True, linestyle='--', alpha=0.5)

    # MI subplot
    ax_mi.plot(days_dd, mi_dd, '-o', color='C1', label='MI drift')
    sm_mi = pd.Series(mi_dd).rolling(5, center=True).mean()
    ax_mi.plot(days_dd, sm_mi, '--', color='C1', alpha=0.7,
               label='MI 5-day mean')
    ax_mi.set_title(f'{reducer} MI day-to-day drift')
    ax_mi.set_xlabel('Day')
    ax_mi.legend(fontsize=10)
    ax_mi.grid(True, linestyle='--', alpha=0.5)

    fig2.suptitle(f'{reducer} Within-Class Center Drift', fontsize=14)
    fig2.tight_layout(rect=[0, 0, 1, 0.92])

    fn_within = os.path.join(directory,
                             f'center_dist_{reducer}_within_center.png')
    fig2.savefig(fn_within, dpi=300)
    plt.close(fig2)

def plot_center_delta(days, idle_ctr,mi_ctr, reducer, directory, threshold=0.5):

    days = np.asarray(days)

    # --- (1) Between‐centers ---
    # distances between Idle & MI for each day
    between = np.linalg.norm(idle_ctr - mi_ctr, axis=1)
    deltas_between = np.diff(between)

    fig, ax = plt.subplots(figsize=(12,4))
    colors = ['C2' if d > 0 else 'C3' for d in deltas_between]
    ax.bar(days[1:], deltas_between, color=colors, alpha=0.8)

    # threshold lines
    ax.axhline(+threshold, color='k', linestyle='--', linewidth=0.8)
    ax.axhline(-threshold, color='k', linestyle='--', linewidth=0.8)

    ax.set_title(f"{reducer}: Daily Change in Idle–MI Center Separation (ΔDistance)")
    ax.set_xlabel('Day index')
    ax.set_ylabel('ΔDistance (Euclidean units)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_xticks(np.arange(days.min(), days.max()+1, 5))

    # legend
    inc_patch = mpl.patches.Patch(color='C2', label='Increase in separation')
    dec_patch = mpl.patches.Patch(color='C3', label='Decrease in separation')
    thr_line  = mpl.lines.Line2D([], [], color='k', linestyle='--',
                                linewidth=0.8, label=f'Threshold ±{threshold}')
    ax.legend(handles=[inc_patch, dec_patch, thr_line],
              title='Change type', loc='upper right')

    plt.tight_layout()
    fn_between = os.path.join(directory, f'center_delta_{reducer}.png')
    fig.savefig(fn_between, dpi=300)
    plt.close(fig)

    
    # --- (2) Within‐centers: each class’s drift changes ---
    # 2a) compute day‐to‐day drift for each class (first‐order)
    idle_dd = np.linalg.norm(idle_ctr[1:] - idle_ctr[:-1], axis=1)
    mi_dd   = np.linalg.norm(mi_ctr[1:]   - mi_ctr[:-1],   axis=1)

    # 2b) compute daily change of that drift (second‐order)
    idle_delta = np.diff(idle_dd)
    mi_delta   = np.diff(mi_dd)
    days_delta = days[2:]  # aligns with two diffs

    fig_wc, (ax0, ax1) = plt.subplots(1, 2, figsize=(14,4), sharey=True)

    # Idle subplot
    cols0 = ['C2' if d > 0 else 'C3' for d in idle_delta]
    ax0.bar(days_delta, idle_delta, color=cols0, alpha=0.8)
    ax0.axhline(+threshold, color='k', linestyle='--', lw=0.8)
    ax0.axhline(-threshold, color='k', linestyle='--', lw=0.8)
    ax0.set_title(f"{reducer}: Daily Change in Idle-center Drift")
    ax0.set_xlabel('Day index')
    ax0.set_ylabel('ΔEuclidean distance')
    ax0.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax0.set_xticks(np.arange(days.min(), days.max()+1, 5))
    ax0.legend(handles=[inc_patch, dec_patch, thr_line],
               title='Change type', loc='upper right')

    # MI subplot
    cols1 = ['C2' if d > 0 else 'C3' for d in mi_delta]
    ax1.bar(days_delta, mi_delta, color=cols1, alpha=0.8)
    ax1.axhline(+threshold, color='k', linestyle='--', lw=0.8)
    ax1.axhline(-threshold, color='k', linestyle='--', lw=0.8)
    ax1.set_title(f"{reducer}: Daily Change in MI-center Drift")
    ax1.set_xlabel('Day index')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.set_xticks(np.arange(days.min(), days.max()+1, 5))
    ax1.legend(handles=[inc_patch, dec_patch, thr_line],
               title='Change type', loc='upper right')

    fig_wc.suptitle(f"{reducer}: Within-Class Drift Changes", fontsize=14)
    fig_wc.tight_layout(rect=[0,0,1,0.92])
    fn_within = os.path.join(directory,
                             f'center_delta_{reducer}_within_changes.png')
    fig_wc.savefig(fn_within, dpi=300)
    plt.close(fig_wc)


def plot_center_drifts(idle_ctr ,mi_ctr, reducer, directory,figsize=(8,6), bins='auto', font_size=12):

    """
    Histogram of each class center’s distance from Day 1 baseline.
    """
    # unpack
    idle = idle_ctr
    mi   = mi_ctr

    # compute drifts
    idle_drift = np.linalg.norm(idle - idle[0], axis=1)
    mi_drift   = np.linalg.norm(mi   - mi[0], axis=1)

    # shared bins
    all_vals = np.concatenate([idle_drift, mi_drift])
    edges = np.histogram_bin_edges(all_vals, bins=bins)

    # stats for annotation
    stats = {
        'Idle': (idle_drift.mean(), idle_drift.std(),  np.median(idle_drift)),
        'MI':   (mi_drift.mean(),   mi_drift.std(),   np.median(mi_drift))
    }

    # plot
    fig1, ax1 = plt.subplots(figsize=figsize)
    for name, data, color in [('Idle', idle_drift, 'C0'),
                              ('MI',   mi_drift,   'C1')]:
        ax1.hist(data, bins=edges,
                alpha=0.6, color=color,
                edgecolor='k', label=f"{name}  " +
                    f"μ={stats[name][0]:.2f}, σ={stats[name][1]:.2f}")
        # mean lines
        ax1.axvline(stats[name][0], color=color, linestyle='--', lw=1)

    # labels & legend
    ax1.set_title(f"{reducer}: Cumulative Center Drift from Day 1", fontsize=font_size+2)
    ax1.set_xlabel("Distance from Day 1 center", fontsize=font_size)
    ax1.set_ylabel("Number of days",               fontsize=font_size)
    ax1.legend(title="Condition (mean ± std)", fontsize=font_size-1, title_fontsize=font_size-1)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)

    # save
    fn = os.path.join(directory, f'center_drift_hist_{reducer}_overlap.png')
    fig1.tight_layout()
    fig1.savefig(fn, dpi=300)
    plt.close(fig1)


    fig2, (ax_idle, ax_mi) = plt.subplots(1, 2, figsize=(12,5), sharey=True)
    ax_idle.hist(idle_drift, bins=edges, alpha=0.7, color='C0',
                 edgecolor='k',
                 label=f"Idle  μ={stats['Idle'][0]:.2f}, σ={stats['Idle'][1]:.2f}")
    ax_idle.axvline(stats['Idle'][0], color='k', linestyle='--')
    ax_idle.set_title(f"{reducer} Idle drift", fontsize=font_size)
    ax_idle.set_xlabel("Distance from Day 1 center", fontsize=font_size)
    ax_idle.set_ylabel("Count", fontsize=font_size)
    ax_idle.legend(fontsize=font_size-1)
    ax_idle.grid(True, axis='y', linestyle='--', alpha=0.4)

    ax_mi.hist(mi_drift, bins=edges, alpha=0.7, color='C1',
               edgecolor='k',
               label=f"MI  μ={stats['MI'][0]:.2f}, σ={stats['MI'][1]:.2f}")
    ax_mi.axvline(stats['MI'][0], color='k', linestyle='--')
    ax_mi.set_title(f"{reducer} MI drift", fontsize=font_size)
    ax_mi.set_xlabel("Distance from Day 1 center", fontsize=font_size)
    ax_mi.legend(fontsize=font_size-1)
    ax_mi.grid(True, axis='y', linestyle='--', alpha=0.4)

    fig2.suptitle(f"{reducer}: Within-Class Center Drift", fontsize=font_size+2)
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig(os.path.join(directory,  f'center_drift_hist_{reducer}_separate.png'), dpi=300)
    plt.close(fig2)

def plot_center_movements(idle_ctr ,mi_ctr, reducer, directory, figsize=(8,6), bins='auto', font_size=12):

    """
    For each space, histogram the day-to-day step sizes of each class center.
    """
     
    ### ONE PLOT
    idle = idle_ctr 
    mi   = mi_ctr

    # compute movements
    idle_moves = np.linalg.norm(np.diff(idle, axis=0), axis=1)
    mi_moves   = np.linalg.norm(np.diff(mi,   axis=0), axis=1)

    # shared bins
    all_vals = np.concatenate([idle_moves, mi_moves])
    bins = np.histogram_bin_edges(all_vals, bins=bins)

    # stats for annotation
    stats = {
        'Idle': (idle_moves.mean(), idle_moves.std(), np.median(idle_moves)),
        'MI':   (mi_moves.mean(),   mi_moves.std(),   np.median(mi_moves))
    }

    fig, ax = plt.subplots(figsize=figsize)
    for name, data, color in [('Idle', idle_moves, 'C0'),
                              ('MI',   mi_moves,   'C1')]:
        ax.hist(data, bins=bins,
                alpha=0.6, color=color,
                edgecolor='k', label=f"{name}  " +
                    f"μ={stats[name][0]:.2f}, σ={stats[name][1]:.2f}")
        # mean & median lines
        ax.axvline(stats[name][0], color=color, linestyle='--', lw=1)
        ax.axvline(stats[name][2], color=color, linestyle=':',  lw=1)

    # labels & legend
    ax.set_title(f"{reducer}: Day-to-Day Center Movements", fontsize=font_size+2)
    ax.set_xlabel("Euclidean movement per day", fontsize=font_size)
    ax.set_ylabel("Count of day-steps",         fontsize=font_size)
    ax.legend(title="Condition (mean ± std)", fontsize=font_size-1, title_fontsize=font_size-1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    # save
    fn = os.path.join(directory, f'center_moves_hist_{reducer}_overlap.png')
    fig.canvas.manager.full_screen_toggle()
    fig.tight_layout()
    fig.savefig(fn, dpi=300)
    plt.close(fig)


    ###TWO SUBPLOTS
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    bins = 10

    # Idle histogram
    axes[0].hist(idle_moves, bins=bins, alpha=0.7, color='C0',
                    label=f"Idle  μ={stats['Idle'][0]:.2f}, σ={stats['Idle'][1]:.2f}")
    mu_idle = idle_moves.mean()
    axes[0].axvline(mu_idle, color='k', linestyle='--')
    axes[0].set_title(f"{reducer} Idle Drift")
    axes[0].set_xlabel("Euclidean movement per day")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Condition (mean ± std)", fontsize=font_size-1, title_fontsize=font_size-1)
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.4)

    # MI histogram
    axes[1].hist(mi_moves, bins=bins, alpha=0.7, color='C1',
               label=f"MI  μ={stats['MI'][0]:.2f}, σ={stats['MI'][1]:.2f}")
    mu_mi = mi_moves.mean()
    axes[1].axvline(mu_mi, color='k', linestyle='--')                   
    axes[1].set_title(f"{reducer} MI Drift")
    axes[1].set_xlabel("Euclidean movement per day")
    axes[1].legend(title="Condition (mean ± std)", fontsize=font_size-1, title_fontsize=font_size-1)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.4)

    fig.suptitle(f"{reducer} Within-Class Center Drift")
    plt.tight_layout(rect=[0,0,1,0.92])

    # save
    fname = os.path.join(directory, f"center_moves_hist_{reducer}_separate.png")
    plt.savefig(fname, dpi=300)
    plt.close()


def plot_relative_drift_and_cumsum(days, idle_ctr, mi_ctr, reducer, directory):
    """
    For each day-to-day step, decompose change in Idle–MI separation into
    contributions from Idle movement vs. MI movement, and plot as a stacked bar.

    simple explanation:
    each vector (contribute_idle/contribute_mi) sign points if the center is getting close (-) to the other or apart(+).
    the value is how may step it is getting close or apart from the other 

    for example if the centers on 1D is the follwoing:
    idle (0,1,2)
    mi   (5,4,6)

    the contrib_idle will be (-1,-1)
    the contrib_mi will be (-1,2)
    total contirb will be(-2,1) aka the net change
    """
  
    # 1) Compute unit separation vectors u_t for t=0..n-2
    sep_vecs = mi_ctr - idle_ctr                   # (n_days, D)
    norms   = np.linalg.norm(sep_vecs, axis=1, keepdims=True)  # (n_days, 1)
    u       = sep_vecs[:-1] / norms[:-1]           # (n_days-1, D)

    # 2) Day-to-day deltas
    delta_idle = idle_ctr[1:] - idle_ctr[:-1]      # (n_days-1, D)
    delta_mi   = mi_ctr[1:]   - mi_ctr[:-1]        # (n_days-1, D)

    # 3) Projections **along** the separation direction
    proj_idle = -np.sum(delta_idle * u, axis=1)    # NEGATE for idle
    proj_mi   =  np.sum(delta_mi   * u, axis=1)
    net       =  proj_mi - proj_idle               # net change in separation

    # 4) Prepare x-axis (days for the deltas)
    days_delta = days[1:]  # length n_days-1


    # —— Figure 1: daily contributions —— #
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days_delta, proj_idle, '-o', label='Idle Contribution',     color='C0', alpha=0.7)
    ax.plot(days_delta, proj_mi,   '-s', label='MI Contribution',       color='C1', alpha=0.7)
    ax.plot(days_delta, net,       '-^', label='Net Change (MI−Idle)', color='C2', alpha=0.7)

    ax.set_title(f'{reducer} — Relative Drift per Day')
    ax.set_xlabel('Day')
    ax.set_ylabel('Projection Along Separation Axis')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)

    fname = os.path.join(directory, f'relative_drift_{reducer}.png')
    plt.savefig(fname, dpi=300)
    plt.close(fig)

    # —— Figure 2: cumulative contributions —— #
    cumsum_idle = np.cumsum(proj_idle)
    cumsum_mi   = np.cumsum(proj_mi)
    cumsum_net  = np.cumsum(net)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(days_delta, cumsum_idle, '-o', label='Idle Cumsum',     color='C0', alpha=0.7)
    ax2.plot(days_delta, cumsum_mi,   '-s', label='MI Cumsum',       color='C1', alpha=0.7)
    ax2.plot(days_delta, cumsum_net,  '-^', label='Net Cumsum',      color='C2', alpha=0.7)

    ax2.set_title(f'{reducer} — Cumulative Relative Drift')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Cumulative Projection')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)

    fname2 = os.path.join(directory, f'relative_drift_cumsum_{reducer}.png')
    plt.savefig(fname2, dpi=300)
    plt.close(fig2)


def plot_cluster_shape_volume(X, y_label, days_label, reducer, directory):
    """
    Per day & per class, compute:
      • volume = det(covariance of trials)
      • elong = (largest eigenvalue)/(smallest eigenvalue)
    and plot both metrics over days for Idle vs MI.
    """
    if X.ndim != 2 or X.shape[1] < 2:
        return

    days = np.unique(days_label)

    vols_idle, vols_mi = [], []
    elong_idle, elong_mi = [], []

    # NEW: dicts to hold full eigs per day
    eigs_idle_by_day = {}
    eigs_mi_by_day   = {}

    for d in days:
        mask = (days_label == d)

        for label, vols_list, elong_list, eigs_dict in [
            (0, vols_idle, elong_idle, eigs_idle_by_day),
            (1, vols_mi,   elong_mi,   eigs_mi_by_day)
        ]:
            Xd = X[mask & (y_label == label)]
            cov = np.cov(Xd, rowvar=False)
            eigs = np.linalg.eigvalsh(cov)

            # record full spectrum
            eigs_dict[d] = eigs

            # volume ∝ det(cov) = prod(eigenvalues)
            vols_list.append(np.prod(eigs))

            # anisotropy = λ_max / λ_min  (guard zero‐min)
            elong_list.append(eigs[-1] / (eigs[0] if eigs[0]>0 else np.nan))

    # ——— Plot Volume ———
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(days, vols_idle, '-o', label='Idle volume', color='C0')
    ax.plot(days, vols_mi, '-o', label='MI volume', color='C1')
    ax.set_title(f'{reducer}: Cluster Volume Over Time')
    ax.set_xlabel('Day')
    ax.set_ylabel('Det(covariance)')

    ax.legend(loc='upper left')
    ax.grid(ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'{reducer}_cluster_volume.png'), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(days, np.log10(np.abs(vols_idle)), '-o', label='Idle volume (log10)', color='C0')
    ax.plot(days, np.log10(np.abs(vols_mi)), '-o', label='MI volume (log10)', color='C1')
    ax.set_ylabel('log10|Det(covariance)|')
    ax.set_title(f'{reducer}: Cluster Volume Over Time')
    ax.set_xlabel('Day')

    ax.legend(loc='upper left')
    ax.grid(ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'{reducer}log10_cluster_volume.png'), dpi=300)
    plt.close()




    # ——— Plot Elongation ———
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(days, elong_idle, '-o', label='Idle elongation', color='C0')
    ax.plot(days, elong_mi, '-o', label='MI elongation', color='C1')
    ax.set_title(f'{reducer}: Cluster Elongation Over Time')
    ax.set_xlabel('Day')
    ax.set_ylabel('λ_max / λ_min')
    ax.legend(loc='upper left')
    ax.grid(ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'{reducer}_cluster_elongation.png'), dpi=300)
    plt.close()


    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(days, np.log10(elong_idle), '-o', label='Idle elongation (log10)', color='C0')
    ax.plot(days, np.log10(elong_mi), '-o', label='MI elongation (log10)', color='C1')
    ax.set_ylabel('log10(λ_max / λ_min)')
    ax.set_title(f'{reducer}: Cluster Elongation Over Time')
    ax.set_xlabel('Day')
    ax.legend(loc='upper left')
    ax.grid(ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'{reducer}log10_cluster_elongation.png'), dpi=300)
    plt.close()



def plot_cluster_shape_volume_reg(X, y_label, days_label, reducer, directory,eps=1e-6):
    """
    Per day & per class, compute:
      • volume = det(covariance of trials)
      • elong = (largest eigenvalue)/(smallest eigenvalue)
    and plot both metrics over days for Idle vs MI.
    """
    if X.ndim != 2 or X.shape[1] < 2:
        return

    
    days = np.unique(days_label)
    logvol_idle, logvol_mi = [], []
    elong_idle, elong_mi   = [], []

    eigs_idle_by_day = {}
    eigs_mi_by_day   = {}
    bad_days = []
    for d in days:
        mask = (days_label == d)
        for label, lv_list, el_list, eigs_dict in [
            (0, logvol_idle, elong_idle, eigs_idle_by_day),
            (1, logvol_mi,   elong_mi, eigs_mi_by_day)
        ]:
            Xd = X[mask & (y_label == label)]

            if Xd.shape[0] <= Xd.shape[1]:
                lv_list.append(np.nan)
                el_list.append(np.nan)
                bad_days.append(d)

                continue
            # DO SVD VALSE
            cov = np.cov(Xd, rowvar=False)
            eigs = np.linalg.eigvalsh(cov)

            # record full spectrum
            eigs_dict[d] = eigs
            eigs_clipped = np.clip(eigs, eps, None)

            # 1) log-volume via stable slogdet
            sign, logdet = np.linalg.slogdet(cov)
            lv_list.append(logdet)

            # 2) elongation ratio
            el_list.append(eigs_clipped[-1] / eigs_clipped[0])






        # === Figure 1: Volume (log|det|) ===
    fig_vol, ax_vol = plt.subplots(figsize=(8,4))
    ax_vol.plot(days, logvol_idle, '-o', color='C0', label='Idle log-volume')
    ax_vol.plot(days, logvol_mi,   '-o', color='C1', label='MI log-volume')
    ax_vol.set_title(f'{reducer}: Cluster Volume (log|det|)')
    ax_vol.set_xlabel('Day')
    ax_vol.set_ylabel('log|det(covariance)|')
    ax_vol.legend()
    ax_vol.grid(True, linestyle='--', alpha=0.5)
    fig_vol.tight_layout()
    fig_vol.savefig(os.path.join(directory, f'{reducer}_cluster_volume.png'), dpi=300)
    plt.close(fig_vol)

    # === Figure 2: Elongation ===
    fig_el, ax_el = plt.subplots(figsize=(8,4))
    ax_el.plot(days, elong_idle, '-o', color='C0', label='Idle elongation')
    ax_el.plot(days, elong_mi,   '-o', color='C1', label='MI elongation')
    ax_el.set_title(f'{reducer}: Cluster Elongation (λ_max/λ_min)')
    ax_el.set_xlabel('Day')
    ax_el.set_ylabel('λ_max / λ_min')
    ax_el.legend()
    ax_el.grid(True, linestyle='--', alpha=0.5)
    fig_el.tight_layout()
    fig_el.savefig(os.path.join(directory, f'{reducer}_cluster_elongation.png'), dpi=300)
    plt.close(fig_el)



def plot_orientation_change(days, idle_ctr, mi_ctr, reducer, directory):
    """
    Compute the unit separation vector u_t = (mi_ctr - idle_ctr)/||…||
    then plot the day-to-day angle Δθ_t = arccos(u_{t-1}·u_t) in degrees.

    unit vector will encode the direction along which idle vs MI are best separated on day t, irrespective of how far apart they are.
    angle delta of the angle sepereation.

    interpretations: 
    near 0 steady separation
    < 5 normal day to dya noise
    5-15 EEG paaterns are slowly evolving.
    15-30 abrupt change
    """
    # 1) build unit‐length sep vectors
    sep = mi_ctr - idle_ctr                   # shape (n_days, D)
    eps = 1e-8
    norms = np.linalg.norm(sep, axis=1, keepdims=True)
    # avoid division by zero
    norms[norms == 0] = 1
    unit = sep / norms                        # shape (n_days, D)

    # 2) compute angles between successive days
    angles = []
    for t in range(1, len(days)):
        cosang = np.dot(unit[t-1], unit[t])
        # clamp for numerical stability
        cosang = np.clip(cosang, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cosang)))
    
    angle_days = days[1:]

    # 3) plot
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(angle_days, angles, '-o', color='C2', label='Δθ (deg)')
    ax.set_title(f'{reducer}: Separation-axis rotation over time')
    ax.set_xlabel('Day')
    ax.set_ylabel('Angle (°)')
    ax.grid(ls='--', alpha=0.5)
    ax.legend(loc='upper left')
    plt.tight_layout()

    fp = os.path.join(directory, f'{reducer}_orientation_change.png')
    plt.savefig(fp, dpi=300)
    plt.close()


    # 4) now per‐axis changes, each component in its own subplot
    n_comp = unit.shape[1]
    fig, axes = plt.subplots(n_comp, 1, figsize=(6, 2.5*n_comp), sharex=True)
    for d in range(n_comp):
        comp_angles = []
        for t in range(1, len(days)):
            # change in the d-th component along the unit vector
            comp_angles.append((unit[t,d] - unit[t-1,d]))
        axes[d].plot(angle_days, comp_angles, '-o', color=f'C{d%10}')
        axes[d].set_ylabel(f'Δu[{d+1}]')
        axes[d].grid(ls='--', alpha=0.5)
    axes[-1].set_xlabel('Day')
    fig.suptitle(f'{reducer}: Per-axis unit-vector drift')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(directory, f'{reducer}_orientation_per_axis.png'), dpi=300)
    plt.close()



def track_centers(spaces,y_label,days_label, directory):

    os.makedirs(directory, exist_ok=True)
    for name, X in spaces.items():

        centers = utils.compute_class_centers(X, y_label, days_label)
        days       = centers['days']      # shape (n_days,)
        idle_ctr   = centers['idle']      # shape (n_days, D)
        mi_ctr     = centers['mi']        # shape (n_days, D)

        # a) overall separation curve
        # plot_center_distance_curve(days, idle_ctr, mi_ctr, reducer=name, directory=directory)

        # b) per-component coordinate trajectories
        # plot_center_coordinates(days, idle_ctr, mi_ctr, reducer=name, directory=directory)

        # # c) day‐to‐day deltas
        # plot_center_delta(days,  idle_ctr, mi_ctr, reducer=name, directory=directory)

        # # d)  centers drift
        # plot_center_drifts(idle_ctr ,mi_ctr, reducer=name, directory=directory)

        # # e) center movements
        # plot_center_movements(idle_ctr ,mi_ctr, reducer=name, directory=directory)


        # # f) cluster "shape" & "volume"
        plot_cluster_shape_volume(X, y_label, days_label, reducer=name, directory=directory)
        # plot_cluster_shape_volume_reg(X, y_label, days_label, reducer=name, directory=directory)
        # # h)
        # plot_orientation_change(days, idle_ctr, mi_ctr, reducer=name, directory=directory)


