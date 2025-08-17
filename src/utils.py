import random
import umap
import os
import joblib
import pandas as pd
import yaml

import itertools
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import mne
from mne.time_frequency import psd_array_multitaper
from mne.decoding import CSP

from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from scipy.ndimage import uniform_filter1d
from scipy.stats import f_oneway
from scipy.spatial.distance import cdist,euclidean
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.signal import butter, filtfilt, iirnotch

import torch
from torch.utils.data import random_split, DataLoader, Dataset

import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score


from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from mne.decoding import CSP




def analyze_csp_component_usage(pipeline, X_data, y_data, class_names=None, plot_expected_vs_empirical=False):

    # 1. Extract CSP
    csp = pipeline.named_steps['CSP']
    if not hasattr(csp, 'transform'):
        raise ValueError("Pipeline must include a CSP step named 'csp'.")

    # 2. Get number of components and ordering
    n_components = csp.n_components
    order_type = csp.component_order
    log_mode = csp.log
    print(f"\nCSP Config: log={log_mode}, n_components={n_components}, order='{order_type}'")
    # 3. Apply transform
    X_csp = csp.transform(X_data)

    # 4. Get or compute log-variance
    if log_mode:
        log_var = X_csp  # Already log-variance
    else:
        if X_csp.ndim == 3:
            log_var = np.log(np.var(X_csp, axis=2))  # Expected case
        elif X_csp.ndim == 2:
            print("[Warning] log=False but output has shape (n_trials, n_components).")
            print("Assuming log-variance was precomputed or misconfiguration occurred.")
            log_var = X_csp  # Treat as already log-variance
        else:
            raise ValueError(f"Unexpected CSP output shape: {X_csp.shape}")

    # 5. Class-wise averages
    classes = np.unique(y_data)
    if len(classes) != 2:
        raise ValueError("This function assumes binary classification.")
    idx0, idx1 = y_data == classes[0], y_data == classes[1]
    log_var_0 = log_var[idx0]
    log_var_1 = log_var[idx1]
    mean_var_0 = log_var_0.mean(axis=0)
    mean_var_1 = log_var_1.mean(axis=0)

    label0 = class_names[0] if class_names else f"Class {classes[0]}"
    label1 = class_names[1] if class_names else f"Class {classes[1]}"

    # 6. Plot
    plt.figure(figsize=(8, 5))
    plt.plot(mean_var_0, label=label0, marker='o')
    plt.plot(mean_var_1, label=label1, marker='x')
    plt.xlabel("CSP Component Index")
    plt.ylabel("Mean Log-Variance")
    plt.title("CSP Component Variance by Class")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Expected vs empirical report
    print("\nComponent-wise expected vs empirical class dominance:")
    for i in range(n_components):
        expected = label1 if i % 2 == 0 else label0 if order_type == 'alternate' else "unknown"
        empirical = label0 if mean_var_0[i] > mean_var_1[i] else label1
        match = "✓" if expected == empirical else "✗"
        delta = abs(mean_var_0[i] - mean_var_1[i])
        print(f"  C{i+1:2d}: Expected: {expected:>5}, Empirical: {empirical:>5}, Δ={delta:.3f} {match}")

    # Optional: plot expected vs empirical match
    if plot_expected_vs_empirical and order_type == 'alternate':
        _plot_csp_expected_vs_empirical(mean_var_0, mean_var_1, [label0, label1])


def _plot_csp_expected_vs_empirical(mean_var_0, mean_var_1, class_names):
    label0, label1 = class_names
    n_components = len(mean_var_0)

    expected_class = [label1 if i % 2 == 0 else label0 for i in range(n_components)]
    empirical_class = [label0 if v0 > v1 else label1 for v0, v1 in zip(mean_var_0, mean_var_1)]
    diff = np.abs(np.array(mean_var_0) - np.array(mean_var_1))
    colors = ['green' if e == d else 'red' for e, d in zip(expected_class, empirical_class)]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(np.arange(n_components), diff, color=colors)

    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{expected_class[i]} / {empirical_class[i]}", ha='center', fontsize=9)

    plt.xticks(np.arange(n_components), labels=[f"C{i+1}" for i in range(n_components)])
    plt.xlabel("CSP Component")
    plt.ylabel("Δ Log-Variance")
    plt.title("Expected vs Empirical Class Dominance per CSP Component")
    plt.grid(True, axis='y')
    plt.legend(handles=[
        plt.Line2D([0], [0], color='green', lw=6, label='Match'),
        plt.Line2D([0], [0], color='red', lw=6, label='Mismatch')
    ])
    plt.tight_layout()
    plt.show()

class EEGDataSet_signal_by_day(Dataset):
    def __init__(self, EEGDict, days_range=[0,1]):
        # Concat dict
        X, y, days_y = self.concat(EEGDict, days_range)
        X = X.astype(np.float64)
        # Convert from numpy to tensor
        self.X = torch.tensor(X)
        self.n_samples = self.X.shape[0]
        self.n_channels = self.X.shape[1]
        self.y = y
        self.days_y = days_y
        self.n_days_labels = days_range[1] - days_range[0]
        self.n_task_labels = y.shape[1]
        #self.printFlag = printFlag

    def __getitem__(self, index):
        return self.X[index].float(), self.y[index], self.days_y[index]

    def __len__(self):
        return self.n_samples

    def getAllItems(self):
        return self.X.float(), self.y, self.days_y

    def concat(self, EEGDict, days_range):
        X = []
        y = []
        days_y = []
        for day, d in enumerate(EEGDict[days_range[0]:days_range[1]]):
            X.append(d['segmentedEEG'])
            y.append(d['labels'])
            days_y.append(np.ones_like(d['labels']) * day)

        X = np.array(X, dtype=object)
        y = np.array(y, dtype=object)
        X = np.concatenate(X)
        y = np.concatenate(y)
        days_y = np.concatenate(days_y)
        #  one hot encode days labels
        y_temp = np.zeros((days_y.size, days_y.max() + 1))
        y_temp[np.arange(days_y.size), days_y] = 1
        days_y = y_temp
        # One hot encode task labels
        y_temp = np.zeros((y.size, y.max() + 1))
        y_temp[np.arange(y.size), y.astype(int)] = 1

#       y_temp[np.arange(y.size), y] = 1
        y = y_temp
        return X, y, days_y
    
def remove_noisy_trials(dictListStacked, amp_thresh, min_trials):
    # Remove noisy trials using amplitude threshold
    new_dict_list = []
    for i, D in enumerate(dictListStacked):
        max_amp = np.amax(np.amax(D['segmentedEEG'], 2), 1)
        min_amp = np.amin(np.amin(D['segmentedEEG'], 2), 1)
        max_tr = max_amp > amp_thresh
        min_tr = min_amp < -amp_thresh
        noisy_trials = [a or b for a, b in zip(max_tr, min_tr)]
        D['segmentedEEG'] = np.delete(D['segmentedEEG'], noisy_trials, axis=0)
        D['labels'] = np.delete(D['labels'], noisy_trials, axis=0)
        #    # One hot the labels
        #     D['labels'][D['labels']==4] = 3
        #     D['labels'] = torch.as_tensor(D['labels']).to(torch.int64) - 1
        #     D['labels'] = F.one_hot(D['labels'], 3)
        if D['segmentedEEG'].shape[0] > min_trials:
            new_dict_list.append(D)
    return new_dict_list

def eegFilters(eegMat, fs, filterLim):
    eegMatFiltered = mne.filter.filter_data(eegMat, fs, filterLim[0], filterLim[1], verbose='warning')
    return eegMatFiltered

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)

def notch_filter(data, fs, freq=35.0, Q=30):
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, data, axis=1)

def detect_artifacts(EEG, fs, amp_thresh_uV=100, std_thresh=5, bandpass=(1, 40),notch_freq=35, window_sec=2):
    
    EEG_filt = bandpass_filter(EEG, fs, bandpass[0], bandpass[1])

    EEG_filt = notch_filter(EEG_filt, fs, freq=notch_freq, Q=30)

    # --- Amplitude Thresholding ---
    amp_artifacts = np.any(np.abs(EEG_filt) > amp_thresh_uV, axis=0)

    # --- Statistical (Standard Deviation) Thresholding ---
    win_len = int(window_sec * fs)
    std_artifacts = np.zeros(EEG.shape[1], dtype=bool)
    for ch in range(EEG.shape[0]):
        rolling_std = np.convolve(
            np.abs(EEG_filt[ch]),
            np.ones(win_len) / win_len,
            mode='same'
        )
        channel_std = np.std(EEG_filt[ch])
        std_artifacts_ch = rolling_std > (std_thresh * channel_std)
        std_artifacts = np.logical_or(std_artifacts, std_artifacts_ch)

    # --- Combine artifact detections ---
    artifacts = np.logical_or(amp_artifacts, std_artifacts).astype(int)
    return artifacts

def preprocess_dataset(dataset, denoiser):
    """
    Apply denoising and center the EEG signals.
    
    Args:
        dataset: EEGDataSet instance
        denoiser: Initialized and loaded Denoiser object

    Returns:
        signal_centered: Denoised and centered EEG signals (trials, channels, time)
        y_labels: Condition labels per trial
        day_labels: Day index labels per trial
    """
    signal, y_onehot, day_onehot, _ = denoiser.denoise(dataset)
    y_labels = np.argmax(y_onehot, axis=1)
    day_labels = np.argmax(day_onehot, axis=1) + 1
    means = np.mean(signal, axis=2, keepdims=True)  # Shape: (536, 11, 1)
    signal_centered = signal - means
    return signal_centered, y_labels, day_labels

def set_seed(seed):
    torch.manual_seed(seed)  # for CPU
    torch.cuda.manual_seed(seed)  # for GPU
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
    np.random.seed(seed)  # for numpy
    random.seed(seed)  # for random module
    torch.backends.cudnn.deterministic = True  # ensures reproducibility for CUDA backend
    torch.backends.cudnn.benchmark = False  # disabling this might slow down training but ensures reproducibility

def transform_csp(X, filters, n_components=1, mean_=None, std_=None):
    pick_filters = filters[:n_components]
    X_proj = np.asarray([np.dot(pick_filters, epoch) for epoch in X])
    X_var = (X_proj ** 2).mean(axis=2)

    if mean_ is not None and std_ is not None:
        X_var = (X_var - mean_) / std_
    return X_var


def log_metric_correlations(metric_name, directory, start_test_day, end_test_day, reducer, dim, 
                            corr_inter, corr_intra_idle, corr_intra_motor):

    # Construct window label
    label = f"{start_test_day+30}-{end_test_day+30}"

    # Format row to append
    row = {
        'Window': label,
        'Inter-cluster': corr_inter,
        'Intra-Idle': corr_intra_idle,
        'Intra-Motor Imgarey': corr_intra_motor
    }

    # Filename based on settings
    filename = f"{dim}D_{reducer}_{metric_name}_correlations.xlsx"
    filepath = os.path.join(directory, filename)

    # Load or create DataFrame
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
    else:
        df = pd.DataFrame(columns=row.keys())

    # Append and save
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(filepath, index=False)

    return filepath


def plot_2d_embedding(transformer,X_csp,labels, title, save_path):
    if transformer.n_components != 2:
        raise ValueError("Use 3D plot for 3-component transforms")
        
    X_red = transformer.transform(X_csp)
    # Trials to exclude (e.g., those with extreme PCA values for sub201)
    # outliers = [95,102, 362]

    # Create a mask for non-outliers
    mask = np.ones(X_red.shape[0], dtype=bool)
    # mask[outliers] = False

    # Apply the mask
    X_red_clean = X_red[mask]
    y_clean = labels[mask]

    plt.figure(figsize=(12,6))
    scatter = plt.scatter(X_red_clean[:,0], X_red_clean[:,1], c=y_clean, cmap='viridis', alpha=0.7)
    plt.title(f'{title} (Explained Variance: {transformer.explained_variance_ratio_.sum():.2f})'
              if hasattr(transformer, 'explained_variance_ratio_') else title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        filename = f'{title}_training_feature_space.jpg'
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path)
    plt.show()
    plt.close()

def plot_3d_embedding(transformer,X_csp,labels, title, save_path):
    X_red = transformer.transform(X_csp)
    # Trials to exclude (e.g., those with extreme PCA values for sub201)
    # outliers = [95,102, 362]

    # Create a mask for non-outliers
    mask = np.ones(X_red.shape[0], dtype=bool)
    # mask[outliers] = False

    # Apply the mask
    X_red_clean = X_red[mask]
    y_clean = labels[mask]

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_red_clean[:, 0], X_red_clean[:, 1], X_red_clean[:, 2],
                         c=y_clean, cmap='viridis', alpha=0.7, edgecolor='k')
    plt.title(f'{title} (Explained Variance: {transformer.explained_variance_ratio_.sum():.2f})'
             if hasattr(transformer, 'explained_variance_ratio_') else title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    fig.colorbar(scatter, ax=ax, label='Class')
    plt.tight_layout()
    if save_path:
        filename = f'{title}_training_feature_space.jpg'
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path)
    plt.show()

def grid_search_alternate_csp(signal, labels,
                              n_components_list=(2, 4, 6, 8, 10),
                              save_path='model_weights/'):
    """
    1) For each n_components:
         • CSP(..., component_order='alternate') + LDA
         • 5‐fold CV → mean±std accuracy
         • fit full data → extract CSP.eigen_values_ & CSP.transform(signal)
         • save as model_csp_lda_{n_components}.pkl
    2) Pick the best by mean CV accuracy, save it as best_model_{n_components}.pkl
    3) On that best model’s CSP features, fit & save:
         PCA2D, UMAP2D, PCA3D, UMAP3D, and the features array
    Returns:
      models       : {n_components: fitted Pipeline}
      cv_results   : {n_components: {'mean_accuracy','std_accuracy'}}
      eigen_values : {n_components: array of length n_components}
      csp_features : {n_components: array (n_trials, n_components)}
    """
    os.makedirs(save_path, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models       = {}
    cv_results   = {}
    eigen_values = {}
    eigen_vectors = {}
    csp_features = {}
    n_channels = signal.shape[1]
    valid_n  = [n for n in n_components_list if n <= n_channels]
    if len(valid_n) < len(n_components_list):
        print(f"Removed invalid components: {set(n_components_list)-set(valid_n)}")
    signal = signal.astype(np.float64)

    # 1) grid
    for n_comp in n_components_list:
        # pipeline
        csp = CSP(n_components=n_comp,
                  reg='ledoit_wolf',
                  log=False,
                  norm_trace=True,
                  component_order='alternate')
        lda = LinearDiscriminantAnalysis()
        pipe = Pipeline([('CSP', csp), ('LDA', lda)])

        # CV
        scores = cross_val_score(pipe, signal, labels,
                                 cv=skf, scoring='accuracy', n_jobs=1)
        mean_acc, std_acc = scores.mean(), scores.std()

        # fit full data to grab eigenvals & features

        pipe.fit(signal, labels)
        covs, sample_weights = pipe.named_steps['CSP']._compute_covariance_matrices(signal, labels)
        vectors, values = pipe.named_steps['CSP']._decompose_covs(covs, sample_weights)
        ix = pipe.named_steps['CSP']._order_components(covs, sample_weights, vectors, values, pipe.named_steps['CSP'].component_order)
        eig_vec = vectors[:, ix]
        eig_val = values [ix]

        Xc = pipe.named_steps['CSP'].transform(signal)

        # save model
        fname = f'model_csp_lda_{n_comp}.pkl'
        joblib.dump(pipe, os.path.join(save_path, fname))

        # stash
        models[n_comp]       = pipe
        cv_results[n_comp]   = {'mean_accuracy': mean_acc,
                                'std_accuracy' : std_acc}
        eigen_vectors[n_comp] = eig_vec[:n_comp]
        eigen_values[n_comp] = eig_val [:n_comp]
        csp_features[n_comp] = Xc

    # 2) pick best
    best_n = max(cv_results,
                 key=lambda k: cv_results[k]['mean_accuracy'])
    # for sub201:
    # because 4, 6 ,8 components had more or less the same accuracy and for technical reasons such as number of trials per class  vs number of features, i chose the middle option
    # therefore, from now on best_n = 6
    # please check if your data have different scores and for that reasons you should change the number of components.

    # for sub205
    # n=6 was the optimal choice

    # for sub206 
    # n=10 was the optimal choice
    


    best_model = models[best_n]
    joblib.dump(best_model,
                os.path.join(save_path, f'best_model_{best_n}.pkl'))

    # 3) build & save reducers on best CSP features

    x_best = csp_features[best_n]

    # PCA 2D
    pca2D = PCA(n_components=2)
    pca2D.fit(x_best)
    joblib.dump(pca2D,
                os.path.join(save_path, 'pca2D_reducer.pkl'))
    
    # plot_2d_embedding(pca2D,x_best,labels, "PCA 2D", save_path)

    # UMAP 2D
    umap2D = umap.UMAP(n_components=2, random_state=42)
    umap2D.fit(x_best)
    joblib.dump(umap2D,
                os.path.join(save_path, 'umap2D_reducer.pkl'))

    # PCA 3D
    pca3D = PCA(n_components=3)
    pca3D.fit(x_best)
    joblib.dump(pca3D,
                os.path.join(save_path, 'pca3D_reducer.pkl'))
    
    # plot_3d_embedding(pca3D,x_best,labels, "PCA 3D", save_path)

    # UMAP 3D
    umap3D = umap.UMAP(n_components=3, random_state=42)
    umap3D.fit(x_best)
    joblib.dump(umap3D,
                os.path.join(save_path, 'umap3D_reducer.pkl'))

    # and save the CSP features eigen values, eigen vectors
    joblib.dump(x_best,
                os.path.join(save_path, f'x_csp_features_train_{best_n}.pkl'))
    
    joblib.dump(eigen_values,
                os.path.join(save_path, 'eigen_values.pkl'))
    joblib.dump(eigen_vectors,
                os.path.join(save_path, 'eigen_vectors.pkl'))

    joblib.dump(cv_results,
                os.path.join(save_path, 'cv_results.pkl'))
    

    print(f"Best CSP‐LDA has {best_n} components "
          f"(mean accuracy={cv_results[best_n]['mean_accuracy']:.3f}).")
    print("PCA/UMAP reducers and CSP features saved.")

    return models, cv_results, eigen_values, eigen_vectors,  csp_features

        
def  old_grid_search_clf(signal, labels, save_path='model_weights/'):

    # Manual search
    mne.set_log_level(verbose='WARNING')
    signal = np.asarray(signal, dtype=np.float64)

    # Parameter grid
    param_grid = {
        'n_components': range(2, 12),
        'reg': ['empirical', 'ledoit_wolf','diagonal_fixed']
    }

    best_score = 0
    best_model = None
    best_params = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = {}

    for n_comp in param_grid['n_components']:
        for reg in param_grid['reg']:
                fold_scores = []

                for train_idx, test_idx in skf.split(signal, labels):
                    X_train, X_test = signal[train_idx], signal[test_idx]
                    y_train, y_test = labels[train_idx], labels[test_idx]

                    csp = CSP(n_components=n_comp, reg=reg, log=False, norm_trace=True)
                    lda = LinearDiscriminantAnalysis()
                    pipeline = Pipeline([('CSP', csp), ('LDA', lda)])

                    pipeline.fit(X_train, y_train)
                    preds = pipeline.predict(X_test)
                    score = accuracy_score(y_test, preds)
                    fold_scores.append(score)

                avg_score = np.mean(fold_scores)
                param_key = f'n_components={n_comp}, reg={reg}'
                all_results[param_key] = avg_score

                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {'CSP__n_components': n_comp, 'CSP__reg': reg}
                    best_model = pipeline


    # Automatic scikit-learn search

    num_comp = range(2,12)
    # Define the CSP-LDA pipeline
    csp = CSP(n_components=6, reg=None, log=False, norm_trace=True)
    lda = LinearDiscriminantAnalysis(solver='svd')

    clf = Pipeline([('CSP', csp), ('LDA', lda)])

    # Define parameter grid
    param_grid = {
        'CSP__n_components': num_comp,  # Try different number of components
        'CSP__reg': [None, 'ledoit_wolf', 'oas'],  # Test different regularization methods
        'LDA__solver': ['svd', 'lsqr'],  # Different LDA solvers can have an impact
    }


    # Perform grid search
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=1,error_score='raise')
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    grid_search.fit(signal, labels)

    # Get the best model and accuracy
    best_model_grid = grid_search.best_estimator_
    best_score_grid = grid_search.best_score_


    joblib.dump(best_model_grid, save_path + 'best_csp_lda_model.pkl')
    joblib.dump(best_score_grid, save_path + 'best_accuracy.pkl')

    # Save the best model and accuracy
    joblib.dump(all_results, save_path + 'all_grid_results.pkl')
    joblib.dump(best_model, save_path + 'best_csp_lda_model_manual.pkl')
    joblib.dump(best_score, save_path + 'best_accuracy_manual.pkl')
    
    print(f"Best params: {best_params}")
    print(f"Best score: {best_score}")

    # manual grid one epoch
    # Best params: {'CSP__n_components':6 , 'CSP__reg': 'empirical'}
    # Best score: 0.7331429560401522  


    # Transform signal using CSP
    x_csp_features = best_model.named_steps['CSP'].transform(signal)
    
    # Fit PCA on CSP features
    pca2D = PCA(n_components=2)
    pca2D.fit(x_csp_features)
    
    # Fit UMAP on CSP features
    umap2D_reducer = umap.UMAP(n_components=2,random_state=42)
    umap2D_reducer.fit(x_csp_features)


    # Fit PCA on CSP features
    pca3D = PCA(n_components=3)
    pca3D.fit(x_csp_features)
    
    # Fit UMAP on CSP features
    umap3D_reducer = umap.UMAP(n_components=3, random_state=42)
    umap3D_reducer.fit(x_csp_features)

    # Save PCA and UMAP reducers
    joblib.dump(pca2D, save_path + 'pca2D_reducer.pkl')
    joblib.dump(umap2D_reducer, save_path + 'umap2D_reducer.pkl')
    joblib.dump(pca3D, save_path + 'pca3D_reducer.pkl')
    joblib.dump(umap3D_reducer, save_path + 'umap3D_reducer.pkl')
    joblib.dump(x_csp_features, save_path + 'x_csp_features.pkl')

    print("PCA and UMAP fit and saved successfully.")

    n_neighbors_list = [7, 10,22]
    min_dist_list = [0.001, 0.01,0.05]

    # Create combinations
    param_combinations = list(itertools.product(n_neighbors_list, min_dist_list))

    # Iterate over each combination
    for n_neighbors, min_dist in param_combinations:
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        X_umap = reducer.fit_transform(x_csp_features)

        # Plot the result
        plt.figure(figsize=(5, 4))
        plt.scatter(X_umap[labels == 0, 0], X_umap[labels == 0, 1], label='Idle', alpha=0.7)
        plt.scatter(X_umap[labels == 1, 0], X_umap[labels == 1, 1], label='MI', alpha=0.7, marker='x')
        plt.title(f'UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # # Optionally save reducer
        # reducer_name = f'umap2D_nn{n_neighbors}_md{min_dist}.pkl'
        # joblib.dump(reducer, save_path + reducer_name)


    return best_model, best_score, x_csp_features, pca2D, umap2D_reducer, pca3D, umap3D_reducer
    #200 epochs
    # Best params: {'CSP__n_components': 4, 'CSP__reg': 'ledoit_wolf'}
    # Best score: 0.7238663897542401

def apply_pca(X, pca_model, n_components=2):
    X_pca = pca_model.transform(X)  # Apply PCA transformation
    return X_pca

def apply_umap(X, umap_model, n_components=2):
    X_umap = umap_model.transform(X)  # Apply UMAP transformation
    return X_umap

def evaluate_classifier(clf_loaded, X, y_label):
    y_pred = clf_loaded.predict(X)  # Use pre-trained classifier
    accuracy = accuracy_score(y_label, y_pred)
    return accuracy

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, axis=0, ddof=1), np.var(group2, axis=0, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    mean_diff = np.mean(group1, axis=0) - np.mean(group2, axis=0)
    return mean_diff / pooled_std

def calculate_cluster_variance(X, y_label):
    unique_labels = np.unique(y_label)
    intra_distances = {}
    for label in unique_labels:
        label_mask = (y_label == label)
        X_label = X[label_mask]
        
        # Use sklearn's pairwise_distances for faster computation
        distances = pairwise_distances(X_label, metric='euclidean')
        upper_tri = distances[np.triu_indices_from(distances, k=1)]
        intra_distances[label] = np.mean(upper_tri) if len(upper_tri) > 0 else 0.0
        
    # Separate data for idle and motor imagery
    idle_data = X[y_label == 0]
    motor_imagery_data = X[y_label == 1]
    cohens_d_values = cohens_d(motor_imagery_data, idle_data) #  accounts for potentially unequal sample sizes and uses the unbiased sample variance estimator
    inter_cluster_distance = np.mean(np.abs(cohens_d_values))

    return {
        'intra_distances': {
            'idle': intra_distances[0],
            'motor_imagery': intra_distances[1]
        },
        'inter_distance': inter_cluster_distance
    }


from sklearn.metrics import pairwise_distances
import numpy as np

def calculate_cluster_variance_window(X, y_label):
    """
    Returns NaN for any intra- or inter-cluster distance that cannot
    be computed (e.g., fewer than 2 samples).
    """
    unique_labels = np.unique(y_label)
    
    # 1) Initialize both classes to NaN
    intra_distances = {0: np.nan, 1: np.nan}
    
    # 2) Compute intra‐cluster distances when possible
    for label in unique_labels:
        label_mask = (y_label == label)
        X_label = X[label_mask]
        
        # need at least 2 samples to compute pairwise distances
        if X_label.shape[0] > 1:
            distances = pairwise_distances(X_label, metric='euclidean')
            upper_tri = distances[np.triu_indices_from(distances, k=1)]
            intra_distances[label] = np.mean(upper_tri)
        # else leave as np.nan
    
    # 3) Separate idle vs. motor imagery
    idle_data          = X[y_label == 0]
    motor_imagery_data = X[y_label == 1]
    
    # 4) Compute inter‐cluster distance only if both classes have ≥2 samples
    if idle_data.shape[0] >= 2 and motor_imagery_data.shape[0] >= 2:
        cohens_d_values       = cohens_d(motor_imagery_data, idle_data)
        inter_cluster_distance = np.mean(np.abs(cohens_d_values))
    else:
        inter_cluster_distance = np.nan
    
    return {
        'intra_distances': {
            'idle':            intra_distances[0],
            'motor_imagery':   intra_distances[1]
        },
        'inter_distance': inter_cluster_distance
    }


def compute_acc_inter_intra_var(X_csp, y_label, days_labels, clf_loaded):
    unique_days = np.unique(days_labels)
    accuracies = []
    auc_scores = []
    inter_variances = []
    intra_variances_idle = []
    intra_variances_motor = []
    lda_loaded = clf_loaded.named_steps['LDA']

    for day in unique_days:
        day_mask = (days_labels == day)
        X_day = X_csp[day_mask]
        y_day = y_label[day_mask]

        # Evaluate the classifier for accuracy on this day's data
        accuracy = evaluate_classifier(lda_loaded, X_day, y_day)
        accuracies.append(accuracy)

        scores_day = lda_loaded.decision_function(X_day)
        auc = roc_auc_score(y_day, scores_day)
        auc_scores.append(auc)

        # Calculate intra-cluster and inter-cluster distances
        cluster_variances = calculate_cluster_variance(X_day, y_day)
        
        # Append intra-cluster distances for idle and motor imagery
        intra_variances_idle.append(cluster_variances['intra_distances']['idle'])
        intra_variances_motor.append(cluster_variances['intra_distances']['motor_imagery'])
        
        # Append inter-cluster distance
        inter_variances.append(cluster_variances['inter_distance'])

    # Convert lists to numpy arrays
    accuracies = np.array(accuracies)
    inter_variances = np.array(inter_variances)
    intra_variances_idle = np.array(intra_variances_idle)
    intra_variances_motor = np.array(intra_variances_motor)

    return accuracies,auc_scores, inter_variances, intra_variances_idle, intra_variances_motor


def find_best_window_size(accuracies, variances, min_window=2, max_window=10):
    best_window_size = min_window
    best_corr_abs = 0  # Track the largest absolute correlation
    best_corr = 0      # Track the actual correlation value

    for window_size in range(min_window, max_window + 1):
        smoothed_accuracies = np.convolve(accuracies, np.ones(window_size)/window_size, mode='valid')
        smoothed_variances = np.convolve(variances, np.ones(window_size)/window_size, mode='valid')
        corr, _ = pearsonr(smoothed_accuracies, smoothed_variances)

        if abs(corr) > best_corr_abs:
            best_corr_abs = abs(corr)  # Update the largest absolute correlation
            best_corr = corr           # Store the actual correlation (with sign)
            best_window_size = window_size

    return best_window_size, best_corr,_

def compute_class_centers(X, y_label, days_label):
    """
    Compute per-day class centers (mean feature vectors) for each class.

    Parameters
    ----------
    X : array, shape (n_trials, n_features)
        Feature matrix for all trials.
    y_label : array, shape (n_trials,)
        Class labels (0=Idle, 1=MI) for each trial.
    days_label : array, shape (n_trials,)
        Day index for each trial.

    Returns
    -------
    centers : dict
        {
          'days': array, shape (n_days,),  # sorted unique days
          'idle': array, shape (n_days, n_features),
          'mi':   array, shape (n_days, n_features)
        }
    """
    unique_days = np.unique(days_label)
    days = []
    idle_centers = []
    mi_centers = []
    for day in unique_days:
        day_mask = (days_label == day)
        X_day = X[day_mask]
        y_day = y_label[day_mask]
        # compute means
        idle_centers.append(X_day[y_day == 0].mean(axis=0))
        mi_centers.append(X_day[y_day == 1].mean(axis=0))
        days.append(day)
    return {
        'days': np.array(days),
        'idle': np.vstack(idle_centers),
        'mi':   np.vstack(mi_centers)
    }

def estimate_covariance(Xd, force_shrink=False, ridge_alpha=1e-6):
    """
    Estimate Σ from samples Xd (shape [n_trials, n_dims]).
    If force_shrink or if n_trials ≤ n_dims, apply Ledoit–Wolf shrinkage.
    Otherwise fall back to the sample covariance (with an optional tiny ridge).
    """
    n, d = Xd.shape
    do_shrink = force_shrink or (n <= d)
    if do_shrink:
        # try Ledoit–Wolf first
        try:
            lw = LedoitWolf().fit(Xd)
            return lw.covariance_
        except Exception:
            # fallback to simple ridge
            cov = np.cov(Xd, rowvar=False)
            eps = ridge_alpha * np.trace(cov)
            return cov + eps * np.eye(d)
    else:
        cov = np.cov(Xd, rowvar=False)
        # still guard against exact zeros if you like
        eps = ridge_alpha * np.trace(cov)
        return cov + eps * np.eye(d)


def calculate_t_statistic(X, y, lda_model):
    y_pred = lda_model.predict(X)
    t_stat, _ = stats.ttest_1samp(y == y_pred, 0.5)
    return t_stat

def t_max_permutation_test(X_csp, y_label, days_labels, clf_loaded,directory, n_permutations=1000, random_state=42):
    # Set random seed for reproducibility
    np.random.seed(random_state)
    significance_level = 0.05

    # Use the LDA model directly from the preloaded classifier pipeline
    lda_loaded = clf_loaded.named_steps['LDA']
    
    # Get the unique days
    unique_days = np.unique(days_labels)
    unique_days_shifted = unique_days + 30  # Adjust day labels to start from day 30

    # Initialize lists to store the original accuracies and permuted accuracies
    original_t_stats = []
    permuted_max_list = []   
    # Evaluate classifier on the original data for each day
    for day in unique_days:
        day_mask = (days_labels == day)
        X_day = X_csp[day_mask]
        y_day = y_label[day_mask]
        
        # Compute accuracy for the true labels
        t_stat = calculate_t_statistic(X_day, y_day, lda_loaded)
        original_t_stats.append(t_stat)

    original_t_stats = np.array(original_t_stats)

    # Perform permutations for all days
    for _ in range(n_permutations):
        permuted_t_stats = []
        
        # Shuffle labels and calculate accuracy for each day
        for day in unique_days:
            day_mask = (days_labels == day)
            X_day = X_csp[day_mask]
            y_day = y_label[day_mask]
            
            # Shuffle the labels for this day
            permuted_labels = np.random.permutation(y_day)
            permuted_t_stat = calculate_t_statistic(X_day, permuted_labels, lda_loaded)
            permuted_t_stats.append(permuted_t_stat)
        permuted_max_list.append(np.max(permuted_t_stats))

    permuted_max_list = np.array(permuted_max_list)
    p_values = np.mean(permuted_max_list >= original_t_stats[:, np.newaxis], axis=1)



    plt.figure(figsize=(12, 8))
    plt.plot(unique_days_shifted, original_t_stats, label='Original t-statistic', color='blue', marker='o')
    significant_days = unique_days_shifted[p_values < significance_level]
    significant_t_stats = original_t_stats[p_values < significance_level]
    plt.scatter(significant_days, significant_t_stats, color='red', label='Significant Days (p < 0.05)', s=100)
    plt.title("Original t-statistics Across Days with Significant Days Highlighted")
    plt.xlabel("Days")
    plt.ylabel("t-statistic")
    plt.legend()
    plt.tight_layout()
    if directory:
        filename = f'T_Max_Permutation_Test_Results_Days_{unique_days[0]}_to_{unique_days[-1]}.jpg'
        full_path = os.path.join(directory, filename)
        plt.savefig(full_path)
    plt.show()
    plt.close()

    return {
        'original_t_stats': original_t_stats,
        'permuted_max_distribution': permuted_max_list,
        'p_values': p_values,
        'significant_days': significant_days
    }

def analyze_psd_over_time(eeg_data, days_labels, unique_days, directory,freq=128):

    mu_band = (8, 13)  # Mu rhythm range in Hz
    beta_band = (13, 30)  # Beta rhythm range in Hz

    avg_mu_power = []
    avg_beta_power = []

    for day in unique_days:
        day_mask = (days_labels == day)
        X_day = eeg_data[day_mask]

        # Option 1: Averaging across trials
        X_day_avg = np.mean(X_day, axis=0)  # Shape: (11, 768), averaging across trials from the same day

        # Calculate PSD using multitaper method for mu and beta bands
        psd, freqs = psd_array_multitaper(X_day_avg,freq, fmin=mu_band[0], fmax=beta_band[1], verbose=False)
        
        # Average the power in each band
        mu_idx = np.logical_and(freqs >= mu_band[0], freqs <= mu_band[1])
        beta_idx = np.logical_and(freqs >= beta_band[0], freqs <= beta_band[1])
        
        avg_mu_power.append(np.mean(psd[:, mu_idx]))
        avg_beta_power.append(np.mean(psd[:, beta_idx]))


    # Calculate the correlation
    correlation, _ = pearsonr(avg_mu_power, avg_beta_power)

    # Plot Mu and Beta power over time
    days = np.array(unique_days) + 30
    plt.figure(figsize=(10, 6))
    plt.plot(days, avg_mu_power, label='Mu Band Power', color='blue')
    plt.plot(days, avg_beta_power, label='Beta Band Power', color='green')
    plt.title('Mu and Beta Band Power Over Time')
    plt.xlabel('Days')
    plt.ylabel('Average Power')
    plt.legend()
    plt.grid(True)

    filename = 'Mu_Beta_Power_Over_Time_noisy_signal.jpg'
    full_path = os.path.join(directory, filename)
    plt.savefig(full_path)
    plt.close()

    # Scatter plot of avg_mu_power vs avg_beta_power
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(avg_mu_power, avg_beta_power, color='purple', alpha=0.7)
    plt.title('Scatter Plot of Mu vs Beta Power')
    plt.xlabel('Average Mu Band Power')
    plt.ylabel('Average Beta Band Power')

    # Add legend with correlation information
    plt.legend([scatter], [f"Correlation between Mu and Beta: {correlation:.2f}"], loc='lower right')

    plt.grid(True)
    # Save the scatter plot
    scatter_filename = 'Mu_vs_Beta_Power_Scatter.jpg'
    scatter_full_path = os.path.join(directory, scatter_filename)
    plt.savefig(scatter_full_path)
    plt.close()

    return np.array(avg_mu_power), np.array(avg_beta_power)

def compare_early_vs_late(accuracies, inter_distances, intra_distances_idle, intra_distances_motor, threshold, directory):
    # Split into early and late phase based on threshold
    # Split into early and late phase based on threshold
    early_phase_acc = accuracies[:threshold]
    late_phase_acc = accuracies[73:103] # last 30 days
    
    early_phase_inter = inter_distances[:threshold]
    late_phase_inter = inter_distances[73:103]
    
    early_phase_intra_idle = intra_distances_idle[:threshold]
    late_phase_intra_idle = intra_distances_idle[-30:]

    early_phase_intra_motor = intra_distances_motor[:threshold]
    late_phase_intra_motor = intra_distances_motor[-30:]
    
    # Perform t-test
    t_stat_acc, p_val_acc = ttest_ind(early_phase_acc, late_phase_acc)
    t_stat_inter, p_val_inter = ttest_ind(early_phase_inter, late_phase_inter)
    t_stat_intra_idle, p_val_intra_idle = ttest_ind(early_phase_intra_idle, late_phase_intra_idle)
    t_stat_intra_motor, p_val_intra_motor = ttest_ind(early_phase_intra_motor, late_phase_intra_motor)

    # Plot each in a separate subplot
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    plt.boxplot([early_phase_acc, late_phase_acc], labels=['Early', 'Late'])
    plt.title(f'Accuracy\nT: {t_stat_acc:.2f}, p: {p_val_acc:.4f}')
    plt.ylabel('Accuracy')

    plt.subplot(2, 2, 2)
    plt.boxplot([early_phase_inter, late_phase_inter], labels=['Early', 'Late'])
    plt.title(f'Inter-Cluster Distance T: {t_stat_inter:.2f}, p: {p_val_inter:.4f}')
    plt.ylabel('Distance (Cohen\'s d)')

    plt.subplot(2, 2, 3)
    plt.boxplot([early_phase_intra_idle, late_phase_intra_idle], labels=['Early', 'Late'])
    plt.title(f'Intra-Cluster Distance Idle T: {t_stat_intra_idle:.2f}, p: {p_val_intra_idle:.4f}')
    plt.ylabel('Distance [std]')

    plt.subplot(2, 2, 4)
    plt.boxplot([early_phase_intra_motor, late_phase_intra_motor], labels=['Early', 'Late'])
    plt.title(f'Intra-Cluster Distance Motor Imagery T: {t_stat_intra_motor:.2f}, p: {p_val_intra_motor:.4f}')
    plt.ylabel('Distance [std]')

    filename = 'compare_early_vs_late.jpg'
    full_path = os.path.join(directory, filename)
    plt.savefig(full_path)
    plt.close()

def compare_early_vs_late_all(
    metrics_data,
    band_data,
    early_end,
    late_start,
    directory
):
    """
    For each array in metrics_data and band_data, perform an independent
    t-test comparing Early (0:early_end) vs Late (late_start:) phases,
    plot side-by-side boxplots, save figures, and return the stats.

    Parameters
    ----------
    metrics_data : dict of name→1D array
        e.g. {'Accuracy': accuracies, 'Inter-Dist': inter_variances, ...}
    band_data    : dict of name→1D array
        e.g. {'Mu Power': avg_mu_power, 'Beta Power': avg_beta_power}
    early_end    : int
        Index at which the Early phase ends (exclusive).
    late_start   : int
        Index at which the Late phase starts; can be negative (e.g. -30).
    directory    : str
        Where to save the figures.

    Returns
    -------
    results : dict
        {
          'metrics': { name: (t_stat, p_val), ... },
          'bands':   { name: (t_stat, p_val), ... }
        }
    """
    os.makedirs(directory, exist_ok=True)

    results = {'metrics': {}, 'bands': {}}

    # ─── Metrics figure (grid) ────────────────────────────────────────────────
    n_metrics = len(metrics_data)
    cols = 2
    rows = int(np.ceil(n_metrics / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, (name, arr) in zip(axes_flat, metrics_data.items()):
        early_vals = arr[:early_end]
        late_vals  = arr[late_start:]
        t_stat, p_val = ttest_ind(early_vals, late_vals, equal_var=False)
        results['metrics'][name] = (t_stat, p_val)

        ax.boxplot([early_vals, late_vals], labels=['Early','Late'])
        ax.set_title(f"{name}\n t={t_stat:.2f}, p={p_val:.3f}")
        ax.set_ylabel(name)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Turn off any unused subplots
    for ax in axes_flat[n_metrics:]:
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(directory, "compare_early_vs_late_metrics.png"), dpi=300)
    plt.close(fig)


    # ─── Band‐power figure (one row) ─────────────────────────────────────────
    n_bands = len(band_data)
    fig, axes = plt.subplots(1, n_bands, figsize=(5*n_bands, 4), squeeze=False)
    axes_flat = axes.ravel()

    for ax, (name, arr) in zip(axes_flat, band_data.items()):
        early_vals = arr[:early_end]
        late_vals  = arr[late_start:]
        t_stat, p_val = ttest_ind(early_vals, late_vals, equal_var=False)
        results['bands'][name] = (t_stat, p_val)

        ax.boxplot([early_vals, late_vals], labels=['Early','Late'])
        ax.set_title(f"{name}\n t={t_stat:.2f}, p={p_val:.3f}")
        ax.set_ylabel("Power")
        ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(directory, "compare_early_vs_late_bands.png"), dpi=300)
    plt.close(fig)

    return results

def compare_phases_all(
    metrics_data,
    band_data,
    early_end,
    late_start,
    directory
):
    """
    Do one‑way ANOVA + Tukey HSD across Early / Intermediate / Late phases
    for TWO sets of variables:
      1) metrics_data: dict of name→1D array (e.g. Accuracy, Inter-Dist, etc.)
      2) band_data:    dict of name→1D array (e.g. 'Mu Power', 'Beta Power')
    
    Saves TWO figures:
      - compare_phases_metrics.png  (2×2 grid for metrics)
      - compare_phases_bands.png    (1×2 grid for band powers)
    
    Returns:
      {
        'metrics': { name: { 'F':…, 'p':…, 'tukey': TukeyObj|None } },
        'bands':   { name: { 'F':…, 'p':…, 'tukey': TukeyObj|None } }
      }
    """
    # 1) Define the three phase slices
    early = slice(0, early_end)
    mid   = slice(early_end, late_start)
    late  = slice(late_start, None)
    phase_names = ['Early','Intermediate','Late']
    slices = [early, mid, late]

    os.makedirs(directory, exist_ok=True)
    results = {'metrics': {}, 'bands': {}}

    # ─── Metrics Figure ──────────────────────────────────────────────────────────
    n_metrics = len(metrics_data)
    rows = cols = int(np.ceil(np.sqrt(n_metrics)))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
    for ax, (name, arr) in zip(axes.flatten(), metrics_data.items()):
        # Gather data for each phase
        grp = [arr[s] for s in slices]
        # ANOVA
        F, p = f_oneway(*grp)
        # Tukey if needed
        tukey = None
        if p < 0.05:
            all_vals = np.concatenate(grp)
            labels  = (['Early']*len(grp[0]) +
                       ['Intermediate']*len(grp[1]) +
                       ['Late']*len(grp[2]))
            tukey = pairwise_tukeyhsd(all_vals, labels, alpha=0.05)
        # Plot
        ax.boxplot(grp, labels=phase_names)
        ax.set_title(f"{name}\nF={F:.2f}, p={p:.3f}")
        ax.set_ylabel(name)
        # Save results
        results['metrics'][name] = {'F':F, 'p':p, 'tukey':tukey}
    # Turn off any unused axes
    for ax in axes.flatten()[n_metrics:]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(directory, "compare_phases_metrics.png"), dpi=300)
    plt.close(fig)

    # ─── Band‑Power Figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(band_data), figsize=(5*len(band_data), 4), sharey=True)
    if len(band_data) == 1:
        axes = [axes]
    for ax, (name, arr) in zip(axes, band_data.items()):
        grp = [arr[s] for s in slices]
        F, p = f_oneway(*grp)
        tukey = None
        if p < 0.05:
            all_vals = np.concatenate(grp)
            labels   = (['Early']*len(grp[0]) +
                        ['Intermediate']*len(grp[1]) +
                        ['Late']*len(grp[2]))
            tukey = pairwise_tukeyhsd(all_vals, labels, alpha=0.05)
        ax.boxplot(grp, labels=phase_names)
        ax.set_title(f"{name}\nF={F:.2f}, p={p:.3f}")
        ax.set_ylabel("Power")
        results['bands'][name] = {'F':F, 'p':p, 'tukey':tukey}
    fig.tight_layout()
    fig.savefig(os.path.join(directory, "compare_phases_bands.png"), dpi=300)
    plt.close(fig)

    return results

def anova_cluster_distances(inter_distances, intra_distances_idle, intra_distances_motor, num_groups=10, directory=None):
    # Split the days into groups
    num_days = len(inter_distances)
    groups = np.array_split(np.arange(num_days), num_groups)
    
    # Prepare data for ANOVA
    inter_grouped = [inter_distances[idxs] for idxs in groups]
    intra_idle_grouped = [intra_distances_idle[idxs] for idxs in groups]
    intra_motor_grouped = [intra_distances_motor[idxs] for idxs in groups]

    # Perform ANOVA on inter-cluster and intra-cluster distances
    f_inter, p_inter = f_oneway(*inter_grouped)
    f_intra_idle, p_intra_idle = f_oneway(*intra_idle_grouped)
    f_intra_motor, p_intra_motor = f_oneway(*intra_motor_grouped)

    # Create subplots for Inter and Intra Distances
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Inter-Cluster Distance Plot
    ax1.plot(np.arange(30, num_days+30), inter_distances, label="Inter-Cluster Distance", marker='o', color='tab:blue')
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Inter-Cluster Distance [Cohen's d]")
    ax1.set_title(f"Inter-Cluster Distance over Days (ANOVA: F={f_inter:.2f}, p={p_inter:.4f})")
    ax1.axhline(y=np.mean(inter_distances), color='r', linestyle='--', label="Mean Distance")
    ax1.legend()

    # Intra-Cluster Distance (Idle) Plot
    ax2.plot(np.arange(30, num_days+30), intra_distances_idle, label="Intra-Cluster Distance (Idle)", marker='o', color='tab:green')
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Intra-Cluster Distance Idle [std]")
    ax2.set_title(f"Intra-Cluster Distance (Idle) over Days (ANOVA: F={f_intra_idle:.2f}, p={p_intra_idle:.4f})")
    ax2.axhline(y=np.mean(intra_distances_idle), color='r', linestyle='--', label="Mean Distance")
    ax2.legend()

    # Intra-Cluster Distance (Motor Imagery) Plot
    ax3.plot(np.arange(30, num_days+30), intra_distances_motor, label="Intra-Cluster Distance (Motor Imagery)", marker='o', color='tab:purple')
    ax3.set_xlabel("Days")
    ax3.set_ylabel("Intra-Cluster Distance Motor [std]")
    ax3.set_title(f"Intra-Cluster Distance (Motor Imagery) over Days (ANOVA: F={f_intra_motor:.2f}, p={p_intra_motor:.4f})")
    ax3.axhline(y=np.mean(intra_distances_motor), color='r', linestyle='--', label="Mean Distance")
    ax3.legend()

    # Adjust layout and display
    plt.tight_layout()

    if directory:
        os.makedirs(directory, exist_ok=True)
        save_path = os.path.join(directory, "anova_cluster_distances.png")
        fig.savefig(save_path, dpi=300)

    plt.show()
    plt.close(fig)

    # Return ANOVA results
    return {
        "Inter-Cluster Distance": {"F": f_inter, "p": p_inter},
        "Intra-Cluster Distance (Idle)": {"F": f_intra_idle, "p": p_intra_idle},
        "Intra-Cluster Distance (Motor)": {"F": f_intra_motor, "p": p_intra_motor}
    }



def load_config(config_path):
    """
    Loads a YAML configuration file.
    Args:
        config_path (str): Path to the config.yaml file.
    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_spaces(subject_id, X_csp_features_scaled, X_csp_features_scaled_2d,
                 X_pca_features_2D, X_umap_features_2D, X_pca_features_3D, X_umap_features_3D):
    """Return spaces and dim_dict based on subject."""
    spaces = {
        'CSP-2D': X_csp_features_scaled_2d,
        'PCA-2D': X_pca_features_2D,
        'UMAP-2D': X_umap_features_2D,
        'PCA-3D': X_pca_features_3D,
        'UMAP-3D': X_umap_features_3D,
        'CSP-full': X_csp_features_scaled
    }
    if subject_id in ['201', '205']:
        dim_dict = {'CSP-full': 6, 'UMAP-3D': 3, 'PCA-3D': 3, 'UMAP-2D': 2, 'PCA-2D': 2, 'CSP-2D': 2}
    elif subject_id == '206':
        dim_dict = {'CSP-full': 10, 'UMAP-3D': 3, 'PCA-3D': 3, 'UMAP-2D': 2, 'PCA-2D': 2, 'CSP-2D': 2}
    else:
        dim_dict = {'CSP-full': X_csp_features_scaled.shape[1], 'UMAP-3D': 3, 'PCA-3D': 3, 'UMAP-2D': 2, 'PCA-2D': 2, 'CSP-2D': 2}
    return spaces, dim_dict



# def cluster_based_permutation_test(X_csp, y_label, days_labels, clf_loaded, directory, n_permutations=1000, random_state=42, cluster_threshold=2):
#     np.random.seed(random_state)
#     significance_level = 0.05

#     lda_loaded = clf_loaded.named_steps['LDA']
#     unique_days = np.unique(days_labels)
#     unique_days_shifted = unique_days + 30

#     def find_clusters(t_stats, threshold):
#         above_threshold = np.abs(t_stats) > threshold
#         clusters = []
#         current_cluster = []
#         for i, above in enumerate(above_threshold):
#             if above:
#                 current_cluster.append(i)
#             elif current_cluster:
#                 clusters.append(current_cluster)
#                 current_cluster = []
#         if current_cluster:
#             clusters.append(current_cluster)
#         return clusters

#     original_t_stats = []
#     for day in unique_days:
#         day_mask = (days_labels == day)
#         X_day = X_csp[day_mask]
#         y_day = y_label[day_mask]
#         t_stat = calculate_t_statistic(X_day, y_day, lda_loaded)
#         original_t_stats.append(t_stat)

#     original_t_stats = np.array(original_t_stats)
#     original_clusters = find_clusters(original_t_stats, cluster_threshold)
#     original_cluster_stats = [np.sum(np.abs(original_t_stats[cluster])) for cluster in original_clusters]

#     permuted_cluster_stats = []
#     for _ in range(n_permutations):
#         permuted_t_stats = []
#         for day in unique_days:
#             day_mask = (days_labels == day)
#             X_day = X_csp[day_mask]
#             y_day = y_label[day_mask]
#             permuted_labels = np.random.permutation(y_day)
#             permuted_t_stat = calculate_t_statistic(X_day, permuted_labels, lda_loaded)
#             permuted_t_stats.append(permuted_t_stat)
        
#         permuted_t_stats = np.array(permuted_t_stats)
#         permuted_clusters = find_clusters(permuted_t_stats, cluster_threshold)
#         if permuted_clusters:
#             max_cluster_stat = np.max([np.sum(np.abs(permuted_t_stats[cluster])) for cluster in permuted_clusters])
#             permuted_cluster_stats.append(max_cluster_stat)
#         else:
#             permuted_cluster_stats.append(0)

#     permuted_cluster_stats = np.array(permuted_cluster_stats)
    
#     significant_clusters = []
#     for i, cluster_stat in enumerate(original_cluster_stats):
#         p_value = np.mean(permuted_cluster_stats >= cluster_stat)
#         if p_value < significance_level:
#             significant_clusters.append(original_clusters[i])

#     plt.figure(figsize=(12, 8))
#     plt.plot(unique_days_shifted, original_t_stats, label='Original t-statistic', color='blue', marker='o')
#     for cluster in significant_clusters:
#         plt.fill_between(unique_days_shifted[cluster], original_t_stats[cluster], color='red', alpha=0.3)
#     plt.title("Original t-statistics Across Days with Significant Clusters Highlighted")
#     plt.xlabel("Days")
#     plt.ylabel("t-statistic")
#     plt.legend()

#     plt.tight_layout()
#     if directory:
#         filename = f'Cluster_Based_Permutation_Test_Results_Days_{unique_days[0]}_to_{unique_days[-1]}.jpg'
#         full_path = os.path.join(directory, filename)
#         plt.savefig(full_path)
#     plt.show()

#     return {
#         'original_t_stats': original_t_stats,
#         'permuted_cluster_stats': permuted_cluster_stats,
#         'significant_clusters': significant_clusters,
#         'cluster_threshold': cluster_threshold
#     }

# def create_epochs_from_data(eeg_data, trial_labels, sfreq, tmin=0, tmax=None, ch_names=None):

    
#     n_trials, n_channels, n_samples = eeg_data.shape

#     if ch_names is None:
#         ch_names = ['FC3', 'C1', 'C3', 'C5', 'CP3', 'O1', 'FC4', 'C2', 'C4', 'C6', 'CP4']  # Your channel names
#     info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

#     # Create events array (shape: n_trials, 3)
#     event_id = {'Idle': 0, 'MotorImagery': 1}
#     events = np.column_stack((np.arange(n_trials), np.zeros(n_trials, dtype=int), trial_labels))
#     epochs = mne.EpochsArray(eeg_data, info, events, tmin=tmin, event_id=event_id, baseline=None)
    
#     return epochs

# def evaluate_csp_filters(X, y_label, csp_filters):

  
#     # Split data into classes
#     X_class_1 = X[y_label == 0]
#     X_class_2 = X[y_label == 1]

#     n_filters = csp_filters.shape[1]
#     pair_scores = {}

#     # Iterate over all combinations of 2 CSP filters
#     for filter_pair in combinations(range(n_filters), 2):
#         # Extract the two CSP filters for this pair
#         csp_filter_1 = csp_filters[:, filter_pair[0]]  # Shape: (n_channels,)
#         csp_filter_2 = csp_filters[:, filter_pair[1]]  # Shape: (n_channels,)

#         # Project the data for both classes using the current CSP filter pair
#         proj_class_1_filter_1 = np.dot(X_class_1, csp_filter_1)  # Shape: (n_trials, n_samples)
#         proj_class_1_filter_2 = np.dot(X_class_1, csp_filter_2)  # Shape: (n_trials, n_samples)

#         proj_class_2_filter_1 = np.dot(X_class_2, csp_filter_1)  # Shape: (n_trials, n_samples)
#         proj_class_2_filter_2 = np.dot(X_class_2, csp_filter_2)  # Shape: (n_trials, n_samples)

#         # Calculate the variance for each trial and each class (for both filters in the pair)
#         var_class_1_filter_1 = np.var(proj_class_1_filter_1, axis=1)  # Shape: (n_trials,)
#         var_class_1_filter_2 = np.var(proj_class_1_filter_2, axis=1)  # Shape: (n_trials,)

#         var_class_2_filter_1 = np.var(proj_class_2_filter_1, axis=1)  # Shape: (n_trials,)
#         var_class_2_filter_2 = np.var(proj_class_2_filter_2, axis=1)  # Shape: (n_trials,)

#         # Calculate the combined variance for each class (sum variances of both filters)
#         mean_var_class_1 = np.mean(var_class_1_filter_1 + var_class_1_filter_2)
#         mean_var_class_2 = np.mean(var_class_2_filter_1 + var_class_2_filter_2)

#         # Evaluate the separation score (Fisher's ratio or variance ratio)
#         score = mean_var_class_1 / (mean_var_class_2 + 1e-6)  # Add small value to avoid division by zero

#         # Store the score for the current pair of filters
#         pair_scores[filter_pair] = score

#     # Rank the CSP filter pairs by their score (higher is better)
#     best_filter_pair_indices = max(pair_scores, key=pair_scores.get)

#     return best_filter_pair_indices, pair_scores

# def calculate_ERD(eeg_epochs, freq_band=(8, 13), baseline_band=(1,40)):
 
#     # Perform time-frequency decomposition (Morlet wavelet) on all epochs
#     eeg_idle = eeg_epochs['Idle']
#     eeg_motor_imagery = eeg_epochs['MotorImagery']

#     tfr_motor_imagery = mne.time_frequency.tfr_morlet(eeg_motor_imagery, freqs=np.arange(freq_band[0], freq_band[1]), n_cycles=2, return_itc=False)
#     tfr_idle  = mne.time_frequency.tfr_morlet(eeg_idle, freqs=np.arange(baseline_band[0], baseline_band[1]), n_cycles=2, return_itc=False)

#     # Calculate mean power for task epochs (all epochs)
#     baseline_power = tfr_idle.data.mean(axis=(1,2)) 

#     task_power = tfr_motor_imagery .data.mean(axis=(1,2))  # Mean over all epochs


#     # Calculate ERD: (Baseline power - Task power) / Baseline power * 100
#     erd = ((baseline_power - task_power) / baseline_power) * 100

#     return erd
