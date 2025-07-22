import os
import yaml
from sklearn.preprocessing import StandardScaler
from load_data_CHIST_ERA import Chist_Era_data_extractor
import utils
from denoiser import Denoiser
import joblib


def main():
    # Load YAML config
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    # Global settings
    seed         = cfg['seed']
    n_components = cfg['n_components']
    AE_mode      = cfg['AE_mode']   # controls AE train vs inference
    mode         = cfg['mode']      # controls CSP-LDA train vs test
    cache_dir    = subj_cfg['cache_dir'].format(**env)

    utils.set_seed(seed)

    # Process each subject
    for subj_cfg in cfg['subjects']:
        print(f"=== Processing subject {subj_cfg['id']} ===")

        # Resolve all placeholders
        env = {**cfg, **subj_cfg}
        subj_cfg['data']['data_dir']       = subj_cfg['data']['data_dir'].format(**env) #"{data_root}".format(data_root=env['data_root'])
        subj_cfg['ae_dir']                 = subj_cfg['ae_dir'].format(**env)
        subj_cfg['clf_dir']                = subj_cfg['clf_dir'].format(**env)
        subj_cfg['cache_dir']              = subj_cfg['cache_dir'].format(**env)
        subj_cfg['ae_paths']['best_epoch'] = subj_cfg['ae_paths']['best_epoch'].format(**env)
        for section in ['ae_paths','figures']:
            subj_cfg[section] = {k: v.format(**env) for k,v in subj_cfg[section].items()}

        for key, template in subj_cfg['clf_paths'].items():
            subj_cfg['clf_paths'][key] = template.format(**env)

        # Subject‑specific day splits
        start_train, end_train = subj_cfg['train_test_days']['train']
        start_test,  end_test  = subj_cfg['train_test_days']['test']

        # AE and train hyperparams
        subj_props = {**cfg['ae_params'], **cfg['training_params']}

        # 1) Data extraction
        extractor = Chist_Era_data_extractor(subj_cfg['data'])
        eeg_dict  = extractor.get_EEG_dict()
        sub_data  = eeg_dict[subj_cfg['id']]
        train_dataset = utils.EEGDataSet_signal_by_day(sub_data, [start_train, end_train])
        test_dataset  = utils.EEGDataSet_signal_by_day(sub_data, [start_test,  end_test])

        # 2) AE training or loading
        if mode == 'train':
            ae_trainer = Denoiser(subj_props, AE_mode, train_dataset)
            ae_trainer.train_and_save(
                train_dataset,
                subj_cfg['ae_train_params']['n_epochs'],
                subj_cfg['ae_train_params']['save_every'],
                subj_cfg['ae_dir']
            )
            denoiser = Denoiser(subj_props, AE_mode, test_dataset)
        else:
            denoiser = Denoiser(subj_props, AE_mode, test_dataset)
        denoiser.load_weights(subj_cfg['ae_paths']['best_epoch'])

        denoised_signal_train_centered, y_label_train, days_label_train = \
            utils.preprocess_dataset(train_dataset, denoiser)
        denoised_signal_test_centered,  y_label,       days_label        = \
            utils.preprocess_dataset(test_dataset,  denoiser)

        # 3) CSP-LDA train or load
        if mode == 'train':
            models, results, eigen_vals, eigen_vecs, auc_scores = \
                utils.grid_search_alternate_csp(
                    denoised_signal_train_centered,
                    y_label_train,
                    n_components,
                    subj_cfg['clf_dir']
                )
            
        for key, template in subj_cfg['clf_paths'].items():
            subj_cfg['clf_paths'][key] = template.format(**env)
        clf_loaded = joblib.load(subj_cfg['clf_paths']['best_model'])

        # 4) Load projection models into variables
        pca_model_2D  = joblib.load(subj_cfg['clf_paths']['pca2d'])
        umap_model_2D = joblib.load(subj_cfg['clf_paths']['umap2d'])
        pca_model_3D  = joblib.load(subj_cfg['clf_paths']['pca3d'])
        umap_model_3D = joblib.load(subj_cfg['clf_paths']['umap3d'])

        # 5) Extract CSP features
        X_csp_features_train = clf_loaded.named_steps['CSP'].transform(denoised_signal_train_centered)
        X_csp_features       = clf_loaded.named_steps['CSP'].transform(denoised_signal_test_centered)

        print("Mean of CSP features (train):", X_csp_features_train.mean(axis=0))
        print("Mean of CSP features (test):",  X_csp_features.mean(axis=0))

        # 6) Scale and compute metrics
        scaler = StandardScaler()
        X_csp_features_scaled_train = scaler.fit_transform(X_csp_features_train)
        X_csp_features_scaled       = scaler.transform(X_csp_features)

        accuracies, auc_scores_metrics, inter_var, intra_idle, intra_motor = \
            utils.compute_acc_inter_intra_var(
                X_csp_features_scaled,
                y_label,
                days_label,
                clf_loaded
            )

        # 7) Projections
        X_csp_features_scaled_1d = X_csp_features_scaled[:, :1]
        X_csp_features_scaled_2d = X_csp_features_scaled[:, :2]
        X_pca_features_2D        = utils.apply_pca(X_csp_features_scaled, pca_model_2D,  n_components=2)
        X_umap_features_2D       = utils.apply_umap(X_csp_features_scaled, umap_model_2D, n_components=2)
        X_pca_features_3D        = utils.apply_pca(X_csp_features_scaled, pca_model_3D,  n_components=3)
        X_umap_features_3D       = utils.apply_umap(X_csp_features_scaled, umap_model_3D, n_components=3)

        # 8) Cache all variables with original names
        os.makedirs(cache_dir, exist_ok=True)
        to_save = {
            "start_test_day": start_test,
            "end_test_day":   end_test,
            "clf_loaded":     clf_loaded,
            "X_csp_features": X_csp_features,
            "X_csp_features_scaled":       X_csp_features_scaled,
            "X_csp_features_train":        X_csp_features_train,
            "X_csp_features_scaled_train": X_csp_features_scaled_train,
            "denoised_signal_train_centered": denoised_signal_train_centered,
            "denoised_signal_test_centered":  denoised_signal_test_centered,
            "y_label":         y_label,
            "y_label_train":   y_label_train,
            "days_label":      days_label,
            "days_label_train":days_label_train,
            "accuracies":      accuracies,
            "inter_variances": inter_var,
            "intra_variances_idle": intra_idle,
            "intra_variances_motor": intra_motor,
            "X_csp_features_scaled_1d": X_csp_features_scaled_1d,
            "X_csp_features_scaled_2d": X_csp_features_scaled_2d,
            "X_pca_features_2D":        X_pca_features_2D,
            "X_umap_features_2D":       X_umap_features_2D,
            "X_pca_features_3D":        X_pca_features_3D,
            "X_umap_features_3D":       X_umap_features_3D
        }
        if mode == 'train':
            to_save.update({
                "models":       models,
                "results":      results,
                "eigen_values": eigen_vals,
                "eigen_vectors":eigen_vecs,
                "auc_scores":   auc_scores
            })
        for name, var in to_save.items():
            joblib.dump(var, os.path.join(cache_dir, f"{name}.pkl"))

        print(f"✅ Subject {subj_cfg['id']} complete — data cached in '{cache_dir}'")

if __name__ == '__main__':
    main()

# legacy
#Using DWT features
# dwt = DWT(denoised_signal)
# dwt_features,_ = dwt.get_dwt_features() # trial,electrode,band,feature
# num_trials, num_electrodes, num_bands, num_features = dwt_features.shape
# total_num_features = num_electrodes * num_bands * num_features
# dwt_features_by_trial = dwt_features.reshape(num_trials, total_num_features) # trial,features
# X_features = np.concatenate((x_csp_features, dwt_features_by_trial), axis=1)


