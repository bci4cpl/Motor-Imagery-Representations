
from sklearn.preprocessing import StandardScaler
from load_data_CHIST_ERA import Chist_Era_data_extractor
from properties import sub201_properties as sub201_props
import utils
import os

from denoiser import Denoiser
import joblib
import umap
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist



def procces(subject):
     pass

def main():
        
    # 1. SETUP
    # Set seed
    utils.set_seed(42)

    # Parameters
    start_train_day, end_train_day = 0, 30
    start_test_day, end_test_day = 30, 133  # last_test_day=133
    n_components= (2,4,6,8,10)

    # Paths
    AE_path_one_epoch = "D:/Niv/Motor imagery skill/AE/sub_201/unsupervised_model_parameters_one_epoch.pt"
    AE_path = "D:/Niv/Motor imagery skill/AE/sub_201/ae_epoch_200.pt"
    save_AE_models = "D:/Niv/Motor imagery skill/AE/sub_201"
    save_models_path ="D:/Niv/Motor imagery skill/clf/sub_201"
    save_directory_fig = r"D:/Niv/Motor imagery skill/Figures/sub_201"

    load_clf_weights_path = r'D:/Niv/Motor imagery skill/clf/sub_201/best_model_6.pkl'
    load_pca2d_weights_path = r'D:/Niv/Motor imagery skill/clf/sub_201/pca2D_reducer.pkl'
    load_umap2d_weights_path = r'D:/Niv/Motor imagery skill/clf/sub_201/umap2D_reducer.pkl'
    load_pca3d_weights_path = r'D:/Niv/Motor imagery skill/clf/sub_201/pca3D_reducer.pkl'
    load_umap3d_weights_path = r'D:/Niv/Motor imagery skill/clf/sub_201/umap3D_reducer.pkl'

    # 2. LOAD & PREPROCESS
    # Instantiate the data extractor object
    data_extractor_201 = Chist_Era_data_extractor()
    # Call the method to extract data and apply bandpass filter
    sub201_data = data_extractor_201.get_EEG_dict()
    train_dataset = utils.EEGDataSet_signal_by_day(sub201_data['201'], [start_train_day, end_train_day])
    test_dataset = utils.EEGDataSet_signal_by_day(sub201_data['201'], [start_test_day, end_test_day]) 

    # Apply denoiser
    mode = 'unsupervised'
    denoiser = Denoiser(sub201_props, mode, test_dataset)

    # Train denoiser and save weights 
    # denoiser.train_and_save(train_dataset, 200,20,save_AE_models)

    # Load weights
    denoiser.load_weights(AE_path_one_epoch)
    denoised_signal_train_centered,y_label_train,days_label_train = utils.preprocess_dataset(train_dataset,denoiser)
    denoised_signal_test_centered,y_label,days_label = utils.preprocess_dataset(test_dataset,denoiser)

    # Train and save models    
    # models, results, eigen_vals,eigen_vecs, _ = utils.grid_search_alternate_csp(denoised_signal_train_centered, y_label_train, n_components,save_models_path)

    # 3. LOAD MODELS & EXTRACT FEATURES
    clf_loaded = joblib.load(load_clf_weights_path)
    pca_model_2D = joblib.load(load_pca2d_weights_path)
    umap_model_2D = joblib.load(load_umap2d_weights_path)
    pca_model_3D = joblib.load(load_pca3d_weights_path)
    umap_model_3D = joblib.load(load_umap3d_weights_path)

    X_csp_features_train = clf_loaded.named_steps['CSP'].transform(denoised_signal_train_centered) # it was trianed before in the grid_serach
    X_csp_features = clf_loaded.named_steps['CSP'].transform(denoised_signal_test_centered) # trials, csp_componenets

    mean_train = X_csp_features_train.mean(axis=0)
    std_train  = X_csp_features_train.std(axis=0)
    print("Mean of CSP features (train):", mean_train)

    mean_test = X_csp_features.mean(axis=0)
    print("Mean of CSP features (test):", mean_test)


    scaler = StandardScaler()
    X_csp_features_scaled_train = scaler.fit_transform(X_csp_features_train)
    X_csp_features_scaled = scaler.transform(X_csp_features)

 

    #4. COMPUTE METRICS & PROJECTIONS
    accuracies,auc_scores, inter_variances, intra_variances_idle, intra_variances_motor = utils.compute_acc_inter_intra_var(X_csp_features_scaled, y_label, days_label, clf_loaded)
    filters = clf_loaded.named_steps['CSP'].filters_
    X_csp_features_scaled_1d = X_csp_features_scaled[:,:1]
    X_csp_features_scaled_2d = X_csp_features_scaled[:,:2]


    X_pca_features_2D = utils.apply_pca(X_csp_features_scaled, pca_model_2D,n_components=2)
    X_umap_features_2D = utils.apply_umap(X_csp_features_scaled,umap_model_2D,n_components=2)
    X_pca_features_3D = utils.apply_pca(X_csp_features_scaled, pca_model_3D,n_components=3)
    X_umap_features_3D = utils.apply_umap(X_csp_features_scaled,umap_model_3D,n_components=3)

    # 5. CACHE EVERYTHING
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    # to_save = {
    #         "start_test_day": start_test_day,
    #         "end_test_day": end_test_day,
    #         "clf_loaded": clf_loaded,
    #         "X_csp_features": X_csp_features,
    #         "X_csp_features_scaled": X_csp_features_scaled,
    #         "X_csp_features_train": X_csp_features_train,
    #         "X_csp_features_scaled_train": X_csp_features_scaled_train,
    #         "denoised_signal_train_centered": denoised_signal_train_centered,
    #         "denoised_signal_test_centered":denoised_signal_test_centered,
    #         "y_label": y_label,
    #         "y_label_train": y_label_train,
    #         "days_label": days_label,
    #         "days_label_train": days_label_train,
    #         "accuracies": accuracies,
    #         "inter_variances": inter_variances,
    #         "intra_variances_idle": intra_variances_idle,
    #         "intra_variances_motor": intra_variances_motor,
    #         "X_csp_features_scaled_1d" :X_csp_features_scaled_1d,
    #         "X_csp_features_scaled_2d": X_csp_features_scaled_2d,
    #         "X_pca_features_2D": X_pca_features_2D,
    #         "X_umap_features_2D": X_umap_features_2D,
    #         "X_pca_features_3D": X_pca_features_3D,
    #         "X_umap_features_3D": X_umap_features_3D,
    #         "eigen_values": eigen_vals,
    #         "eigen_vectors": eigen_vecs,
    #         "cv_results": results,
    #         "auc_scores": auc_scores
    #     }
    
    to_save = {"X_csp_features_scaled_1d": X_csp_features_scaled_1d, 
               "X_csp_features_scaled_2d":X_csp_features_scaled_2d
                }

    for name, var in to_save.items():
            joblib.dump(var, os.path.join(cache_dir, f"{name}.pkl"))

    print("✅ Part A complete — all variables cached in 'cache/'.")

     

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


