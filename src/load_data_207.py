import scipy.io
import numpy as np
import os
from utils import eegFilters

class VR_207_data_extractor:
    def __init__(self, config):
        self.config = config
        # Allow hardcoding the path or using config path
        self.file_path = os.path.join(config.get('data_dir', ''), "207_1_VR01.mat")
        self.filter_lim = config.get('filter_lim', [1, 40])
        # Use all 16 channels if not specified, otherwise slice
        self.elec_idxs = config.get('elec_idxs', range(16)) 
        self.trial_len = config.get('trial_len', 4) 
        self.subject_id = config.get('id', '207')

    def get_EEG_dict(self):
        """
        Returns: { '207': [ { 'segmentedEEG': array, 'labels': array, ... } ] }
        """
        print(f"Loading {self.file_path}...")
        try:
            mat = scipy.io.loadmat(self.file_path)
        except FileNotFoundError:
            # Fallback if file is in local dir
            mat = scipy.io.loadmat("207_1_VR01.mat")

        # 1. Extract Raw Data
        # 'y' is usually (Channels x Time), e.g., (16 x Samples)
        raw_eeg = mat['y'] 
        
        # 'trig' is usually (1 x Samples) or (Samples x 1)
        triggers = mat['trig'].flatten()
        
        # Check FS
        if 'fs' in mat:
            fs = int(mat['fs'][0][0])
        else:
            fs = 250 # Fallback
            print(f"Warning: 'fs' not found in .mat, defaulting to {fs}Hz")

        # 2. Filter (Channels, Time)
        # utils.eegFilters uses mne.filter.filter_data which expects (n_channels, n_times)
        filtered_eeg = eegFilters(raw_eeg, fs, self.filter_lim)
        
        # Select Electrodes (if you want all 16, ensure config has range(16))
        filtered_eeg = filtered_eeg[self.elec_idxs, :]

        # 3. Epoching Logic
                # We need to find the ONSET of triggers.
        # Logic: Where trigger signal goes from 0 to X
        
        # Get indices where trigger changes
        diff_trig = np.diff(triggers, prepend=0)
        
        # Assuming trigger codes: 
        # You need to verify what 1 and 2 mean. 
        # Standard: 1=Left, 2=Right (or similar).
        # We will extract any trigger > 0.
        
        trial_starts = np.where(diff_trig > 0)[0]
        
        epochs_list = []
        labels_list = []
        
        samples_len = int(self.trial_len * fs)

        print(f"Found {len(trial_starts)} potential trials.")
        
        for start_idx in trial_starts:
            trig_code = triggers[start_idx]
            
            # Filter for specific classes if needed (e.g. only 1 and 2)
            # Assuming 1 and 2 are the classes we care about
            if trig_code in [1, 2]:
                end_idx = start_idx + samples_len
                
                if end_idx <= filtered_eeg.shape[1]:
                    # Extract epoch: (Channels, Time)
                    epoch = filtered_eeg[:, start_idx:end_idx]
                    epochs_list.append(epoch)
                    
                    # Map to 0 and 1 for Binary Classification
                    # e.g., Trigger 1 -> Class 0, Trigger 2 -> Class 1
                    label = 0 if trig_code == 1 else 1
                    labels_list.append(label)

        X = np.array(epochs_list) # Result: (Trials, Channels, Time)
        y = np.array(labels_list)
        
        print(f"Extracted {X.shape[0]} trials. Shape: {X.shape}")

        # 4. Package as a "Day" dictionary
        # The main script expects a LIST of these dictionaries.
        session_dict = {
            'segmentedEEG': X,
            'labels': y,
            'fs': fs,
            'chanLabels': [f'Ch{i+1}' for i in self.elec_idxs], # Dummy labels
            'trials_N': len(y)
        }
        
        # Return nested format: { '207': [ session_dict ] }
        return {self.subject_id: [session_dict]}