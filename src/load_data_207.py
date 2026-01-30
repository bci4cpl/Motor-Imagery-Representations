import os
import numpy as np
import scipy.io
import mne
from utils import eegFilters

class Chist_Era_207_extractor:

    def __init__(self, config):
        self.sub = config['sub']
        self.data_dir = os.path.join(config['data_dir'], f"sub{self.sub}")
        self.trial_len = config['trial_len']
        self.filter_lim = config['filter_lim']
        self.elec_idxs = config['elec_idxs']
        self.days = config['num_days']
        self.block = config['block']
        self.fs = config['fs']
        self.EEG_dict = None

    def reformat_session_data(self, mat_data):
        # EEG data is saved in 'data' variable
        EEG = mat_data['data'][0][0][0].T 
        
        # dataOrgVR cell structure
        # Index 0: Timestamps (Seconds)
        # Index 2: Event Labels (Strings)
        dataOrgVR = mat_data['dataOrgVR'][0]
        chanLabels = [ch[0] for ch in mat_data['data']['chanLabels'][0][0][0]]
        fs = int(mat_data['data']['sampleFreq'][0][0][0][0])

        timestamps = dataOrgVR[0].flatten()
        scores = dataOrgVR[1].flatten()
        events = dataOrgVR[2].flatten()
        
        data = {'EEG': EEG,
                'events': events,
                'timestamps': timestamps,
                'scores': scores,
                'chanLabels': chanLabels,
                'fs': fs}
        
        return data
    def segment_EEG(self, eegArrangedDict, printFlag=1):

        EEG = []
        labels = []

        removedCount = 0
        idleCount = 0
        imagineCount = 0

        fs = self.fs
        expected_samples = int(self.trial_len * fs)

        eeg = eegArrangedDict['EEG']
        events = eegArrangedDict['events']

        i = 0
        n_events = len(events)

        while i < n_events:

            event = events[i]

            # ==========================
            # IDLE: relax → 1st_pause
            # ==========================
            if event == 'relax':

                j = i + 1
                while j < n_events and events[j] != '1st_pause':
                    j += 1

                if j >= n_events:
                    break

                start_idx = i * 125
                end_idx   = j * 125 + 500

                segment = eeg[:, start_idx:end_idx]

                if segment.shape[1] < expected_samples:
                    removedCount += 1
                else:
                    EEG.append(segment[:, :expected_samples])
                    labels.append(0)
                    idleCount += 1

                i = j + 1
                continue

            # ==========================
            # MI: move → robot or 2nd_pause
            # ==========================
            if event == 'move':

                j = i + 1
                while j < n_events:
                    if events[j] == '2nd_pause' or 'robot' in events[j]:
                        break
                    j += 1

                if j >= n_events:
                    break

                start_idx = i * 125
                end_idx   = j * 125 + 500

                segment = eeg[:, start_idx:end_idx]

                if segment.shape[1] < expected_samples:
                    removedCount += 1
                else:
                    if events[j]=='robot':
                        segment = mne.filter.notch_filter(
                                segment.astype(np.float64), self.fs, 
                                freqs=[35], verbose=False
                            )
                    EEG.append(segment[:, :expected_samples])
                    labels.append(1)
                    imagineCount += 1

                i = j + 1
                continue

            i += 1

        eegArrangedDict['segmentedEEG'] = np.asarray(EEG)
        eegArrangedDict['labels'] = np.asarray(labels)

        if printFlag:
            print(f'Imagine Trials: {imagineCount}')
            print(f'Idle Trials: {idleCount}')
            print(f'Removed Trials: {removedCount}')

        return eegArrangedDict



    def get_EEG_dict(self):
        all_sessions_EEG = []
        for day in self.days:
            for block_i in self.block:
                fileFormat = f"207_day{day}_VR0{block_i}.mat"
                path = os.path.join(self.data_dir, fileFormat)
                
                if not os.path.exists(path):
                    continue
                
                session_mat = scipy.io.loadmat(path)
                day_data = self.reformat_session_data(session_mat)
                
                # Apply filters from your utils
                day_data['EEG'] = eegFilters(day_data['EEG'], self.fs, self.filter_lim)
                
                # Segment 
                all_sessions_EEG.append(self.segment_EEG(day_data, printFlag=0))

        # Uses your original merge logic
        self.EEG_dict = {self.sub: self.merge_session_blocks(all_sessions_EEG)}
        return self.EEG_dict

    def merge_session_blocks(self, eegDictList):
        # Your existing stacking logic remains here
        stackedList = []
        i = 0
        while i < len(eegDictList):
            blocks_in_day = len(self.block)
            tempArray = eegDictList[i]['segmentedEEG']
            tempLabels = eegDictList[i]['labels']
            for j in range(1, blocks_in_day):
                if (i + j) < len(eegDictList):
                    tempArray = np.concatenate((tempArray, eegDictList[i + j]['segmentedEEG']))
                    tempLabels = np.concatenate((tempLabels, eegDictList[i + j]['labels']))

            eegDict = eegDictList[min(i + blocks_in_day - 1, len(eegDictList)-1)]
            stackedDict = {
                'segmentedEEG': tempArray,
                'labels': tempLabels,
                'chanLabels': eegDict['chanLabels'],
                'events': eegDict['events'],
                'timestamps': eegDict['timestamps'],
                'scores': eegDict['scores'],
                'fs': self.fs,
                'trials_N': len(tempLabels)
            }
            stackedList.append(stackedDict)
            i += blocks_in_day
        return stackedList