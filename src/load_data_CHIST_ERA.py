import os
import numpy as np
import mne
import scipy.io
from utils import eegFilters


class Chist_Era_data_extractor:

    def __init__(self, config):
        self.sub = config['sub']
        self.eyes_state = config['eyes_state']
        self.data_dir = config['data_dir']
        self.block = config['block']
        self.trial_len = config['trial_len']
        self.full_path = os.path.join(self.data_dir, f"sub{self.sub}", f"RA{self.eyes_state}")
        self.filter_lim = config['filter_lim']
        self.elec_idxs = config['elec_idxs']
        self.days = self.get_all_days()
        self.EEG_dict = None
        self.paramRA = []
        self.dataOrgVR = []

    def get_all_days(self):
        onlyfiles = [f for f in os.listdir(self.full_path) if f.endswith('.mat')]
        days = []
        for item in onlyfiles:
            days.append(int(item.split('-')[1][3:]))

        return np.unique(days)

    # def createMontage(self, chanLabels):
    #     """
    #     Creates standard 10-20 location montage for given channel set
    #     """
    #     montageGeneral = mne.channels.make_standard_montage('standard_1020')
    #     locationDict = montageGeneral.get_positions()
    #     locationDict = locationDict['ch_pos']
    #     montageDict = {}

    #     for elec_i in chanLabels:
    #         montageDict[elec_i] = locationDict[elec_i]

    #     montage = mne.channels.make_dig_montage(montageDict)
    #     return montage

    # def merge_session_blocks(self, eegDictList):
    #     """
    #     Stack blocks from same day into one EEG + labels dictionary
    #     sub206 didnt have day01-block01, and just had 2 blocks for day 1 (block2 and block3).
    #     """
    #     stackedList = []
    #     count = 0
    #     for i, eegDict in enumerate(eegDictList):
    #         if i % len(self.block) == 0:
    #             tempArray = eegDict['segmentedEEG']
    #             tempLabels = eegDict['labels']
    #         else:
    #             tempArray = np.concatenate((tempArray, eegDict['segmentedEEG']))
    #             tempLabels = np.concatenate((tempLabels, eegDict['labels']))
    #             count += 1
    #         if count == len(self.block) - 1:
    #             stackedDict = {'segmentedEEG': tempArray, 'labels': tempLabels, 'fs': eegDict['fs'],
    #                            'chanLabels': eegDict['chanLabels'], 'trigLabels': eegDict['trigLabels'],
    #                            'trials_N': len(tempLabels)}
    #             stackedList.append(stackedDict)
    #             count = 0

    #     return stackedList

    def merge_session_blocks(self, eegDictList):
        """
        Stack blocks from same day into one EEG + labels dictionary.
        Special case for sub206: day 1 has only 2 blocks (block2 and block3).
        """
        stackedList = []
        i = 0

        while i < len(eegDictList):
            # Handle sub206 day 1 (first 2 blocks only)
            if self.sub == '206' and i == 0:
                blocks_in_day = 2
            else:
                blocks_in_day = len(self.block)

            # Stack blocks_in_day blocks
            tempArray = eegDictList[i]['segmentedEEG']
            tempLabels = eegDictList[i]['labels']
            for j in range(1, blocks_in_day):
                tempArray = np.concatenate((tempArray, eegDictList[i + j]['segmentedEEG']))
                tempLabels = np.concatenate((tempLabels, eegDictList[i + j]['labels']))

            eegDict = eegDictList[i + blocks_in_day - 1]  # Use last block's metadata
            stackedDict = {
                'segmentedEEG': tempArray,
                'labels': tempLabels,
                'fs': eegDict['fs'],
                'chanLabels': eegDict['chanLabels'],
                'trigLabels': eegDict['trigLabels'],
                'trials_N': len(tempLabels)
            }
            stackedList.append(stackedDict)

            i += blocks_in_day  # move to the next set of blocks

        return stackedList





    def segment_EEG(self, eegArrangedDict, printFlag=1):
        """
        Segment the data into epochs of MI and idle.
        """
        EEG = []
        labels = []
        removedCount = 0
        idleCount = 0
        imagineCount = 0

        # Timestamps of "move" command
        imgIdx = np.where(eegArrangedDict['triggers'] == 3)[0]
        # Timestamps of 1st pause
        idleIdx = np.where(eegArrangedDict['triggers'] == 2)[0]
        for idx in imgIdx:
            # Check if there's artifacts in trial (more than half the trial is labeled with artificats)
            if np.sum(eegArrangedDict['artifacts'][idx + 1: idx + 1 + int(self.trial_len * eegArrangedDict['fs'])]) > \
                    self.trial_len * eegArrangedDict['fs'] * 0.9:
                removedCount += 1
                # Check that the trial is atleast as the given trial length (not ended before)
            elif np.sum(
                    eegArrangedDict['triggers'][idx + 1: idx + 1 + int(self.trial_len * eegArrangedDict['fs'])]) == 0:
                EEG.append(eegArrangedDict['EEG'][:, idx: idx + int(self.trial_len * eegArrangedDict['fs'])])
                labels.append(1)
                imagineCount += 1
            else:
                removedCount += 1

        for idx in idleIdx:
            if np.sum(eegArrangedDict['artifacts'][idx + 1: idx + 1 + int(self.trial_len * eegArrangedDict['fs'])]) > 0:
                removedCount += 1
            else:
                EEG.append(eegArrangedDict['EEG'][:, idx: idx + int(self.trial_len * eegArrangedDict['fs'])])
                labels.append(0)
                idleCount += 1

        # Add to the dictionary the segmented data
        eegArrangedDict['segmentedEEG'] = np.asarray(EEG)
        eegArrangedDict['labels'] = np.asarray(labels)

        if printFlag:
            # Print number of trials of each class and number of removed trials
            print(f'Imagine Trials-{imagineCount} \nIdle Trials- {idleCount} \nRemoved Trials- {removedCount}\n')

        # Return the dictionary
        return eegArrangedDict

    def reformat_session_data(self, eegDict):
        """
        Arrange the given dictionary to more comfort dictionary
        """
        # EEG will be channels_N X timestamps_N
        EEG = eegDict['dat']['X'][0][0].T
        # Triggers
        triggers = np.squeeze(eegDict['dat']['Y'][0][0])
        # Artifacts marker
        artifacts = np.squeeze(eegDict['dat']['E'][0][0])
        # Sampling rate
        fs = eegDict['header']['sampleFreq'][0][0][0][0]
        # Electrodes labels
        chanLabels = [ch[0] for ch in eegDict['header']['Xlabels'][0][0][0]]
        # Triggers labels
        trigLabels = [trig[0] for trig in eegDict['header']['Ymarkers'][0][0][0]]
        # Trials time (in secs)
        imagineLength = eegDict['paramRA']['c_robot'][0][0][0][0]
        idleLength = eegDict['paramRA']['b_pause'][0][0][0][0]

        Data = {'EEG': EEG, 'triggers': triggers, 'artifacts': artifacts, 'fs': fs,
                'chanLabels': chanLabels, 'trigLabels': trigLabels, 'imagineLength': imagineLength,
                'idleLength': idleLength}
        return Data

    def extract_data(self):
        """
        Iterate over days given, of specific subject and get a list of all the files of the relevant days
        """

        data = []
        for day in self.days:
            dayStr = str(day)
            if len(dayStr) == 1:
                dayStr = '0' + dayStr
            for block_i in self.block:
                fileFormat = 'sub' + self.sub + '-day' + dayStr + '-block' + str(
                    block_i) + '-condRA' + self.eyes_state + '.mat'
                try:
                     data.append(scipy.io.loadmat(self.full_path + '/' + fileFormat))
                except FileNotFoundError:
                        print(f"⚠️ File not found: {fileFormat} — Skipping.")


        return data

    def get_EEG_dict(self):
        if not self.EEG_dict:
            all_sessions_data = self.extract_data()
            

            # Extract and segment all the data
            all_sessions_EEG = []
            for session_data in all_sessions_data:
                # Extract each day data
                day_data = self.reformat_session_data(session_data)

                # This condition is to remove some corrupted files in subject 201
                if day_data['EEG'].dtype != np.dtype('float64'):
                    continue

                # Filter the data
                day_data['EEG'] = eegFilters(day_data['EEG'], day_data['fs'], self.filter_lim)
                day_data['EEG'] = day_data['EEG'][self.elec_idxs, :]

                # Segment the data
                all_sessions_EEG.append(self.segment_EEG(day_data, printFlag=0))

                # for 205, MI 154/240 = 64% and Idle 212/240  = 88%
            # Stack block of same day
            self.EEG_dict = {self.sub: self.merge_session_blocks(all_sessions_EEG)}
        return self.EEG_dict
    