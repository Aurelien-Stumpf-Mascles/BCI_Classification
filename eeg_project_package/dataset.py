import torch
import os
import mne
import numpy as np
import scipy as sc
import sys 
sys.path.append("/home/aurelien.stumpf/Development/BCI-Classification")
from eeg_project_package import spectral_analysis

def load_file_eeg(filepath):
    raw_Training_EEG =  mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    events_from_annot,event_id = mne.events_from_annotations(raw_Training_EEG,event_id='auto')
    return raw_Training_EEG, events_from_annot,event_id

def select_Event(event_name,RAW_data,events_from_annot,event_id,t_min,t_max,number_electrodes):
    epochs_training = mne.Epochs(RAW_data, events_from_annot, event_id,tmin=t_min, tmax=t_max,preload=True,event_repeated='merge',baseline = None,picks = np.arange(0,number_electrodes))
    return epochs_training[event_name]
    
# EEG Dataset for multiple subject
def time_dataset_creator(files, list_idx_channels, list_labels, freq=500):

    """
    Take the data from every file and collect only those corresponding to GDF-left or GDF-right in the interest_data list
    Those events last for a number of data corresponding to segment_length

    parameters: files: string list corresponding to the data folders paths
                events: int list corresponding to the numbers representatives of the events of interest
                segment_length: int corresponding to the length of the data segment we will keep
    """
    features = []
    labels = []
    dict_sorted_labels = {list_labels[i]:i for i in range(len(list_labels))}

    for file in files:
        try : 
            for event_name in list_labels:
                raw_training, events_from_annot,event_id = load_file_eeg(filepath = file)
                tmin = 0
                tmax = 4
                event_var = select_Event(event_name,raw_training,events_from_annot,event_id,tmin,tmax,64)
                data = event_var.get_data()
                print(data.shape)
                data = data[:,list_idx_channels,:]
                features.append(data)
                labels.append(dict_sorted_labels[event_name]*np.ones(data.shape[0]))
        except Exception as e:
            print("Error in file: ", file)
            print(e)
        
    features = np.concatenate(features, axis=0)
    features = torch.from_numpy(features).unsqueeze(1).float()
    labels = np.concatenate(labels,axis=0)
    labels = torch.Tensor(labels).long()

    return features, labels

# Create dataset for PSD
def psd_dataset_creator(files,list_idx_channels,list_labels,type_psd="welch"):

    """
    Take the data from every file and collect only those corresponding to GDF-left or GDF-right in the interest_data list
    Those events last for a number of data corresponding to segment_length

    parameters: files: string list corresponding to the data folders paths
                events: int list corresponding to the numbers representatives of the events of interest
                segment_length: int corresponding to the length of the data segment we will keep
    """
    features = []
    labels = []
    dict_sorted_labels = {list_labels[i]:i for i in range(len(list_labels))}
    freqs = None

    for file in files:
        try : 
            for event_name in list_labels:
                raw_training, events_from_annot,event_id = load_file_eeg(filepath = file)
                tmin = 0
                tmax = 4
                event_var = select_Event(event_name,raw_training,events_from_annot,event_id,tmin,tmax,64)
                data = event_var.get_data()
                print(data.shape)
                fs = event_var.info['sfreq']
                data = data[:,list_idx_channels,:]
                if type_psd == "welch":
                    f, Pxx = sc.signal.welch(data, fs = fs, nperseg=300, noverlap=150, detrend="constant")
                    idx_freq = np.argwhere((f >= 4) & (f <= 100)).flatten()
                    if freqs is None:
                        freqs = f[idx_freq]
                    Pxx = Pxx[:,:,idx_freq]
                if type_psd == "burg":
                    noverlap = 150
                    nperseg = 500
                    fs = 500
                    f_max = 100
                    N_fft = 200
                    filter_order = 19
                    Pxx, Time_freq, time = spectral_analysis.Power_burg_calculation(data,noverlap,N_fft,f_max, nperseg,filter_order)
                    
                features.append(Pxx)
                labels.append(dict_sorted_labels[event_name]*np.ones(data.shape[0]))
        except Exception as e:
            print("Error in file: ", file)
            print(e)

    if len(features) == 1:
        features = np.array(features)
    else:
        features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels,axis=0)
    
    return features,labels,freqs

def band_psd_dataset_creator(files,list_idx_channels,list_labels,type_psd="welch"):

    """
    Take the data from every file and collect only those corresponding to GDF-left or GDF-right in the interest_data list
    Those events last for a number of data corresponding to segment_length

    parameters: files: string list corresponding to the data folders paths
                events: int list corresponding to the numbers representatives of the events of interest
                segment_length: int corresponding to the length of the data segment we will keep
    """
    features = []
    labels = []
    band_freqs = {"delta":[1, 4],"theta": [4, 8],"alpha": [8, 14],"beta": [14, 31],"gamma": [31, 49]}
    dict_sorted_labels = {list_labels[i]:i for i in range(len(list_labels))}
    freqs = None

    for file in files:
        print(file)
        try : 
            for event_name in list_labels:
                raw_training, events_from_annot,event_id = load_file_eeg(filepath = file)
                tmin = 0
                tmax = 4
                event_var = select_Event(event_name,raw_training,events_from_annot,event_id,tmin,tmax,64)
                data = event_var.get_data()
                fs = event_var.info['sfreq']
                data = data[:,list_idx_channels,:]
                if type_psd == "welch":
                    f, Pxx = sc.signal.welch(data, fs = fs, nperseg=300, noverlap=150, detrend="constant")
                    band_features = np.zeros((data.shape[0],data.shape[1],5))
                    idx = 0
                    for key in band_freqs.keys():
                        idx_freq = np.argwhere((f >= band_freqs[key][0]) & (f <= band_freqs[key][1])).flatten()
                        band_features[:,:,idx] = np.mean(Pxx[:,:,idx_freq],axis=2)
                        idx += 1
                    band_features = np.array(band_features)
                if type_psd == "burg":
                    noverlap = 150
                    nperseg = 500
                    fs = 500
                    f_max = 100
                    N_fft = 200
                    filter_order = 19
                    Pxx, Time_freq, time = spectral_analysis.Power_burg_calculation(data,noverlap,N_fft,f_max, nperseg,filter_order)
                
                print(band_features.shape)
                features.append(band_features)
                labels.append(dict_sorted_labels[event_name]*np.ones(data.shape[0]))
        except Exception as e:
            print("Error in file: ", file)
            print(e)

    if len(features) == 1:
        features = np.array(features)
    else:
        features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels,axis=0)
    
    return features,labels
    
# create dataset for coherence
# Create dataset
def coh_dataset_creator(files,list_idx_channels,list_labels,type_="numpy"):

    """
    Take the data from every file and collect only those corresponding to GDF-left or GDF-right in the interest_data list
    Those events last for a number of data corresponding to segment_length

    parameters: files: string list corresponding to the data folders paths
                events: int list corresponding to the numbers representatives of the events of interest
                segment_length: int corresponding to the length of the data segment we will keep
    """
    features = []
    labels = []
    dict_sorted_labels = {list_labels[i]:i for i in range(len(list_labels))}

    for file in files:
        data = mne.io.read_raw_edf(file, preload=True)
        raw_data = data.get_data()
        fs = data.info['sfreq']
        total_events, dict = mne.events_from_annotations(data)
        for i in range(len(total_events)-1):
            start = total_events[i][0]
            if i == len(total_events) - 1:
                end = raw_data.shape[1]
            else:
                end = total_events[i+1][0]
            label = total_events[i][2]
            if label in list_labels:
                coh = np.zeros((len(list_idx_channels),len(list_idx_channels)))
                for j in range(len(list_idx_channels)):
                    for k in range(j+1,len(list_idx_channels)):
                        f,Cxy = sc.signal.coherence(raw_data[list_idx_channels[j], start:end], raw_data[list_idx_channels[k], start:end], fs=fs, nperseg=200, noverlap=100)
                        coh[j,k] = np.mean(Cxy[np.argwhere((f >= 4) & (f <= 40)).flatten()])
                features.append(coh.flatten())
                labels.append(dict_sorted_labels[label])

    features = np.array(features)
    labels = np.array(labels)

    if type_ == "torch":
        features = torch.Tensor(features).unsqueeze(1).float()
        labels = torch.Tensor(labels).long()
    
    return features,labels

class EEG_COH_Dataset_Multi_Subject():
    def __init__(self, parent_folder_path, num_subjects, num_sessions, num_runs, list_idx_channels, list_labels, type_="numpy", band=False, data="physionet"):

        if data == "physionet":
            subject_folders = []
            for i in num_subjects:
                if i < 10:
                    subject_folders.append("S00" + str(i))
                elif i < 100:
                    subject_folders.append("S0" + str(i))
                else:
                    subject_folders.append("S" + str(i))

            file_list = []
            for folder_name in subject_folders:
                subject_folder_path = os.path.join(parent_folder_path, folder_name)

                for i in num_runs:
                    if i < 10:
                        file_list.append(os.path.join(subject_folder_path, folder_name + "R0" + str(i) + ".edf"))
                    else:
                        file_list.append(os.path.join(subject_folder_path, folder_name + "R" + str(i) + ".edf"))
            
            self.file_list = file_list

        if data == "bci":
            subject_folders = os.path.join(parent_folder_path, "sub-" + num_subjects[0])
            sessions_folders = [os.path.join(os.path.join(subject_folders, "ses-0" + str(i)),"EEG") for i in num_sessions]
            files_list = []
            for session_folder in sessions_folders:
                li = os.listdir(session_folder)
                for file in li:
                    if file.endswith(".edf"):
                        files_list.append(os.path.join(session_folder, file))
            self.file_list = files_list
        self.features,self.labels = coh_dataset_creator(self.file_list,list_idx_channels,type_=type_)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def transform_datase_numpy_to_torch(self):
        self.features = torch.Tensor(self.features).unsqueeze(1).float()
        self.labels = torch.Tensor(self.labels).long()
    
class Braccio_Dataset_Multi_Subject():
    def __init__(self, parent_folder_path, num_subjects, num_sessions, list_idx_channels, list_labels, feature_type):

        subject_folders = [os.path.join(parent_folder_path, "sub-0" + str(num_sub)) if num_sub < 10 else os.path.join(parent_folder_path, "sub-" + str(num_sub)) for num_sub in num_subjects]
        sessions_folders = [os.path.join(os.path.join(subject_folders[j], "ses-0" + str(i)),"EEG") for i in num_sessions for j in range(len(num_subjects))]
        files_list = []
        for session_folder in sessions_folders:
            li = [f for f in os.listdir(session_folder) if not f.startswith("._")] 
            for file in li:
                if file.endswith(".edf"):
                    files_list.append(os.path.join(session_folder, file))

        self.file_list = files_list

        if feature_type == "time":
            self.features,self.labels = time_dataset_creator(self.file_list,list_idx_channels,list_labels)
        if feature_type == "psd":
            self.features,self.labels,self.freqs = psd_dataset_creator(self.file_list,list_idx_channels,list_labels)
        if feature_type == "band":
            self.features,self.labels = band_psd_dataset_creator(self.file_list,list_idx_channels,list_labels)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def transform_dataset_numpy_to_torch(self):
        self.features = torch.Tensor(self.features).float()
        self.labels = torch.Tensor(self.labels).long()
    
class Physio_Dataset_Multi_Subject():
    # Create dataset for training on physionet data with multiple subjects

    def __init__(self, parent_folder_path, num_subjects, num_runs, list_idx_channels, list_labels, feature_type):

        subject_folders = []
        for i in num_subjects:
            if i < 10:
                subject_folders.append("S00" + str(i))
            elif i < 100:
                subject_folders.append("S0" + str(i))
            else:
                subject_folders.append("S" + str(i))

        file_list = []
        for folder_name in subject_folders:
            subject_folder_path = os.path.join(parent_folder_path, folder_name)

            for i in num_runs:
                if i < 10:
                    file_list.append(os.path.join(subject_folder_path, folder_name + "R0" + str(i) + ".edf"))
                else:
                    file_list.append(os.path.join(subject_folder_path, folder_name + "R" + str(i) + ".edf"))
        
        self.file_list = file_list

        if feature_type == "time":
            self.features,self.labels = time_dataset_creator(self.file_list,list_idx_channels,list_labels)
        if feature_type == "psd":
            self.features,self.labels,self.freqs = psd_dataset_creator(self.file_list,list_idx_channels,list_labels)
        if feature_type == "band":
            self.features,self.labels = band_psd_dataset_creator(self.file_list,list_idx_channels,list_labels)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def transform_dataset_numpy_to_torch(self):
        self.features = torch.Tensor(self.features).float()
        self.labels = torch.Tensor(self.labels).long()