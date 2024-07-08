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

# Extract the EEG data
def extract_EEG_data(files, list_labels, tmin=0, tmax=4):
    eeg_data = []
    labels = []
    dict_sorted_labels = {list_labels[i]:i for i in range(len(list_labels))}

    for file in files:
        try : 
            for event_name in list_labels:
                raw_training, events_from_annot,event_id = load_file_eeg(filepath = file)
                event_var = select_Event(event_name,raw_training,events_from_annot,event_id,tmin,tmax,64)
                event_var.filter(l_freq=1.0, h_freq=50.0)
                data = event_var.get_data()
                eeg_data.append(data)
                labels.append(dict_sorted_labels[event_name]*np.ones(data.shape[0]))
        except Exception as e:
            print("Error in file: ", file)
            print(e)
        
    eeg_data = np.concatenate(eeg_data, axis=0)
    labels = np.concatenate(labels,axis=0)

    return eeg_data, labels

def preprocess_data(eeg_data, list_idx_channels, polynomial_degree=None):

    # Select the channels
    eeg_data = eeg_data[:,list_idx_channels,:]

    # Polynomial detrending
    if polynomial_degree is not None:
        n_samples, n_channels, n_times = eeg_data.shape
        time = np.arange(n_times)

        for k in range(n_samples):
            for l in range(n_channels):
                values = eeg_data[k,l,:]
                # Fitting a polynomial of degree 1 and 13
                coefficients = np.polyfit(time, values, polynomial_degree)
                poly_function = np.poly1d(coefficients)
                predicted_values = poly_function(time)
                eeg_data[k,l,:] = values - predicted_values

    # Filtering 


    return eeg_data

    
# EEG Dataset for multiple subject
def time_dataset_creator(eeg_data, labels):

    """
    Take the data from every file and collect only those corresponding to GDF-left or GDF-right in the interest_data list
    Those events last for a number of data corresponding to segment_length

    parameters: files: string list corresponding to the data folders paths
                events: int list corresponding to the numbers representatives of the events of interest
                segment_length: int corresponding to the length of the data segment we will keep
    """
    features = eeg_data
    features = torch.from_numpy(features).unsqueeze(1).float()
    labels = torch.Tensor(labels).long()

    return features, labels

# Create dataset for PSD
def psd_dataset_creator(eeg_data,type_psd="welch",fs=500,fmin=4,fmax=100,nfft=300,noverlap=150,nperseg=300,filter_order=19):

    """
    Take the data from every file and collect only those corresponding to GDF-left or GDF-right in the interest_data list
    Those events last for a number of data corresponding to segment_length

    parameters: files: string list corresponding to the data folders paths
                events: int list corresponding to the numbers representatives of the events of interest
                segment_length: int corresponding to the length of the data segment we will keep
    """
    eeg_data = eeg_data
    freqs = None

    if type_psd == "welch":
        f, Pxx = sc.signal.welch(eeg_data, fs = fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
        idx_freq = np.argwhere((f >= fmin) & (f <=fmax)).flatten()
        if freqs is None:
            freqs = f[idx_freq]
        Pxx = Pxx[:,:,idx_freq]
    if type_psd == "burg":
        Pxx, f = spectral_analysis.Power_burg_calculation(eeg_data,noverlap,nfft,fs, nperseg,filter_order)
        idx_freq = np.argwhere((f >= fmin) & (f <=fmax)).flatten()
        Pxx = Pxx[:,:,idx_freq]
        if freqs is None:
            freqs = f[idx_freq]
    features = Pxx
    
    return features,freqs

def band_psd_dataset_creator(eeg_data,type_psd="welch",fs=500,nfft=300,noverlap=150,nperseg=300,filter_order=19):

    """
    Take the data from every file and collect only those corresponding to GDF-left or GDF-right in the interest_data list
    Those events last for a number of data corresponding to segment_length

    parameters: files: string list corresponding to the data folders paths
                events: int list corresponding to the numbers representatives of the events of interest
                segment_length: int corresponding to the length of the data segment we will keep
    """
    eeg_data = eeg_data
    #band_freqs = {"delta":[1, 4],"theta": [4, 8],"alpha": [8, 14],"beta": [14, 31],"gamma": [31, 49]}
    band_freqs = {"band1":[1, 4],"band2": [4, 8],"band3": [8, 12],"band4": [12, 16],"band5": [16, 20], "band6": [20, 28], "band7": [28, 36], "band8": [36,44],}
    freqs = None

    if type_psd == "welch":
        f, Pxx = sc.signal.welch(eeg_data, fs = fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
        band_features = np.zeros((eeg_data.shape[0],eeg_data.shape[1],len(band_freqs.keys())))
        idx = 0
        for key in band_freqs.keys():
            idx_freq = np.argwhere((f >= band_freqs[key][0]) & (f <= band_freqs[key][1])).flatten()
            band_features[:,:,idx] = np.mean(Pxx[:,:,idx_freq],axis=2)
            idx += 1
        band_features = np.array(band_features)
    if type_psd == "burg":
        Pxx, f = spectral_analysis.Power_burg_calculation(eeg_data,noverlap,nfft,fs,nperseg,filter_order)
        band_features = np.zeros((eeg_data.shape[0],eeg_data.shape[1],len(band_freqs.keys())))
        idx = 0
        for key in band_freqs.keys():
            idx_freq = np.argwhere((f >= band_freqs[key][0]) & (f <= band_freqs[key][1])).flatten()
            band_features[:,:,idx] = np.mean(Pxx[:,:,idx_freq],axis=2)
            idx += 1
        band_features = np.array(band_features)

    features = band_features
    
    return features
    
# create dataset for coherence
# Create dataset
def coh_dataset_creator(files,list_idx_channels,list_labels,type_coh="welch",tmin=0,tmax=4,fs=500,fmin=4,fmax=100,nfft=300,noverlap=150,nperseg=300,filter_order=19):

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
            #print(file)
            for event_name in list_labels:
                raw_training, events_from_annot,event_id = load_file_eeg(filepath = file)
                event_var = select_Event(event_name,raw_training,events_from_annot,event_id,tmin,tmax,64)
                data = event_var.get_data()
                fs = event_var.info['sfreq']
                data = data[:,list_idx_channels,:]
                if type_coh == "welch":
                    for idx_chan1 in range(data.shape[1]):
                        for idx_chan2 in range(data.shape[1]):
                            f, Cxy = sc.signal.coherence(data[:,idx_chan1,:],data[:,idx_chan2,:],fs = fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
                            idx_freq = np.argwhere((f >= fmin) & (f <= fmax)).flatten()
                            coh_feature = np.zeros((data.shape[0],data.shape[1],data.shape[1],len(idx_freq)))
                            if freqs is None:
                                freqs = f[idx_freq]
                            coh_feature[:,idx_chan1,idx_chan2,:] = np.abs(Cxy[:,idx_freq])
                            features.append(coh_feature)
                            labels.append(dict_sorted_labels[event_name]*np.ones(data.shape[0]))
                if type_coh == "burg":
                    noverlap = 150
                    nperseg = 500
                    fs = 500
                    f_max = 100
                    N_fft = 200
                    filter_order = 19

        except Exception as e:
            print("Error in file: ", file)
            print(e)

    if len(features) == 1:
        features = np.array(features)
    else:
        features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels,axis=0)
    
    return features,labels,freqs
    
class Braccio_Dataset_Multi_Subject():
    def __init__(self, parent_folder_path, num_subjects, num_sessions, list_idx_channels, list_labels, feature_type, dict_):
        
        # Create the list of folders
        subject_folders = [os.path.join(parent_folder_path, "sub-0" + str(num_sub)) if num_sub < 10 else os.path.join(parent_folder_path, "sub-" + str(num_sub)) for num_sub in num_subjects]
        sessions_folders = [os.path.join(os.path.join(subject_folders[j], "ses-0" + str(i)),"EEG") for i in num_sessions for j in range(len(num_subjects))]

        # Create the list of files
        files_list = []
        for session_folder in sessions_folders:
            li = [f for f in os.listdir(session_folder) if not f.startswith("._")] 
            for file in li:
                if file.endswith(".edf"):
                    files_list.append(os.path.join(session_folder, file))
        self.file_list = files_list

        # Extract the EEG data
        self.eeg_data,self.labels = extract_EEG_data(self.file_list, list_labels, tmin=dict_["tmin"], tmax=dict_["tmax"])

        # Preprocess the EEG data
        self.eeg_data = preprocess_data(self.eeg_data, list_idx_channels, polynomial_degree=dict_["polynomial_degree"])

        # Choose the type of feature to extract
        if feature_type == "time":
            self.features,self.labels = time_dataset_creator(self.eeg_data,self.labels)
        if feature_type == "psd":
            self.features,self.freqs = psd_dataset_creator(self.eeg_data,type_psd=dict_["type_psd"],fs=dict_["fs"],fmin=dict_["fmin"],fmax=dict_["fmax"],nfft=dict_["nfft"],noverlap=dict_["noverlap"],nperseg=dict_["nperseg"],filter_order=dict_["filter_order"])
        if feature_type == "band":
            self.features = band_psd_dataset_creator(self.eeg_data,type_psd=dict_["type_psd"],fs=dict_["fs"],nfft=dict_["nfft"],noverlap=dict_["noverlap"],nperseg=dict_["nperseg"],filter_order=dict_["filter_order"])
        if feature_type == "coh":
            self.features,self.labels,self.freqs = coh_dataset_creator(self.file_list,list_labels,type_psd=dict_["type_psd"],fs=dict_["fs"],fmin=dict_["fmin"],fmax=dict_["fmax"],nfft=dict_["nfft"],noverlap=dict_["noverlap"],nperseg=dict_["nperseg"],filter_order=dict_["filter_order"])
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def transform_dataset_numpy_to_torch(self):
        # Transform the numpy arrays to torch tensors
        self.features = torch.Tensor(self.features).float()
        self.labels = torch.Tensor(self.labels).long()
    
class Physio_Dataset_Multi_Subject():
    """
    Class which creates a dataset for the PhysioNet data
    """

    def __init__(self, parent_folder_path, num_subjects, num_runs, list_idx_channels, list_labels, feature_type, dict_):

        # Create the list of folders
        subject_folders = []
        for i in num_subjects:
            if i < 10:
                subject_folders.append("S00" + str(i))
            elif i < 100:
                subject_folders.append("S0" + str(i))
            else:
                subject_folders.append("S" + str(i))

        # Create the list of files
        file_list = []
        for folder_name in subject_folders:
            subject_folder_path = os.path.join(parent_folder_path, folder_name)

            for i in num_runs:
                if i < 10:
                    file_list.append(os.path.join(subject_folder_path, folder_name + "R0" + str(i) + ".edf"))
                else:
                    file_list.append(os.path.join(subject_folder_path, folder_name + "R" + str(i) + ".edf"))
        self.file_list = file_list

        # Extract the EEG data
        self.eeg_data,self.labels = extract_EEG_data(self.file_list, list_labels, tmin=dict_["tmin"], tmax=dict_["tmax"])

        # Preprocess the EEG data
        self.eeg_data = preprocess_data(self.eeg_data, list_idx_channels, polynomial_degree=dict_["polynomial_degree"])

        # Choose the type of feature to extract
        if feature_type == "time":
            self.features,self.labels = time_dataset_creator(self.eeg_data,self.labels)
        if feature_type == "psd":
            self.features,self.freqs = psd_dataset_creator(self.eeg_data,type_psd=dict_["type_psd"],tmin=dict_["tmin"],tmax=dict_["tmax"],fs=dict_["fs"],fmin=dict_["fmin"],fmax=dict_["fmax"],nfft=dict_["nfft"],noverlap=dict_["noverlap"],nperseg=dict_["nperseg"],filter_order=dict_["filter_order"])
        if feature_type == "band":
            self.features = band_psd_dataset_creator(self.eeg_data,type_psd=dict_["type_psd"],fs=dict_["fs"],nfft=dict_["nfft"],noverlap=dict_["noverlap"],nperseg=dict_["nperseg"],filter_order=dict_["filter_order"])
        if feature_type == "coh":
            self.features,self.labels,self.freqs = coh_dataset_creator(self.file_list,list_labels,type_psd=dict_["type_psd"],tmin=dict_["tmin"],tmax=dict_["tmax"],fs=dict_["fs"],fmin=dict_["fmin"],fmax=dict_["fmax"],nfft=dict_["nfft"],noverlap=dict_["noverlap"],nperseg=dict_["nperseg"],filter_order=dict_["filter_order"])
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def transform_dataset_numpy_to_torch(self):
        # Transform the numpy arrays to torch tensors
        self.features = torch.Tensor(self.features).float()
        self.labels = torch.Tensor(self.labels).long()

class EEG_Dataset():
    """
    Class which creates a dataset for the PhysioNet data
    """

    def __init__(self, files_list, list_idx_channels, list_labels, feature_type, dict_):

        self.file_list = files_list

        # Extract the EEG data
        self.eeg_data,self.labels = extract_EEG_data(self.file_list, list_labels, tmin=dict_["tmin"], tmax=dict_["tmax"])

        # Preprocess the EEG data
        self.eeg_data = preprocess_data(self.eeg_data, list_idx_channels, polynomial_degree=dict_["polynomial_degree"])

        # Choose the type of feature to extract
        if feature_type == "time":
            self.features,self.labels = time_dataset_creator(self.eeg_data,self.labels)
        if feature_type == "psd":
            self.features,self.freqs = psd_dataset_creator(self.eeg_data,type_psd=dict_["type_psd"],fs=dict_["fs"],fmin=dict_["fmin"],fmax=dict_["fmax"],nfft=dict_["nfft"],noverlap=dict_["noverlap"],nperseg=dict_["nperseg"],filter_order=dict_["filter_order"])
        if feature_type == "band":
            self.features = band_psd_dataset_creator(self.eeg_data,type_psd=dict_["type_psd"],fs=dict_["fs"],nfft=dict_["nfft"],noverlap=dict_["noverlap"],nperseg=dict_["nperseg"],filter_order=dict_["filter_order"])
        if feature_type == "coh":
            self.features,self.labels,self.freqs = coh_dataset_creator(self.file_list,list_labels,type_psd=dict_["type_psd"],tmin=dict_["tmin"],tmax=dict_["tmax"],fs=dict_["fs"],fmin=dict_["fmin"],fmax=dict_["fmax"],nfft=dict_["nfft"],noverlap=dict_["noverlap"],nperseg=dict_["nperseg"],filter_order=dict_["filter_order"])
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def transform_dataset_numpy_to_torch(self):
        # Transform the numpy arrays to torch tensors
        self.features = torch.Tensor(self.features).float()
        self.labels = torch.Tensor(self.labels).long()