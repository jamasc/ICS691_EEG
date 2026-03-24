import mne
from mne_connectivity import spectral_connectivity_epochs
import numpy as np
import yasa
from scipy.stats import entropy
import torch
from EEGPT_mcae_finetune import EEGPTClassifier

### biomarker functions

def get_biomarkers(raw_eeg):
    """takes a raw eeg as input and returns mean of:
        alpha
        beta
        theta
        delta
        theta alpha ratio
        slowing index
        coherence
        entropy
    """
    
    data = raw_eeg.get_data()   # shape: channels × time
    sf = raw_eeg.info["sfreq"]  # 500

    # A,B,C,D: alpha, beta, theta, delta band powers 
    bp = yasa.bandpower(
        data,
        sf=sf,
        bands=[
            (1,4,"delta"),
            (4,7,"theta"),
            (8,12,"alpha"),
            (13,30,"beta")
        ]
    )
    alpha = bp['alpha'].mean()
    beta = bp['beta'].mean()
    theta = bp['theta'].mean()
    delta = bp['delta'].mean()

    # E: theta/alpha ratio (high in AD)
    theta_alpha_ratio = bp["theta"] / bp["alpha"]
    theta_alpha_ratio = theta_alpha_ratio.mean()
    
    # F: global slowing index (high indicate dementia)
    slow_power = bp["delta"] + bp["theta"]
    fast_power = bp["alpha"] + bp["beta"]
    slowing_index = slow_power / fast_power
    slowing_index = slowing_index.mean()
    
    # G: coherence (low alpha band coherence in AD)
    data1 = data[np.newaxis, :, :]   # shape: (epochs, channels, time)
    con = spectral_connectivity_epochs(
        data1,
        method="coh",
        sfreq=sf,
        fmin=8,
        fmax=12,
        verbose=False
    )
    coherence = con.get_data().mean()
    
    # H: entropy (lower in AD)
    psd, freqs = mne.time_frequency.psd_array_welch(data, sf)
    entropy_val = entropy(psd, axis=1).mean()

    return alpha, beta, theta, delta, theta_alpha_ratio, slowing_index, coherence, entropy_val

def get_biomarkers_from_path(eeg_path):
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
    return get_biomarkers(raw)


### feature extraction functions

def segment_signal(data, window_size=2048, stride=2048):
    segments = []
    for start in range(0, data.shape[1] - window_size + 1, stride):
        seg = data[:, start:start+window_size]
        segments.append(seg)
    return np.stack(segments)
    
def get_eeg_features(raw_eeg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    rename_map = {
        'Fp1': 'FP1',
        'Fp2': 'FP2',
        'T3': 'T7',
        'T4': 'T8',
        'T5': 'P7',
        'T6': 'P8',
        'Fz': 'FZ',
        'Cz': 'CZ',
        'Pz': 'PZ'
    }

    use_channels_names = [rename_map.get(ch, ch) for ch in raw_eeg.ch_names]

    model = EEGPTClassifier(
        num_classes=0,
        in_channels=19,
        img_size=[19, 2048],
        patch_stride=64,
        use_channels_names=use_channels_names,
        use_chan_conv=False,
        use_predictor=True
    )
    model = model.to(device)
    ckpt_path = "checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Extract features
    data = raw_eeg.get_data()
    segments = segment_signal(data)
    data_batch = torch.tensor(segments, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        features = model.forward_features(data_batch)
    
    final_features = features.mean(dim=0).cpu()
    
    return final_features

def get_eeg_features_from_path(eeg_path):
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
    return get_eeg_features(raw)

def save_eeg_features_from_path(eeg_path, store_path, identification):
    eeg_features = get_eeg_features_from_path(eeg_path)
    save_path = store_path + f"/features_{identification}.pt"
    torch.save(eeg_features, save_path)
    return save_path