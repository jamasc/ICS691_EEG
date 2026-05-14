### contains all functions needed for feature extraction, classification, and other stuff if need be adapted for the pipeline

import torch
from EEGPT_mcae_finetune import EEGPTClassifier
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_channels_names = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']
model = EEGPTClassifier(
    num_classes=0,
    in_channels=19,
    img_size=[19, 128],
    patch_stride=64,
    use_channels_names=use_channels_names,
    use_chan_conv=False,
    use_predictor=False,
    use_out_proj=True,
    desired_time_len=128
).to(device)
ckpt_path = "checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckptmodel.load_state_dict(state_dict, strict=False)
model.eval()

def get_eegpt_features(eeg):
    '''
    input: eeg shape (128, 19)
    output: feature vector (64,)
    '''    
    x = torch.tensor(eeg.T).float().unsqueeze(0).to(device)  # (1, 19, 128)
    with torch.no_grad():
        features = model.forward_features(x)
        return features.squeeze(0).cpu().numpy()
    
def get_mean_eegpt_features(eeg):
    '''
    input: eeg shape (19, t>128)
    output: mean (64)
    '''
    assert(eeg.shape[1] >= 128)
    features = []
    for s in range(0, eeg.shape[1] - 127, 128):
        seg = eeg[:, s:s+128]
        feats = get_eegpt_features(seg.T)
        features.append(feats)

    features = np.array(features)
    mean_features = features.mean(axis=0)
    return mean_features

def get_batch_eegpt_features(eeg):
    assert(eeg.shape[1] >= 128)
    features = []
    assert(eeg.shape[1] >= 128)
    for s in range(0, eeg.shape[1] - 127, 128):
        seg = eeg[:, s:s+128]
        feats = get_eegpt_features(seg.T)
        features.append(feats)

    return np.array(features)
