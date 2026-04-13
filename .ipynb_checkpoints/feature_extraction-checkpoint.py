import torch
from EEGPT_mcae_finetune import EEGPTClassifier
import numpy as np

def segment_signal(data, window_size=2048, stride=1024):
    segments = []
    for start in range(0, data.shape[1] - window_size + 1, stride):
        seg = data[:, start:start+window_size]
        segments.append(seg)
    return np.stack(segments)
    
def extract_features(raw_eeg):
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

    raw_eeg = raw_eeg.copy().pick(['CZ'])
    
    use_channels_names = [rename_map.get(ch, ch) for ch in raw_eeg.ch_names]

    model = EEGPTClassifier(
        num_classes=0,
        in_channels=1,
        img_size=[1, 2048],
        patch_stride=64,
        use_channels_names=use_channels_names,
        use_chan_conv=False,
        use_predictor=True
    )
    model = model.to(device)
    ckpt_path = "../eeg/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"
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