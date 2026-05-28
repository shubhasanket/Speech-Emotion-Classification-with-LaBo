from datasets import load_dataset, Audio
import torch
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
import io
import os

SR = 48000   # for clap
SR_MEL = 16000   # for mel spectrograms
MAX_LEN = SR*4 # 4 secs
N_MELS  = 128


mel_transform = T.MelSpectrogram(sample_rate=SR_MEL, n_fft=1024, hop_length=160, n_mels=N_MELS, f_max=8000)
log_amp = T.AmplitudeToDB(top_db=80)
resample_mel = T.Resample(orig_freq=SR, new_freq=SR_MEL)


def decode_audio(row):
    raw_bytes = row["audio"]["bytes"]
    wav, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != SR:
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav_t = T.Resample(orig_freq=sr, new_freq=SR)(wav_t)
        wav = wav_t.squeeze(0).numpy()

    if len(wav) > MAX_LEN:
        wav = wav[:MAX_LEN]
    else:
        wav = np.pad(wav, (0, MAX_LEN - len(wav)))
    row["wav"] = wav
    return row



class IEMOCAPDataset(torch.utils.data.Dataset):
    def __init__(self, split, label_map):
        self.data = split
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        wav_t = torch.tensor(row["wav"], dtype=torch.float32) 
        wav_mel = resample_mel(wav_t.unsqueeze(0)).squeeze(0)
        log_mel = log_amp(mel_transform(wav_mel)).unsqueeze(0)  # [1 x N_MELS x T]
        label = self.label_map[row["major_emotion"]]
        return {"waveform": wav_t, "mel": log_mel, "label": torch.tensor(label, dtype=torch.long)}



def get_dataloaders(batch_size=16, num_workers=4, seed=42):
    raw = load_dataset("mteb/iemocap")
    raw = raw.cast_column("audio", Audio(decode=False))

    if "train" in raw and "test" in raw:
        train_raw, test_raw = raw["train"], raw["test"]
    else:
        full = raw[list(raw.keys())[0]]
        splits = full.train_test_split(test_size=0.1, seed=seed)
        train_raw, test_raw = splits["train"], splits["test"]

    val_split = train_raw.train_test_split(test_size=0.1, seed=seed)
    train_raw, val_raw = val_split["train"], val_split["test"]

    # caching for faster reruns
    os.makedirs("./data", exist_ok=True)
    print("Decoding audio (cached in ./data)...")
    train_raw = train_raw.map(decode_audio, cache_file_name="./data/train.arrow")
    val_raw = val_raw.map(decode_audio, cache_file_name="./data/val.arrow")
    test_raw = test_raw.map(decode_audio, cache_file_name="./data/test.arrow")

    
    class_names = sorted(set(train_raw["major_emotion"]))
    label_map   = {name: i for i, name in enumerate(class_names)}

    print(f"Train: {len(train_raw)} | Val: {len(val_raw)} | Test: {len(test_raw)}")
    train_ds = IEMOCAPDataset(train_raw, label_map)
    val_ds = IEMOCAPDataset(val_raw, label_map)
    test_ds = IEMOCAPDataset(test_raw, label_map)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names
