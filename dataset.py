import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import librosa

class BirdDataset(Dataset):
    def __init__(self, df, width, hop, seq_len, device, label2idx):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.width = width
        self.hop = hop
        self.seq_len = seq_len
        self.device = device
        self.label2idx = label2idx
    
    def __len__(self):
        return len(self.df)
    
    def prepare_X(self, path):
        wav, _ = librosa.load(path)
        
        if len(wav) < self.seq_len:
            wav = np.append(wav, np.zeros(self.seq_len - len(wav)))
        
        start_idx = np.random.choice(range(len(wav) - self.seq_len)) if len(wav) > self.seq_len else 0
        wav = wav[start_idx : start_idx + self.seq_len]
        num_slices = int((len(wav) - self.width) / (self.width - self.hop))
        wav = [wav[i*self.width - i*self.hop: (i+1)*self.width - i*self.hop] for i in range(num_slices)]
        return np.array(wav, dtype=np.float32)
    
    def __getitem__(self, idx):
        if self.__len__() <= idx:
            raise KeyError
        
        wav_path = self.df.loc[idx, 'filepath']
        wav = self.prepare_X(wav_path)
        wav = torch.tensor(wav, dtype=torch.float32).to(self.device)
        
        label = self.df.loc[idx, 'primary_label']
        label = self.label2idx[label]
        y = torch.zeros(264)
        y[label] = 1
        label = torch.tensor(label).to(self.device)
        return wav, y