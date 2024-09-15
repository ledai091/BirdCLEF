import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import BirdNet
from dataset import BirdDataset
from processing import read_data, stratified_train_test_split, balance_df
from sklearn.metrics import f1_score
# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sr = 32000  # sampling rate
seq_len = sr * 5
width = int(sr * 0.5)  # frame_size
hop = int(sr * 0.25)  # hop_size

# Set cuDNN configurations
torch.backends.cudnn.benchmark = True

# Data Preparation
df = read_data()
train, test = stratified_train_test_split(df, test_size=0.2)
train, aug_prob = balance_df(train)
label2idx = {label: idx for idx, label in enumerate(train['primary_label'].unique())}
idx2label = {idx: label for idx, label in enumerate(train['primary_label'].unique())}
train_dataset = BirdDataset(train, width, hop, seq_len, device, label2idx)
test_dataset = BirdDataset(test, width, hop, seq_len, device, label2idx)

def ddp_setup(rank, world_size):
    print(f"Setting up DDP for rank {rank} with world size {world_size}")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(self, model, train_loader, test_loader, optimizer, gpu_id, save_every, criterion):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.save_every = save_every
        self.criterion = criterion
        self.model = DDP(model, device_ids=[gpu_id])
    
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        source = source.to(dtype=torch.float32)  # Ensure the input is float32
        output = self.model(source)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def _run_epoch(self, epoch):
        print(f"[GPU{self.gpu_id}] Starting epoch {epoch}")
        b_sz = len(next(iter(self.train_loader))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        self.train_loader.sampler.set_epoch(epoch)
        train_loss = 0.0
        for batch_idx, (source, targets) in enumerate(self.train_loader):
            source = source.to(self.gpu_id).to(dtype=torch.float32)  # Ensure the input is float32
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            train_loss += loss
            if batch_idx % 100 == 0:
                print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss:.6f}")
        preds = []
        ys = []
        val_loss = 0.0
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.gpu_id).to(torch.float32)
                y = y.to(self.gpu_id)
                
                pred = self.model(X)
                preds.append(pred)
                ys.append(y)
                
                loss = self.criterion(pred, y)
                val_loss += loss
            preds = torch.cat(preds, dim=0).detach().cpu().numpy()
            ys = torch.cat(ys, dim=0).detach().cpu().numpy()
            
            scores = []
            for i in range(preds.shape[-1]):
                score = f1_score(ys[:, i], (preds[:, i] > .5).astype(int))
                scores.append(score)
            
            plt.figure(figsize=(15, 5))
            plt.bar(range(len(scores)), scores)
            plt.savefig(f'epoch_{epoch}_f1_scores.png')
            plt.close()
        print(f'Average Training Loss: {train_loss/len(self.train_loader)} | Average Validation Loss: {val_loss/len(self.test_loader)}')
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = 'checkpoint.pt'
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
                
def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=DistributedSampler(train_dataset), num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=DistributedSampler(test_dataset), num_workers=0)
    model = BirdNet().to(rank)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model, train_loader, test_loader, optimizer, rank, save_every, criterion)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
