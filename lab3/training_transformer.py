import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

seed = 42
# Set the Python built-in random module seed
# random.seed(seed)
# Set the NumPy random seed
np.random.seed(seed)
# Set the PyTorch random seed for CPU
torch.manual_seed(seed)

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.ckpt_path = args.save_path
        self.prepare_training(self.ckpt_path)
        self.device = args.device
        self.tol_epochs = args.epochs
        self.lr = args.learning_rate
        self.optim,self.scheduler = self.configure_optimizers(self.tol_epochs,self.lr)
        
        
    @staticmethod
    def prepare_training(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)

    # FP 16 scaling
    def train_one_epoch(self, train_loader, epoch):
        self.model.train()  # Set model to training mode
        losses = 0.0
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.tol_epochs}", leave=True)

        for img in progress_bar:
            img = img.to(self.device)
            self.optim.zero_grad()
            with torch.cuda.amp.autocast():
                pred_y, y = self.model(img)
                loss = F.cross_entropy(pred_y, y)
            scaler.scale(loss).backward()
            scaler.step(self.optim)
            scaler.update()
            losses += loss.item()
            lr = self.optim.param_groups[0]["lr"]
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        self.scheduler.step()
        return losses / len(train_loader)

    def eval_one_epoch(self, val_loader):
        self.model.eval()
        losses = 0.0
        with torch.no_grad():
            for i, img in tqdm(enumerate(val_loader), total=len(val_loader)):
                img = img.to(self.device)
                pred_y, y = self.model(img)
                loss = F.cross_entropy(pred_y, y)
                losses += loss.item()
        return losses / len(val_loader)

    def configure_optimizers(self, epoches, lr):
        optimizer = torch.optim.AdamW(self.model.parameters(),lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches, eta_min=lr/10.0)
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab3_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab3_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./VQGAN/checkpoints/VQGAN.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')
    parser.add_argument('--save_path', type=str, default='./transformer_checkpoints', help='Path to save transformer checkpoint.')
    args = parser.parse_args()
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    min_val_loss = 1000.0
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = train_transformer.train_one_epoch(train_loader,epoch)
        val_loss = train_transformer.eval_one_epoch(val_loader)
        print(f"Epoch : {epoch}, Training Loss : {train_loss:.4f}, Validation Loss : {val_loss:.4f}, LR : {train_transformer.scheduler.get_last_lr()[0]:.6f}")
        if  min_val_loss > val_loss:
            min_val_loss = val_loss
            torch.save(train_transformer.model.transformer.state_dict(),f"{args.save_path}/Epoch_{epoch}.pt")
            print(f"✅ Saved Model at {args.save_path}/Epoch_{epoch}.pt, (Validation Loss = {min_val_loss:.4f})")
            continue
        if epoch % 10 == 0:
            torch.save(train_transformer.model.transformer.state_dict(),f"{args.save_path}/Epoch_{epoch}.pt")
            print(f"✅ Saved Model at {args.save_path}/Epoch_{epoch}.pt, (Validation Loss = {min_val_loss:.4f})")
            
