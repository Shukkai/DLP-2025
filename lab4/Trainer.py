import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

from torch.utils.tensorboard import SummaryWriter

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    mse = torch.clamp(mse, min=1e-10)  # Avoid log(0)
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
    logvar = torch.clamp(logvar, min=-20, max=20)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    B, C, H, W = mu.shape
    KLD /= (B * C * H * W)
    # KLD /= batch_size  
    return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.current_epoch = current_epoch
        self.tol_epoch = args.num_epoch
        self.kl_type = args.kl_anneal_type
        self.kl_ratio = args.kl_anneal_ratio
        self.kl_cycle = args.kl_anneal_cycle
        if self.kl_type == "None":
            self.beta = 1.0
        else:
            self.beta = 0.0
        
    def update(self):
        self.current_epoch += 1
        if self.kl_type == "Cyclical":
            self.beta = self.frange_cycle_linear(
                step = self.current_epoch,
                n_iter=self.tol_epoch,
                start=0.0,
                stop=1.0,
                n_cycle=self.kl_cycle,
                ratio=self.kl_ratio
            )
        elif self.kl_type == "Monotonic":
            self.beta = min(1.0, self.current_epoch / (self.kl_ratio * 10))
        elif self.kl_type == "None":
            self.beta = 1.0
        else:
            raise ValueError("No such kl annealing type")
    
    def get_beta(self):
        return self.beta

    def frange_cycle_linear(self,step, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        period = n_iter / n_cycle
        cycle_num = int(step / period)
        cycle_pos = step - cycle_num * period
        ramp = period * ratio
        if cycle_pos <= ramp:
            return start + (stop - start) * (cycle_pos / ramp)
        else:
            return stop
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        if args.optim == "Adam":
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr,weight_decay=0.00001)
        elif args.optim == "AdamW":
            self.optim      = optim.AdamW(self.parameters(), lr=self.args.lr,weight_decay=0.00001)
        elif args.optim == "SGD":
            self.optim      = optim.SGD(self.parameters(), lr=self.args.lr,momentum=0.9)
        else:
            raise ValueError("No such optimizer")
        # self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.num_epoch = args.num_epoch
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=self.num_epoch, eta_min=self.args.lr/10.0)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        self.writer = SummaryWriter(log_dir=f"{args.exp_name}_logs")
        self.val_psnr_max = 0.0
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            epoch_loss = 0.0
            
            for step, (img, label) in enumerate(pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                epoch_loss += loss.item()

                beta = self.kl_annealing.get_beta()
                
                self.writer.add_scalar("Train/Step_Loss", loss.item(), self.current_epoch * len(train_loader) + step)
                self.writer.add_scalar("Train/Beta", beta, self.current_epoch * len(train_loader) + step)

                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {:.4f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {:.4f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

            avg_loss = epoch_loss / len(train_loader)
            self.writer.add_scalar("Train/Epoch_Loss", avg_loss, self.current_epoch)
            self.writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], self.current_epoch)
            self.writer.add_scalar('Train/tfr', self.tfr, self.current_epoch)
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.exp_name, f"epoch={self.current_epoch}.ckpt"))

            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        val_loss = 0.0
        val_psnr = 0.0
        for step, (img, label) in enumerate(pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr = self.val_one_step(img, label)
            val_loss += loss.item()
            val_psnr += psnr.item()
            self.tqdm_bar(f'val | PSNR: {psnr:.2f}', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        if val_psnr / len(val_loader) > self.val_psnr_max:
            self.val_psnr_max = val_psnr / len(val_loader)
            self.save(os.path.join(self.args.exp_name, f"best.ckpt"))
        self.writer.add_scalar("Val/Epoch_Loss", val_loss / len(val_loader), self.current_epoch)
        self.writer.add_scalar("Val/Epoch_PSNR", val_psnr / len(val_loader), self.current_epoch)

    def __del__(self):
        self.writer.close()

    def training_one_step(self, imgs, labels, adapt_TeacherForcing):
        # B, T, C, H, W = img.shape
        imgs = imgs.permute(1, 0, 2, 3, 4)  # (T, B, C, H, W)
        labels = labels.permute(1, 0, 2, 3, 4)
        losses = 0.0
        mse_losses, kl_losses = 0.0,0.0
        last_img = imgs[0]
        for t in range(1,self.train_vi_len):
            cur_img = imgs[t]
            if adapt_TeacherForcing:
                last_img = imgs[t-1]

            cur_img = self.frame_transformation(cur_img)
            last_img = self.frame_transformation(last_img)
            label = self.label_transformation(labels[t])

            z, mu, logvar = self.Gaussian_Predictor(cur_img,label)
            output = self.Decoder_Fusion(last_img,label,z)
            # output[output != output] = 0.5
            pred = self.Generator(output)
            # pred = torch.clamp(pred, 0.0, 1.0)
            pred = torch.sigmoid(pred)
            last_img = pred.detach()

            mse_losses += self.mse_criterion(pred,imgs[t])
            kl_losses += kl_criterion(mu,logvar,self.batch_size)
        # print(f"kl losses:{kl_losses}")
        # print(f"mse losses:{mse_losses}")
        self.optim.zero_grad()
        losses = mse_losses + self.kl_annealing.get_beta() * kl_losses
        losses.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer_step()
        return losses


    
    def val_one_step(self, imgs, labels):
        imgs = imgs.permute(1, 0, 2, 3, 4)  # (T, B, C, H, W)
        labels = labels.permute(1, 0, 2, 3, 4)
        with torch.no_grad():
            mse_losses = 0.0
            psnrs = 0.0
            last_img = imgs[0]
            for t in range(1,self.val_vi_len):
                cur_img = imgs[t]
                cur_img = self.frame_transformation(cur_img)
                last_img = self.frame_transformation(last_img)
                label = self.label_transformation(labels[t])
                z, mu, logvar = self.Gaussian_Predictor(cur_img,label)
                z = mu 
                output = self.Decoder_Fusion(last_img,label,z)
                output[output != output] = 0.5
                pred =self.Generator(output)
                # pred = torch.clamp(pred, 0.0, 1.0)
                pred = torch.sigmoid(pred)
                last_img = pred.detach()

                mse_losses += self.mse_criterion(pred,imgs[t])
                # kl_loss = kl_criterion(mu,logvar,self.batch_size)
                # losses += mse + self.kl_annealing.get_beta() * kl_loss

                psnrs += Generate_PSNR(pred,imgs[t])
        return mse_losses, psnrs / self.val_vi_len
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch > self.tfr_sde:
            self.tfr -= self.tfr_d_step
            self.tfr = max(0.0, self.tfr)
        
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    # Set save_root to: experiments/exp1, experiments/exp2, etc.
    os.makedirs(args.save_root, exist_ok=True)
    args.exp_name = os.path.join(args.save_root, args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)

    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()

    if args.test:
        model.eval()
    else:
        model.training_stage()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW","SGD"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default="Cyclical",  choices=["Cyclical", "Monotonic", "None"],    help="KL annealing strategy")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10, help="KL Cyclical annealing changing cycle")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1, help="KL Monotonic annealing growing ratio (ratio * total epoch)")
    
    parser.add_argument('--exp_name', type=str, default="exp1", help="Name of this experiment (used for TensorBoard log grouping)")
    

    args = parser.parse_args()
    
    main(args)
