import argparse,random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from diffusers import UNet2DModel        
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler
from dataloader import ICLEVERDataset
import argparse
from torch.utils.tensorboard import SummaryWriter
from evaluator import evaluation_model
import os
from diffusers.optimization import get_cosine_schedule_with_warmup


class learn_CondUNet(nn.Module):
    def __init__(self, n_cls, emb_dim=64):
        super().__init__()

        # Richer label embedding projection
        self.linear_projection = nn.Sequential(
            nn.Linear(n_cls, 128),
            nn.SiLU(),  
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, emb_dim),
        )
        # Deeper UNet config
        self.unet = UNet2DModel(
            sample_size=64,                     # latent 8×8
            in_channels=4 + emb_dim,           # concat conditioning
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 768),  # deeper
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D"
            ),
            time_embedding_type="positional",
        )

    def forward(self, lat, t, onehot):         # lat: B×4×8×8
        emb = self.label_embedding(onehot)          # B×64
        emb = emb[:, :, None, None].expand(-1, -1, lat.size(2), lat.size(3))  # B×64×8×8
        x = torch.cat([lat, emb], dim=1)       # B×(4+64)×8×8
        return self.unet(x, t).sample
# # ---------------------------- utils -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def param_groups(model, wd):
#     decay, no_decay = [], []
#     for n, p in model.named_parameters():
#         if p.ndim >= 2 and "norm" not in n and "embedding" not in n:
#             decay.append(p)
#         else:
#             no_decay.append(p)
#     return [{"params": decay, "weight_decay": wd},
#             {"params": no_decay, "weight_decay": 0.0}]



class CondUnet(nn.Module):
    def __init__(self, n_cls, embedding_label_size=4) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(n_cls, embedding_label_size)
        self.model = UNet2DModel(
            sample_size=64,
            in_channels=3 + n_cls,
            out_channels=3,
            time_embedding_type="positional",
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),  
            down_block_types=(
                "DownBlock2D", 
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  
                "AttnUpBlock2D",  
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            # class_embed_type="timestep",
        )
    def forward(self, x, t, label):
        bs, c, w, h = x.shape
        embeded_label = label.view(bs, label.shape[1], 1, 1).expand(bs, label.shape[1], w, h)
        unet_input = torch.cat((x, embeded_label), 1)
        unet_output = self.model(unet_input, t).sample
        return unet_output

class ConditionlDDPM():
    def __init__(self, args, writer):
        self.args = args
        self.writer = writer
        
        self.device = args.device
        self.epochs = args.epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.num_train_timestamps = args.num_train_timestamps
        self.svae_root = args.save_root
        self.train_dataset = ICLEVERDataset(root_dir=self.args.data_dir, json_file="train.json", objects_json="objects.json")
        # self.train_dataset = iclevrDataset(root=self.args.data_dir, mode="train")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.n_cls = len(self.train_dataset.object_to_index)
        self.cfg_weight = args.cfg_weight
        if self.args.cfg_weight > 0:
            self.noise_scheduler = DDIMScheduler(num_train_timesteps=self.num_train_timestamps, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon",clip_sample=True)
            self.noise_scheduler.set_timesteps(self.num_train_timestamps)
        else:
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timestamps, beta_schedule="squaredcos_cap_v2")
            self.noise_scheduler.set_timesteps(self.num_train_timestamps)
        self.noise_predicter = CondUnet(n_cls=self.n_cls).to(self.device)
        self.eval_model = evaluation_model()
        self.best_acc = 0.0
        
        self.optimizer = torch.optim.Adam(self.noise_predicter.parameters(), lr=self.lr)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=len(self.train_dataloader) * self.epochs,
            num_cycles=50
        )
    # # ---------------------- training loss ----------------------
    def diffusion_loss(self,img, lbl, cfg_dropout_p=0.1):
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps-1, (img.size(0),), device=img.device).long()
        noise = torch.randn_like(img)
        noisy = self.noise_scheduler.add_noise(img, noise, t)
        if self.args.cfg_weight > 0:
            drop_mask = (torch.rand(img.size(0), device=img.device) < cfg_dropout_p)
            lbl[drop_mask] = 0.0
        return F.mse_loss(self.noise_predicter(noisy.to(torch.float), t, lbl.to(torch.float)), img)

    def train(self):
        # training 
        for i, epoch in enumerate(range(1, self.epochs+1)):
            loss_sum = 0
            for x, y in tqdm(self.train_dataloader, desc=f"Training/epoch={epoch}", total=len(self.train_dataloader), leave=True):
                x = x.to(self.device)
                y = y.to(self.device)
                loss = self.diffusion_loss(x, y)
                loss.backward()
                nn.utils.clip_grad_value_(self.noise_predicter.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.lr = self.lr_scheduler.get_last_lr()[0]
                
                loss_sum += loss.item()
            avg_loss = loss_sum / len(self.train_dataloader)
            print(f"Training/epoch={epoch}/avg_loss: {avg_loss:.4f}")
            self.writer.add_scalar("Loss/train", avg_loss, epoch)

            if(epoch % 5 == 0 or epoch == 1):
                test_acc = self.evaluate(epoch, split="test", cfg_weight=self.args.cfg_weight)
                self.writer.add_scalar("test_accuracy", test_acc, epoch)
                new_test_acc = self.evaluate(epoch, split="new_test", cfg_weight=self.args.cfg_weight)
                self.writer.add_scalar("new_test_accuracy", new_test_acc, epoch)
                if (test_acc + new_test_acc) / 2 > self.best_acc:
                    self.best_acc = (test_acc + new_test_acc) / 2
                    torch.save({
                        "noise_predicter": self.noise_predicter,
                        "noise_scheduler": self.noise_scheduler,
                        "optimizer": self.optimizer,
                        "lr"        : self.lr,  
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                        "last_epoch": epoch
                    }, os.path.join(self.args.save_root, self.args.ckpt_root, f"best_epoch={epoch}.ckpt"))
                    print(f"save ckpt to {os.path.join(self.args.save_root, self.args.ckpt_root, f'best_epoch={epoch}.ckpt')}")
                
            if (epoch == 1 or epoch % 10 == 0):
                torch.save({
                    "noise_predicter": self.noise_predicter,
                    "noise_scheduler": self.noise_scheduler,
                    "optimizer": self.optimizer,
                    "lr"        : self.lr,
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "last_epoch": epoch
                }, os.path.join(self.args.save_root, self.args.ckpt_root, f"epoch={epoch}.ckpt"))
                print(f"save ckpt to {os.path.join(self.args.save_root, self.args.ckpt_root, f'epoch={epoch}.ckpt')}")

    def evaluate(self, epoch, split="test", cfg_weight=0.0):
        test_dataset = ICLEVERDataset(root_dir=self.args.data_dir, json_file=f"{split}.json", objects_json="objects.json")
        test_dataloader = DataLoader(test_dataset, batch_size=32)
        for y in test_dataloader:
            y = y.to(self.device)
            x = torch.randn(32, 3, 64, 64).to(self.device)
            B = x.shape[0]
            if(cfg_weight == 0):
                for t in tqdm(self.noise_scheduler.timesteps, desc="Sampling", total=self.noise_scheduler.config.num_train_timesteps, leave=False):
                    with torch.no_grad():
                        pred_noise = self.noise_predicter(x.to(torch.float), t.to(torch.float), y.to(torch.float))
                    x = self.noise_scheduler.step(pred_noise, t, x).prev_sample
            else:
                # With CFG
                for t in tqdm(self.noise_scheduler.timesteps, desc="Sampling (CFG)", leave=False):
                    with torch.no_grad():
                        x_2B = torch.cat([x, x], dim=0)
                        y_2B = torch.cat([y, torch.zeros_like(y)], dim=0)
                        eps_2B = self.noise_predicter(x_2B, t, y_2B)
                        eps_c, eps_u = eps_2B.chunk(2)
                        eps = eps_u + cfg_weight * (eps_c - eps_u)
                    x = self.noise_scheduler.step(eps, t, x).prev_sample
            acc = self.eval_model.eval(images=x.detach(), labels=y)
            denormalized_x = (x.detach() / 2 + 0.5).clamp(0, 1)
            print(f"accuracy of {split}.json on epoch {epoch}: ", round(acc, 3))
            generated_grid_imgs = make_grid(denormalized_x)
            save_image(generated_grid_imgs, f"{self.svae_root}/{split}_{epoch}.jpg")
        return round(acc, 3)
    
    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path,weights_only=False)
            self.noise_predicter = checkpoint["noise_predicter"]
            self.noise_scheduler = checkpoint["noise_scheduler"]
            self.optimizer = checkpoint["optimizer"]
            self.lr = checkpoint["lr"]
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint['last_epoch']
    
        
    def progressive_generate_image(self):
        label_one_hot = [0] * 24
        label_one_hot[2] = 1
        label_one_hot[19] = 1
        label_one_hot[3] = 1
        label_one_hot = torch.tensor(label_one_hot).to( self.device)
        label_one_hot = torch.unsqueeze(label_one_hot, 0)
        x = torch.randn(1, 3, 64, 64).to(self.device)
        img_list = []
        for t in tqdm(self.noise_scheduler.timesteps, desc="Progressive Generation", total=self.noise_scheduler.config.num_train_timesteps, leave=False):
            with torch.no_grad():
                pred_noise = self.noise_predicter(x.to(torch.float),t, label_one_hot.to(torch.float))
            x = self.noise_scheduler.step(pred_noise, t, x).prev_sample
            if(t % 100 == 0):
                denormalized_x = (x.detach() / 2 + 0.5).clamp(0, 1)
                save_image(denormalized_x, f"{self.args.save_root}/{t}.jpg")
                img_list.append(denormalized_x)
        grid_img = make_grid(torch.cat(img_list, dim=0), nrow=5)
        save_image(grid_img, f"{self.args.save_root}/progressive_genrate_image.jpg")
    
    def sample_dataset_images(self, model, scheduler, split, save_dir, device, cfg_weight=0.0):
        dataset = ICLEVERDataset(root_dir=self.args.data_dir, json_file=f"{split}.json", objects_json="objects.json")
        dataloader = DataLoader(dataset, batch_size=32)

        model.eval()
        scheduler.set_timesteps(1000)
        total_idx = 0

        for y in tqdm(dataloader, desc="Sampling dataset"):
            y = y.to(device)
            x = torch.randn(32, 3, 64, 64).to(device)

            for t in scheduler.timesteps:
                with torch.no_grad():
                    if cfg_weight > 0:
                        x_2B = torch.cat([x, x], dim=0)
                        y_2B = torch.cat([y, torch.zeros_like(y)], dim=0)
                        eps_2B = model(x_2B, t, y_2B)
                        eps_c, eps_u = eps_2B.chunk(2)
                        eps = eps_u + cfg_weight * (eps_c - eps_u)
                    else:
                        eps = model(x, t, y)
                    x = scheduler.step(eps, t, x).prev_sample

            denormalized_x = (x.detach() / 2 + 0.5).clamp(0, 1)
            for i in range(denormalized_x.size(0)):
                save_path = os.path.join(save_dir, f"img_{total_idx + i}.jpg")
                save_image(denormalized_x[i], save_path)

            total_idx += denormalized_x.size(0)

        print(f"Saved {total_idx} images to {save_dir}")
def main(args):
    writer = SummaryWriter(log_dir=os.path.join(args.save_root, "runs"))
    set_seed(args.seed)
    conditionlDDPM = ConditionlDDPM(args, writer)
    if args.sample_only:
        conditionlDDPM.load_checkpoint()
        os.makedirs(os.path.join(args.save_root, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.save_root, "images", "test"), exist_ok=True)
        conditionlDDPM.sample_dataset_images(conditionlDDPM.noise_predicter, conditionlDDPM.noise_scheduler, "test", os.path.join(args.save_root, "images", "test"), args.device, args.cfg_weight)
        os.makedirs(os.path.join(args.save_root, "images", "new_test"), exist_ok=True)
        conditionlDDPM.sample_dataset_images(conditionlDDPM.noise_predicter, conditionlDDPM.noise_scheduler, "new_test", os.path.join(args.save_root, "images", "new_test"), args.device, args.cfg_weight)
    else:
        if args.eval_only:
            conditionlDDPM.load_checkpoint()
            conditionlDDPM.progressive_generate_image()
        else:
            conditionlDDPM.train()
    writer.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--data_dir', type=str, default="iclevr")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--num_train_timestamps', type=int, default=1000)
    parser.add_argument('--lr_warmup_steps', default=0, type=int)
    parser.add_argument('--save_root', type=str, default="ddpm")
    # ckpt
    parser.add_argument('--ckpt_root', type=str, default="ckpt") # fro save
    parser.add_argument('--ckpt_path', type=str, default=None) # for load
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample_only', action='store_true')
    
    args = parser.parse_args()
    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(os.path.join(args.save_root, args.ckpt_root), exist_ok=True)
    main(args)