#!/usr/bin/env python
# ------------------------------------------------------------
# conditional_ddpm.py  –  64×64 ICLEVR DDPM with AMP + EMA + CFG
# ------------------------------------------------------------
import argparse, copy, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from diffusers import UNet2DModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataloader import ICLEVERDataset          # your dataset class
from evaluator import evaluation_model

LATENT_SCALE = 0.1821
# ------------------------------------------------------------
def build_loader(args):
    ds = ICLEVERDataset(args.root_dir, args.ann_file, args.objs_file)
    dl = DataLoader(ds,
                    batch_size=args.batch,
                    shuffle=ds.mode == "train",
                    num_workers=args.num_workers,
                    pin_memory=True)
    return dl, ds.num_classes
# ---------------------------- utils -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def param_groups(model, wd):
    """weight-decay only for weights, not for norm/bias."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim >= 2 and "norm" not in n and "embedding" not in n:
            decay.append(p)
        else:
            no_decay.append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]

# -------------------------- model ---------------------------
class CondUNet(nn.Module):
    def __init__(self, labels_num=24, embedding_label_size=4) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(labels_num, embedding_label_size)
        # self.label_embedding = nn.Linear(labels_num, 512)
        # self.timestep_embedding = TimestepEmbedding()
        self.model = UNet2DModel(
            sample_size=64,
            # in_channels=3+labels_num * embedding_label_size,
            in_channels=4 + labels_num,
            out_channels=4,
            time_embedding_type="positional",
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
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
# -------------------- diffusion loss -----------------------
def diffusion_loss(args, img, lbl, *, net, vae, sched, cfg_dropout_p=0.1):
    """ε-prediction MSE with classifier-free dropout."""
    with torch.no_grad():
        # lat = vae.encode(img).latent_dist.sample() * 0.18215   # B×4×8×8
        lat = vae.encode(img).latent_dist.mean * LATENT_SCALE

    t   = torch.randint(0, sched.config.num_train_timesteps,
                        (img.size(0),), device=img.device).long()
    noise = torch.randn_like(lat)
    noisy = sched.add_noise(lat, noise, t)

    # CFG dropout
    if args.use_cfg:
        drop_mask = (torch.rand(img.size(0), device=img.device) < cfg_dropout_p)
        lbl[drop_mask] = torch.zeros_like(lbl[drop_mask])
    eps = net(noisy, t, lbl)
    return F.mse_loss(eps, noise)

# ------------------------ sampler --------------------------
@torch.no_grad()
def sample_cfg(args, net, vae, cond, *, w=3.0):
    """duplicate-batch CFG sampler — always uses EMA weights."""
    B, H, W = cond.size(0), 8, 8
    lat   = torch.randn(B, 4, H, W, device=cond.device)
    sched = DDPMScheduler(1000, beta_schedule="squaredcos_cap_v2")
    sched.set_timesteps(args.n_steps)

    if args.use_cfg:
        lat_2B   = torch.cat([lat, lat])
        label_2B = torch.cat([cond, torch.zeros_like(cond)])

        for t in sched.timesteps:
            with torch.cuda.amp.autocast():
                eps_dual = net(lat_2B, t, label_2B)
                eps_c, eps_u = eps_dual.chunk(2)
                eps = eps_u + w * (eps_c - eps_u)

            lat = sched.step(eps, t, lat).prev_sample
            lat_2B = torch.cat([lat, lat])          # rebuild duplicate
    else:
        for t in sched.timesteps:
            lat = sched.step(net(lat, t, cond), t, lat).prev_sample

    return vae_decode(vae, lat)


@torch.no_grad()
def vae_decode(vae, lat):
    img = vae.decode(lat / LATENT_SCALE).sample
    return (img.clamp(-1, 1) + 1) / 2           # to [0,1]

@torch.no_grad()
def run_eval(args, net_ema, vae, acc_model, *, split="test", n_steps=50):
    """
    Evaluate model on a split (test/new_test) using sample_cfg for generation.
    """
    ds = ICLEVERDataset(args.root_dir, f"{split}.json", args.objs_file)
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    correct = total = 0
    vis_images = []

    print(f"Evaluating {split} at epoch")
    for labels in tqdm(dl, desc=f"eval {split}"):
        labels = labels.to(args.device)

        imgs_rgb = sample_cfg(args, net_ema, vae, labels, w=args.w)

        acc = acc_model.eval(imgs_rgb, labels)
        correct += acc * int(labels.sum().item())
        total += int(labels.sum().item())

        if len(vis_images) < 32:
            vis_images.append(imgs_rgb[:8].cpu())

    # Save visualization (optional)
    if vis_images:
        grid = make_grid(torch.cat(vis_images[:4], dim=0), nrow=4)
        save_image(grid, args.out_dir / f"{split}_grid_eval.png")

    acc_final = correct / total
    print(f"{split} accuracy = {acc_final:.3f}")
    return acc_final



# -------------------------- main ---------------------------
def main():
    # ---------- CLI ----------
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir",  type=Path, required=True)
    ap.add_argument("--ann_file",  type=Path, default="train.json")
    ap.add_argument("--objs_file", type=Path, default="objects.json")
    ap.add_argument("--epochs",    type=int,   default=300)
    ap.add_argument("--batch",     type=int,   default=64)
    ap.add_argument("--accum",     type=int,   default=4)   # grad-accum
    ap.add_argument("--lr",        type=float, default=1e-4)
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--device",    type=str,
                     default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir",   type=Path,  default=Path("runs/prototype"))
    ap.add_argument("--ckpt",      type=Path)
    ap.add_argument("--sample_only", action="store_true")
    ap.add_argument("--num_workers", type=int, default=32)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--w", type=float, default=3.0)
    ap.add_argument("--sampler", type=str, default="DDPM")
    ap.add_argument("--n_steps", type=int, default=1000)
    ap.add_argument("--cfg_dropout_p", type=float, default=0.1)
    ap.add_argument("--use_cfg", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # ---------- data ----------
    dl, n_cls = build_loader(args)

    # ---------- model & EMA ----------
    net = CondUNet(labels_num=n_cls,embedding_label_size=4).to(args.device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    if args.ckpt and args.ckpt.exists():
        net.load_state_dict(torch.load(args.ckpt, map_location=args.device))
        ema.load_state_dict(torch.load(args.ckpt, map_location=args.device))
        print(f"Loaded checkpoint {args.ckpt}")


    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(args.device)
    vae.requires_grad_(False)
    sched = DDPMScheduler(1000, beta_schedule="squaredcos_cap_v2")
    acc_model = evaluation_model()
    if args.eval_only:
        run_eval(args, ema, vae, acc_model)
        return

    # ---------- optimiser ----------
    opt = torch.optim.AdamW(param_groups(net, 0.01), lr=args.lr, betas=(0.9, 0.95))
    total_steps = args.epochs * len(dl) // args.accum
    lr_sched = get_cosine_schedule_with_warmup(opt, 2_000, total_steps)

    scaler = torch.amp.GradScaler('cuda')
    ema_decay = 0.9999

    # ---------- sample-only ----------
    if args.sample_only:
        lbl = torch.stack([dl.dataset[i][1] for i in range(16)]).to(args.device)
        img = sample_cfg(args, ema, vae, lbl, w=args.w)
        save_image(make_grid(img, nrow=4), args.out_dir / "sample_grid.png")
        print("sample_grid.png saved")
        return

    # ---------- training ----------
    step = 0
    for ep in range(1, args.epochs + 1):
        net.train(); running = 0.0
        for i, (img, lbl) in enumerate(tqdm(dl, leave=False)):
            img, lbl = img.to(args.device), lbl.to(args.device)
            with torch.amp.autocast('cuda'):
                loss = diffusion_loss(args, img, lbl, net=net, vae=vae, sched=sched)

            scaler.scale(loss / args.accum).backward()
            running += loss.item()

            # update weights every 'accum' steps
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
                lr_sched.step(); step += 1

                # EMA update
                with torch.no_grad():
                    for p_ema, p in zip(ema.parameters(), net.parameters()):
                        p_ema.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

        avg = running / len(dl)
        tqdm.write(f"epoch {ep} | loss {avg:.4f} | lr {lr_sched.get_last_lr()[0]:.2e}")
        # sample grid
        if ep % 5 == 0 or ep == 1:
            ema.eval()
            # lbl16 = torch.stack([dl.dataset[i][1] for i in range(16)]).to(args.device)
            # grid = sample_cfg(args, ema, vae, lbl16, w=args.w)
            # save_image(make_grid(grid, nrow=4), args.out_dir / f"grid_ep{ep:03d}.png")
            acc_model = evaluation_model()
            acc1 = run_eval(args, ema, vae, acc_model, split='test')
            acc2 = run_eval(args, ema, vae, acc_model, split='new_test')
            if (acc1+acc2)/2 > best_acc:
                best_acc = (acc1+acc2)/2
                torch.save(ema.state_dict(), args.out_dir / f"ema_best.pt")
                torch.save(net.state_dict(), args.out_dir / f"net_best.pt")

        # checkpoint
        if ep % 10 == 0 or ep == args.epochs:
            torch.save(ema.state_dict(), args.out_dir / f"ema_ep{ep:03d}.pt")
            torch.save(net.state_dict(), args.out_dir / f"net_ep{ep:03d}.pt")

    print("Training finished ✔")


if __name__ == "__main__":
    main()