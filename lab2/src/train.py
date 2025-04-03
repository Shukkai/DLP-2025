from tqdm import tqdm
import torch
import random
import numpy as np
import os
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler

from oxford_pet import OxfordPetData
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import dice_score, parse_arguments, BCEDiceLoss, plot_metrics
from evaluate import evaluate

seed = 42

# Set the Python built-in random module seed
random.seed(seed)

# Set the NumPy random seed
np.random.seed(seed)

# Set the PyTorch random seed for CPU
torch.manual_seed(seed)

# If using GPU, set the PyTorch CUDA random seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# For deterministic behavior in cuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(args, model, train_loader, valid_loader):
    losses, dices = [], []
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    if args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optim == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    if args.scheduler == "Cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoches, eta_min=args.lr_eta)
    else:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    if args.loss_func == "BCE":
        critirion = nn.BCEWithLogitsLoss()
    else:
        critirion = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5, use_logits=True)
    scaler = GradScaler()  # Helps stabilize float16 training
    save_dir = os.path.join(args.root,"saved_models/")
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    checkpoint_path = os.path.join(save_dir, f"{args.model}_best_model.pth")
    best_dice = 0.0
    for ep in range(args.epoches):
        train_loss = 0.0
        model.train()

        # Training Loop (FP16)
        progress_bar = tqdm(iter(train_loader), desc=f"Epoch {ep+1}/{args.epoches}", leave=True)
        for img, mask in progress_bar:
            img, mask = img.to(device), mask.to(device)

            optimizer.zero_grad()
            # Enable Mixed Precision (float16)
            with autocast():  
                pred_mask = model(img)  # Ensure model outputs (N, C, H, Wd)
                loss = critirion(pred_mask, mask)  # Compute loss

            # Backward pass using FP16 scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")  # Update tqdm bar with real-time loss

        train_loss /= len(train_loader)

        # Validation Loop (FP16, No Gradients)
        model.eval()
        with torch.no_grad():
            sum_dice = 0
            for img, mask in tqdm(valid_loader, desc="Validating", leave=False):
                img, mask = img.to(device), mask.to(device)

                with autocast():
                    pred_mask = model(img)
                pred_mask = torch.sigmoid(pred_mask)
                sum_dice += dice_score(pred_mask, mask)
            dice = sum_dice / len(valid_loader)

        if dice > best_dice:
            best_dice = dice
            torch.save({
                'epoch': ep + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice_score': best_dice
            }, checkpoint_path)
            print(f"✅ Saved Best Model at {checkpoint_path} (Epoch {ep+1}, Dice Score: {best_dice:.4f})")

        print(f"Epoch {ep+1}, Loss: {train_loss:.4f}, Dice Score: {dice:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Step the LR Scheduler
        scheduler.step()

        # Save Metrics
        losses.append(train_loss)
        dices.append(dice)
    # with open(f"{args.root}/saved_data/{args.model}_{args.loss_func}_{args.learning_rate}_{args.optim}_{args.scheduler}_loss_cmp.txt","a") as f:
    #     f.write(f"losses :\n")
    #     output_str = "\n".join(str(item) for item in losses)
    #     f.write(output_str)
    # with open(f"{args.root}/saved_data/{args.model}_{args.loss_func}_{args.learning_rate}_{args.optim}_{args.scheduler}_dice_cmp.txt","a") as f:
    #     f.write(f"dices :\n")
    #     output_str = "\n".join(str(item) for item in dices)
    #     f.write(output_str)
    # save_dir = os.path.join(args.root,"saved_fig",f"{args.model}_{args.loss_func}_lr{args.learning_rate}_loss_dice.png")
    # plot_metrics(save_dir = save_dir,epochs= args.epoches, loss_values=losses, dice_scores= dices)


if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    if args.model == "unet":
        print("Using unet")
        model = UNet(3,1).to(device)
    else:
        print("Using resnet34_unet")
        model = ResNet34_UNet(3,1).to(device)
    print("####Loading Data####")
    data_module = OxfordPetData(root_dir=args.root, batch_size=args.batch, num_workers=4)
    train_loader, valid_loader, test_loader = data_module.get_dataloaders()
    print("####Training####")
    train(args= args,model=model, train_loader=train_loader, valid_loader=valid_loader)
    
# class data_prefetcher:
#     def __init__(self, loader):
#         self.loader = iter(loader)
#         self.stream = torch.cuda.Stream()
#         self.preload()

#     def preload(self):
#         try:
#             self.next_batch = next(self.loader)
#         except StopIteration:
#             self.next_input = None
#             self.next_target = None
#             return

#         with torch.cuda.stream(self.stream):
#             self.next_input = self.next_batch[0].cuda(non_blocking=True).float()
#             self.next_target = self.next_batch[1].cuda(non_blocking=True)

#     def next(self):
#         torch.cuda.current_stream().wait_stream(self.stream)
#         input = self.next_input
#         target = self.next_target
#         self.preload()
#         return input, target
