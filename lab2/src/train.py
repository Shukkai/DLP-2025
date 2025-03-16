from tqdm import tqdm
import argparse
import torch
import os
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler

from oxford_pet import OxfordPetData
from models.unet import UNet
from utils import dice_score
from evaluate import evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def parse_arguments():
    parser = argparse.ArgumentParser(description="Training.")
    parser.add_argument("--model", type=str, default= "unet", required=False, help="unet or resnet34")
    parser.add_argument("--path", type=str, default="../dataset/oxford-iiit-pet/", required=True, help="Data root path(relative)")
    parser.add_argument("--epoches", type=int, default=101, required=False, help="Epoches #")
    parser.add_argument("--batch", type=int, default= 8 ,required = False, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, required=False, help="Learning rate #")

    args = parser.parse_args()
    return args

def train(args, model, train_loader, valid_loader):
    losses, dices = [], []
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    critirion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()  # Helps stabilize float16 training

    save_dir = "../saved_models"
    os.makedirs(save_dir, exist_ok=True)  # 🔥 Ensure directory exists
    checkpoint_path = os.path.join(save_dir, "best_model.pt")
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
                pred_mask = model(img)  # Ensure model outputs (N, C, H, W)
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
        # scheduler.step()

        # Save Metrics
        losses.append(train_loss)
        dices.append(dice)


if __name__ == "__main__":
    args = parse_arguments()
    data_module = OxfordPetData(root_dir=args.path, batch_size=args.batch, num_workers=4)
    train_loader, valid_loader, test_loader = data_module.get_dataloaders()
    model = UNet(3,1).to(device)
    train(args= args,model=model, train_loader=train_loader, valid_loader=valid_loader)
    test_dice = evaluate(test_loader)
    print(f"🚀 Test Dice Score (Merged Classes): {test_dice:.4f}")
    
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
