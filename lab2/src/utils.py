import torch
import argparse
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def plot_metrics(save_dir, epochs, loss_values, dice_scores):
    plt.figure(figsize=(10, 5))
    
    epoch_range = list(range(1, epochs + 1))
    
    plt.plot(epoch_range, loss_values, label="Loss", color="blue")
    plt.plot(epoch_range, dice_scores, label="Dice Score", color="orange")
    
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss and Dice Score Over Epochs")
    plt.legend()
    plt.grid(True)
    
    # Ensure the directory for save_dir exists
    output_dir = os.path.dirname(save_dir)
    if not os.path.exists(output_dir):
        print(f"Directory '{output_dir}' does not exist. Creating it now...")
        os.makedirs(output_dir)
    
    plt.savefig(save_dir, bbox_inches="tight")
    plt.close()
    # plt.show()

def dice_score(predicted: torch.Tensor, ground_truth: torch.Tensor, eps: float = 1e-6) -> float:
    with torch.no_grad():  # Ensuring no gradients are tracked
        predicted = predicted.float()
        ground_truth = ground_truth.float()

        intersection = torch.sum(predicted * ground_truth)
        sum_sizes = torch.sum(predicted) + torch.sum(ground_truth)

        dice = (2.0 * intersection + eps) / (sum_sizes + eps)  # Add epsilon to avoid division by zero
        return dice.item()  

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training.")
    parser.add_argument("--model", type=str, default= "unet", required=False, help="unet or resnet34")
    parser.add_argument("--root", type=str, default="../", required=False, help="Root path")
    parser.add_argument("--epoches", type=int, default=300, required=False, help="Epoches #")
    parser.add_argument("--batch", type=int, default= 128 ,required = False, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, required=False, help="Learning rate #")
    parser.add_argument("--optim", type=str, default="AdamW", required=False, help="Optimizer: AdamW, SGD")
    parser.add_argument("--scheduler", type=str, default="Cos", required=False, help="Scheduler: Cos, Exp")
    parser.add_argument("--lr_eta",type=float,default=1e-5, required=False, help="Learning rate scheduler eta")
    parser.add_argument("--loss_func", type=str, default= "DiceBCE", required=False, help="BCE or DiceBCE")
    parser.add_argument("--load_pt",type=str,default= "",required=False, help="Load .pth with relative path")
    parser.add_argument("--cuda",type=str,default= "0",required=False, help="Using device #")
    args = parser.parse_args()
    return args

class DiceLoss(nn.Module):
    """
    Computes the Dice Loss, which is defined as 1 - Dice Coefficient.
    
    Dice Coefficient:
      Dice = (2 * intersection + smooth) / (sum(inputs) + sum(targets) + smooth)
    """
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten the input and target tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice loss.
    
    Args:
        weight_bce (float): Weight of the BCE component.
        weight_dice (float): Weight of the Dice component.
        use_logits (bool): If True, use BCEWithLogitsLoss (inputs are logits).
                           Otherwise, use BCELoss (inputs are probabilities).
    """
    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5, use_logits: bool = True):
        super(BCEDiceLoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.use_logits = use_logits
        
        if self.use_logits:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute BCE loss
        loss_bce = self.bce(inputs, targets)
        
        # For Dice, if using logits, convert inputs to probabilities with sigmoid
        if self.use_logits:
            inputs = torch.sigmoid(inputs)
        loss_dice = self.dice(inputs, targets)
        
        # Combine the losses using the provided weights.
        return self.weight_bce * loss_bce + self.weight_dice * loss_dice

def read_vals(file_path):
    """
    Reads a text file and returns a list of float loss values.
    Non-numeric lines (like headers) are skipped.
    """
    losses = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                loss = float(line.strip())
                losses.append(loss)
            except ValueError:
                continue  # Skip lines that cannot be converted to float
    return losses

def plot_curve(file1, file2, epochs=300, ptype = "Loss"):
    """
    Reads loss values from two files and plots them over a specified number of epochs.
    The first file is plotted in blue, the second in orange.
    """
    # Read losses from each file
    losses1 = read_vals(file1)
    losses2 = read_vals(file2)
    
    # Use only the first 'epochs' values
    losses1 = losses1[:epochs]
    losses2 = losses2[:epochs]
    
    # Create an x-axis corresponding to epochs
    epoch_range = list(range(1, epochs + 1))
    
    # Plot the two loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_range, losses1, color='blue', label='CosineAnnea')
    plt.plot(epoch_range, losses2, color='orange', label='Exponential')
    plt.xlabel('Epochs')
    plt.ylabel(f"{ptype}")
    plt.title(f"{ptype} Curves")
    plt.legend()
    # Ensure the directory for save_dir exists
    save_dir = "./saved_fig"
    output_dir = os.path.dirname(save_dir)
    if not os.path.exists(output_dir):
        print(f"Directory '{output_dir}' does not exist. Creating it now...")
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(save_dir,f"{ptype}_cmp.png"), bbox_inches="tight")
    plt.close()
    # plt.show()

# if __name__ == "__main__":
#     plot_losses = plot_curve(file1="./saved_data/resnet_BCE_0.001_AdamW_Cos_loss_cmp.txt",file2="./saved_data/resnet_BCE_0.001_AdamW_Exp_loss_cmp.txt",epochs=300,ptype="loss")
#     plot_dices = plot_curve(file1="./saved_data/resnet_BCE_0.001_AdamW_Cos_dice_cmp.txt",file2="./saved_data/resnet_BCE_0.001_AdamW_Exp_dice_cmp.txt",epochs=300,ptype="dice")