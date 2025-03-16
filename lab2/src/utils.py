import torch
import argparse

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
    parser.add_argument("--root", type=str, default="./lab2/", required=True, help="Root path")
    parser.add_argument("--epoches", type=int, default=101, required=False, help="Epoches #")
    parser.add_argument("--batch", type=int, default= 16 ,required = False, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, required=False, help="Learning rate #")
    parser.add_argument("--lr_scheduler",type=float,default=1e-5, required=False, help="Learning rate scheduler gamma")
    parser.add_argument("--load_pt",type=str,default= "",required=False, help="Load .pt with relative path")
    parser.add_argument("--cuda",type=str,default= "0",required=False, help="Using device #")
    args = parser.parse_args()
    return args

import torch.nn as nn
import torch.nn.functional as F

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