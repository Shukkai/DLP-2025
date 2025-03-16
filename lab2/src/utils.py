# import torch

# def dice_score(predicted: torch.Tensor, ground_truth: torch.Tensor, eps: float = 1e-6) -> float:
#     with torch.no_grad():  # Ensuring no gradients are tracked
#         predicted = predicted.float()
#         ground_truth = ground_truth.float()

#         intersection = torch.sum(predicted * ground_truth)
#         sum_sizes = torch.sum(predicted) + torch.sum(ground_truth)

#         dice = (2.0 * intersection + eps) / (sum_sizes + eps)  # Add epsilon to avoid division by zero
#         return dice.item()

import torch
import argparse

def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    with torch.no_grad():
        sum = 0
        pred_mask = pred_mask > 0.5
        # pred_mask = pred_mask.float().flatten(start_dim=1)
        for i in range(pred_mask.shape[0]):
            intersection = torch.sum(pred_mask[i] * gt_mask[i])
            sum += 2.0 * intersection / (torch.sum(pred_mask[i]) + torch.sum(gt_mask[i]))
    return (sum / pred_mask.shape[0]).item()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training.")
    parser.add_argument("--model", type=str, default= "unet", required=False, help="unet or resnet34")
    parser.add_argument("--root", type=str, default="./lab2/", required=True, help="Root path")
    parser.add_argument("--epoches", type=int, default=101, required=False, help="Epoches #")
    parser.add_argument("--batch", type=int, default= 16 ,required = False, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, required=False, help="Learning rate #")
    parser.add_argument("--lr_scheduler",type=float,default=1e-5, required=False, help="Learning rate scheduler gamma")
    parser.add_argument("--load_pt",type=str,default= "",required=False, help="load .pt with relative path")

    args = parser.parse_args()
    return args