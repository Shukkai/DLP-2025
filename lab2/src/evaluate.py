import torch
import os
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
from utils import dice_score

def plot_result(name, img, mask, pred_mask, W=256, H=256, num_classes=2, save_dir = "../results"):
    sep_width = 10  # White separator width

    # Extract the first sample from the batch
    # Input image: shape (3, H, W)
    img0 = img[0].cpu().detach().numpy()
    img0 = np.transpose(img0, (1, 2, 0)) * 255.0  # Convert to (H, W, 3) and scale to [0,255]
    img0 = np.clip(img0, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img0).resize((W, H), resample=Image.BILINEAR)

    # Ground truth mask: shape (1, H, W) -> (H, W)
    mask0 = mask[0].cpu().detach().numpy()
    mask0 = np.squeeze(mask0, axis=0)
    # Scale mask: assume values in {0, ..., num_classes-1}
    mask0 = (mask0 * (255 / (num_classes - 1))).astype(np.uint8)
    mask_pil = Image.fromarray(mask0).convert("L").resize((W, H), resample=Image.NEAREST)
    mask_pil = mask_pil.convert("RGB")

    # Predicted mask: shape (1, H, W) -> (H, W)
    pred0 = pred_mask[0].cpu().detach().numpy()
    pred0 = np.squeeze(pred0, axis=0)
    pred0 = (pred0 * (255 / (num_classes - 1))).astype(np.uint8)
    pred_pil = Image.fromarray(pred0).convert("L").resize((W, H), resample=Image.NEAREST)
    pred_pil = pred_pil.convert("RGB")

    # Create a white separator
    separator = Image.new("RGB", (sep_width, H), (255, 255, 255))

    # Create a final blank image to concatenate the three images side by side.
    total_width = W * 3 + sep_width * 2
    final_img = Image.new("RGB", (total_width, H))
    final_img.paste(img_pil, (0, 0))
    final_img.paste(separator, (W, 0))
    final_img.paste(mask_pil, (W + sep_width, 0))
    final_img.paste(separator, (W * 2 + sep_width, 0))
    final_img.paste(pred_pil, (W * 2 + sep_width * 2, 0))
    
    # ---- Save the result ----
    save_dir = os.path.join(save_dir,"result")
    if not os.path.exists(save_dir):
        print(f"Directory '{save_dir}' does not exist. Creating it now...")
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"{name}.png")
    final_img.save(save_path)

def evaluate(device, model, test_loader, args):
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        sum = 0
        for i, (img, mask) in tqdm(enumerate(test_loader), total=len(test_loader)):
            img, mask = img.to(device), mask.to(device)
            pred_mask = model(img)
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask > 0.4 #best 0.4
            sum += dice_score(pred_mask, mask)
            # if i % 10 == 0:
            #     plot_result(i, img=img, mask=mask, pred_mask=pred_mask, save_dir=args.root)
    return sum / len(test_loader)