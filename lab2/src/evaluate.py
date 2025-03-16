import torch
from tqdm import tqdm
from utils import dice_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(model, test_loader):
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        sum = 0
        for img, mask in tqdm(test_loader):
            img, mask = img.to(device), mask.to(device)
            pred_mask = model(img)
            pred_mask = torch.sigmoid(pred_mask)
            sum += dice_score(pred_mask, mask)
    return sum / len(test_loader)