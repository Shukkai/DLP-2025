import torch
from tqdm import tqdm
from utils import dice_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(test_loader, checkpoint_path="../saved_models/best_model.pt"):
    # implement the evaluation function here
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    with torch.no_grad():
        sum = 0
        for img, mask in tqdm(test_loader):
            img, mask = img.to(device), mask.to(device)
            pred_mask = model(img)
            sum += dice_score(pred_mask, mask)
    return sum / len(data)