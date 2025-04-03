import torch
import os
import random
import numpy as np

from utils import parse_arguments
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate
from oxford_pet import OxfordPetData

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

if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    if args.model == "unet":
        print("Using unet")
        model = UNet(3,1).to(device)
    else:
        print("Using resnet34_unet")
        model = ResNet34_UNet(3,1).to(device)
    if args.load_pt != "":
        ckpt_dir = os.path.join(args.root,"saved_models/",args.load_pt)
    else:
        ckpt_dir = os.path.join(args.root,"saved_models/",f"{args.model}_best_model.pth")
    try:
        checkpoint = torch.load(ckpt_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loading model from {ckpt_dir}'")
    except Exception as e:
        print(f"❌ Can't load model from {ckpt_dir}, retype the correct relative path")
        raise RuntimeError(f"Model loading failed from {ckpt_dir}") from e

    data_module = OxfordPetData(root_dir=args.root, batch_size=args.batch, num_workers=8)
    _, _, test_loader = data_module.get_dataloaders()
    print("####Testing####")
    dice = evaluate(device=device, model=model, test_loader=test_loader, args = args)
    print(f"Dice score : {dice:.4f}")