import torch
import os

from utils import parse_arguments
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate
from oxford_pet import OxfordPetData

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    args = parse_arguments()

    if args.model == "unet":
        print("Using unet")
        model = UNet(3,1).to(device)
    else:
        print("Using resnet34_unet")
        model = ResNet34_UNet(3,1).to(device)
    if args.load_pt != "":
        ckpt_dir = os.path.join(args.root,"saved_models/",args.load_pt)
    else:
        ckpt_dir = os.path.join(args.root,"saved_models/",f"{args.model}_best_model.pt")
    try:
        checkpoint = torch.load(ckpt_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loading model from {ckpt_dir}'")
    except:
        print(f"❌ Can't load model from {ckpt_dir}, retype correct relative path")

    data_module = OxfordPetData(root_dir=args.root, batch_size=args.batch, num_workers=8)
    _, _, test_loader = data_module.get_dataloaders()
    print("####Testing####")
    dice = evaluate(model=model, test_loader=test_loader)
    print(f"Dice score : {dice:.4f}")