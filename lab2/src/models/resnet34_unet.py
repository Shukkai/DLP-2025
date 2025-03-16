import torch
from oxford_pet import OxfordPetData


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Resnet34():
    pass