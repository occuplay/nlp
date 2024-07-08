import torch
from model_save import *
import torchvision
from torch import nn

model = torch.load("vgg16_method1.pth")
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))

model = torch.load('tudui_method1.pth')
print(model)