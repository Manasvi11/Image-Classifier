import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import iutils

pp = argparse.ArgumentParser(description='Train.py')


pp.add_argument('data_dir', action="store", default="./flowers/")
pp.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)
pp.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
pp.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
pp.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
pp.add_argument('--epochs', dest="epochs", action="store", type=int, default=2)
pp.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
pp.add_argument('--gpu', dest="gpu", action="store", default="gpu")


arg = pp.parse_args()
root = arg.data_dir
path = arg.save_dir
lr = arg.learning_rate
structure = arg.arch
dropout = arg.dropout
hidden_layer1 = arg.hidden_units
device = arg.gpu
epochs = arg.epochs

def main():
    

    trainloader, v_loader, testloader = iutils.load_data(root)
    model, optimizer, criterion = iutils.network_construct(structure,dropout,hidden_layer1,lr,device)
    iutils.do_deep_learning(model, optimizer, criterion, epochs, 20, trainloader, device)
    iutils.save_checkpoint(model,path,structure,hidden_layer1,dropout,lr)
    print("Training Completed !")


if __name__== "__main__":
    main()

