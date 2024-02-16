import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import iutils

pp = argparse.ArgumentParser(description='Predict.py')

pp.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str)
pp.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
pp.add_argument('--checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
pp.add_argument('--top_k', default=4, dest="top_k", action="store", type=int)
pp.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
pp.add_argument('--gpu', default="gpu", action="store", dest="gpu")

arg = pp.parse_args()
path_image = arg.input
number_of_outputs = arg.top_k
device = arg.gpu

path = arg.checkpoint

arg = pp.parse_args()

def main():
    model=iutils.load_checkpoint(path)
    with open('cat_to_name.json', 'r') as json_file:
        cat_name = json.load(json_file)
    probabilities = iutils.predict(path_image, model, number_of_outputs, device)
    labels = [cat_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability = np.array(probabilities[0][0])
    i=0
    while i < number_of_outputs:
        print("{} with probability of {}".format(labels[i], probability[i]))
        i += 1
    print("Done!")

    
if __name__== "__main__":
    main()