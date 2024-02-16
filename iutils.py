import matplotlib.pyplot as plt
import numpy as np
import argparse
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
from collections import OrderedDict

# Dictionary containing architecture specifications
architectures = {"vgg16":25088, "densenet121":1024}

# Function to perform image transformations
def process_image(data_dir):
    # Define data directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define data transformations
    transformations = {
        'train_transforms': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test_transforms': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation_transforms': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Load datasets with transformations
    image_datasets = {
        'train_data': datasets.ImageFolder(train_dir, transform=transformations['train_transforms']),
        'test_data': datasets.ImageFolder(test_dir, transform=transformations['test_transforms']),
        'valid_data': datasets.ImageFolder(valid_dir, transform=transformations['validation_transforms'])
    }
    
    return image_datasets['train_data'], image_datasets['valid_data'], image_datasets['test_data']

# Function to load data
def load_data(data_dir):
    train_data, valid_data, test_data = process_image(data_dir)
    
    # Define data loaders
    data_loaders = {
        'train_loader': torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True),
        'test_loader': torch.utils.data.DataLoader(test_data, batch_size=32),
        'valid_loader': torch.utils.data.DataLoader(valid_data, batch_size=32)
    }
    
    return data_loaders['train_loader'], data_loaders['valid_loader'], data_loaders['test_loader']

# Function to construct the neural network
def build_network(structure='vgg16', dropout=0.5, hidden_layer1=4096, lr=0.001, device='gpu'):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Please select either 'vgg16' or 'densenet121' for the structure.")
        return None
    
    # Freeze parameters to avoid backpropagation
    for param in model.parameters():
        param.requires_grad = False
    
    # Define classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(architectures[structure], hidden_layer1)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_layer1, 1024)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(dropout)),
        ('fc3', nn.Linear(1024, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()
    
    return model, criterion, optimizer

# Function for training the neural network
def train_network(model, criterion, optimizer, epochs=3, print_every=40, train_loader=None, valid_loader=None, device='gpu'):
    steps = 0

    for e in range(epochs):
        running_loss = 0
        
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            
            if torch.cuda.is_available() and device == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0

                for inputs, labels in valid_loader:
                    if torch.cuda.is_available():
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        model.to('cuda')

                    with torch.no_grad():
                        outputs = model.forward(inputs)
                        validation_loss = criterion(outputs, labels)
                        ps = torch.exp(outputs).data
                        equality = (labels.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                validation_loss = validation_loss / len(valid_loader)
                accuracy = accuracy / len(valid_loader)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(validation_loss),
                      "Accuracy: {:.4f}".format(accuracy))

                running_loss = 0
                model.train()

# Function to save the checkpoint
def save_checkpoint(model, path='checkpoint.pth', structure='vgg16', hidden_layer1=4096, dropout=0.5, lr=0.001, epochs=3):
    model.class_to_idx = train_data.class_to_idx
    model.cpu()
    torch.save({
        'structure': structure,
        'hidden_layer1': hidden_layer1,
        'dropout': dropout,
        'lr': lr,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }, path)

# Function to load the checkpoint
def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr = checkpoint['lr']
    epochs = checkpoint['epochs']

    model, _, _ = build_network(structure, dropout, hidden_layer1, lr)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

# Function to process an image
def process_image(image_path):
    img = Image.open(image_path)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = preprocess(img)
    
    return img_tensor

# Function for making predictions
def predict(image_path, model, topk=5, device='gpu'):
    if torch.cuda.is_available() and device == 'gpu':
        model.to('cuda')
    
    img_tensor = process_image(image_path)
    img_tensor = img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.float()

    if device == 'gpu':
        with torch.no_grad():
            output = model.forward(img_tensor.cuda())
    else:
        with torch.no_grad():
            output = model.forward(img_tensor)

    probability = F.softmax(output.data, dim=1)

    return probability.topk(topk)

# Example usage:
# train_loader, valid_loader, test_loader = load_data('./flowers/')
# model, criterion, optimizer = build_network(structure='vgg16', dropout=0.5, hidden_layer1=4096, lr=0.001, device='gpu')
# train_network(model, criterion, optimizer, epochs=3, print_every=40, train_loader=train_loader, valid_loader=valid_loader, device='gpu')
# save_checkpoint(model, path='checkpoint.pth', structure='vgg16', hidden_layer1=4096, dropout=0.5, lr=0.001, epochs=3)
# loaded_model = load_checkpoint(path='checkpoint.pth')
# probs, classes = predict(image_path='/home/downloads/manu/flowers/test/1/image_06752.jpg', model=loaded_model, topk=5, device='gpu')
