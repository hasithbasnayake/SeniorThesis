# Modified train decoder function to train a decoder based off a backprop SNN 

import torch
import numpy as np 
import matplotlib.pyplot as plt 
from torchvision import datasets, transforms
from joblib import dump

from model.spikegen import *
from hyperparameters.hyperparameters_backprop import *
from sklearn.linear_model import LinearRegression


transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
            dog_transform])

train_dataset = datasets.FashionMNIST('assets/FashionMNIST', train=True, transform=transform, download=True)

train_set = torch.utils.data.Subset(train_dataset, range(10))

decoder_train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, drop_last=True) 

def train_decoder_backprop(backprop_model, num_steps, epoch, save_path):

    # We don't really need to modify the actual backprop model at all 

    num_output = backprop_model.fc1.weight.shape[0]

    X = []
    Y = []

    for idx, (image, label) in enumerate(decoder_train_loader):

        print(f"TRAINING DECODER: Image {idx} / 1000", end='\r', flush=True)

        conv_image = image[:, 0:2, :, :]
        org_image = image[:, 2:3, :, :]

        _, _, spk_rec, mem_rec = backprop_model(conv_image.view(1, -1))

        values, indices = spk_rec.max(dim=0)

        first_spk_rec = torch.where(values == 0, 255, indices)

        X.append(first_spk_rec)
        Y.append(org_image.squeeze().view(-1))

    X = torch.stack(X).squeeze(1).detach().numpy()
    Y = torch.stack(Y).detach().numpy()

    print(X.shape)
    print(Y.shape)

    decoder = LinearRegression()
    decoder.fit(X,Y)

    dump(decoder, f"{save_path}/{num_output}_Decoder--{epoch}_Epoch.joblib")

    return decoder
