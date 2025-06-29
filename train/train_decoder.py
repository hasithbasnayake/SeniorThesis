import torch
import numpy as np 
import matplotlib.pyplot as plt 
from joblib import dump

from model.spikegen import *
from model.model_stdp import *
from hyperparameters.hyperparameters_stdp import *
from sklearn.linear_model import LinearRegression

def train_decoder(wta_model, net_size, train_loader, num_steps, epoch, save_path):

    num_output = wta_model.fc1.weight.shape[0]
    num_input = wta_model.fc1.weight.shape[1]
    train_set_size = len(train_loader)

    model = Net(num_input=num_input,
                num_output=net_size,
                beta=beta,
                threshold=threshold,
                reset_mechanism=reset_mechanism)
    
    state_dict = torch.load(f"{save_path}/{num_output}_Neurons_Network--{epoch}_Epoch.pt", weights_only=False)
    model.load_state_dict(state_dict)


    X = []
    Y = []

    for idx, (image, label) in enumerate(train_loader):

        print(f"DECODER: Image {idx} / {train_set_size}", end='\r', flush=True)

        conv_image = image[:, 0:2, :, :]
        org_image = image[:, 2:3, :, :]

        conv_image = conv_image.squeeze()
        conv_image = torch.flatten(conv_image, start_dim=0)

        spike_conv_image = spikegen(image=conv_image, num_steps=num_steps)

        spk_rec, mem_rec = model(spike_conv_image)

        values, indices = spk_rec.max(dim=0)

        first_spk_rec = torch.where(values == 0, 255, indices)

        X.append(first_spk_rec)
        Y.append(org_image.squeeze().view(-1))

    X = torch.stack(X).detach().numpy()
    Y = torch.stack(Y).detach().numpy()

    decoder = LinearRegression()
    decoder.fit(X,Y)

    dump(decoder, f"{save_path}/{num_output}_Decoder--{epoch}_Epoch.joblib")

    return decoder

