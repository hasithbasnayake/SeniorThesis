import torch
import numpy as np 
import matplotlib.pyplot as plt 

from model.model_stdp_WTA import *

from model.spikegen import *
from learning.stdp import *

def train_model(model, update_params, train_loader, num_steps, epoch, save_path):
    '''
    train_model trains a SNN for one epoch, or one full pass over the training
    set. It returns the model and saves an image of the model weights. 
    
    '''
    num_output = model.fc1.weight.shape[0]
    num_input = model.fc1.weight.shape[1]
    train_set_size = len(train_loader)

    for idx, (image, label) in enumerate(train_loader):

        print(f"TRAINING: Image {idx} / {train_set_size}", end='\r', flush=True)

        conv_image = image[:, 0:2, :, :]
        org_image = image[:, 2:3, :, :]

        conv_image = conv_image.squeeze()
        conv_image = torch.flatten(conv_image, start_dim=0)

        spike_conv_image = spikegen(image=conv_image, num_steps=num_steps)

        spk_rec, mem_rec = model(spike_conv_image)

        out_neuron, delta_w = stdp_time(weight_matrix=model.fc1.weight,
                                        in_spike=spike_conv_image,
                                        out_spike=spk_rec,
                                        params=update_params)
        
        with torch.no_grad():

            model.fc1.weight[out_neuron] += delta_w
            model.fc1.weight[out_neuron].clamp_(0.0, 1.0)

    torch.save(model.state_dict(), f"{save_path}/{num_output}_Neurons_Network--{epoch}_Epoch.pt")
    
    return None 

def viz_model(model, epoch, save_path):

    W = model.fc1.weight.detach().cpu().numpy()
    num_neurons = W.shape[0]
    num_input = W.shape[1]
    
    cols = int(np.ceil(np.sqrt(num_neurons)))
    rows = int(np.ceil(num_neurons / cols))
    
    def plot_grid(maps, title, fname):
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
        axes = axes.flatten()
        for i in range(num_neurons):
            ax = axes[i]
            # each map is 28×28
            ax.imshow(maps[i].reshape(28,28), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"{title} {i}", fontsize=6)
            ax.axis("off")
        # turn off extras
        for ax in axes[num_neurons:]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"{save_path}/{num_neurons}_Neurons_Network_{title.lower()}--{epoch}_Epoch.png", dpi=100)
        plt.close()
    
    # build ON, OFF, and RF arrays
    on_maps  = W[:, :784]
    off_maps = W[:, 784:]
    rf_maps  = on_maps - off_maps
    
    # plot each
    plot_grid(on_maps,  "ON_RF",  "on_fields")
    plot_grid(off_maps, "OFF_RF", "off_fields")
    plot_grid(rf_maps,  "ACTUAL_RF", "actual_fields")