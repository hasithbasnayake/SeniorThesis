import torch
import numpy as np 
import matplotlib.pyplot as plt 

def viz_model(model, epoch, save_path):

    W = model.fc1.weight.detach().cpu().numpy()
    num_neurons = W.shape[0]

    cols = int(np.ceil(np.sqrt(num_neurons)))
    rows = int(np.ceil(num_neurons / cols))
    
    def plot_grid(maps, title, fname):
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
        axes = axes.flatten()
        for i in range(num_neurons):
            ax = axes[i]
            # each map is 28×28
            ax.imshow(maps[i].reshape(28,28), cmap="gray") # Turned off clamping
            ax.set_title(f"{title} {i}", fontsize=6)
            ax.axis("off")
        # turn off extras
        for ax in axes[num_neurons:]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"{save_path}/{num_neurons}_Neurons_Network_{title.lower()}--{epoch}_Epoch.png", dpi=200)
        plt.close()
    
    # build ON, OFF, and RF arrays
    on_maps  = W[:, :784]
    off_maps = W[:, 784:]
    rf_maps  = on_maps - off_maps
    
    # plot each
    plot_grid(on_maps,  "ON_RF",  "on_fields")
    plot_grid(off_maps, "OFF_RF", "off_fields")
    plot_grid(rf_maps,  "ACTUAL_RF", "actual_fields")