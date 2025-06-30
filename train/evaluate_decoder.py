import torch
import numpy as np 
import math
import matplotlib.pyplot as plt 
from joblib import dump

from model.spikegen import *
from model.model_stdp_WTA import *

from sklearn.linear_model import LinearRegression
from hyperparameters.hyperparameters_stdp import *
from skimage.metrics import mean_squared_error, structural_similarity

# Decoder Evaluation 

def evaluate_decoder(wta_model, net_size, test_loader, decoder, num_steps, epoch, save_path):

    num_output = wta_model.fc1.weight.shape[0]
    num_input = wta_model.fc1.weight.shape[1]
    test_set_size = len(test_loader)

    model = Net(num_input=num_input,
                num_output=net_size,
                beta=beta,
                threshold=threshold,
                reset_mechanism=reset_mechanism)

    state_dict = torch.load(f"{save_path}/{num_output}_Neurons_Network--{epoch}_Epoch.pt", weights_only=False)
    model.load_state_dict(state_dict)


    orig_images = []
    recon_images = [] 

    for idx, (image, label) in enumerate(test_loader):

        print(f"EVALUATING MODEL: Image {idx} / {test_set_size}", end='\r', flush=True)

        conv_image = image[:, 0:2, :, :]
        org_image = image[:, 2:3, :, :]

        conv_image = conv_image.squeeze()

        conv_image = torch.flatten(conv_image, start_dim=0)

        spike_image = spikegen(image = conv_image, num_steps = num_steps)

        spk_rec, mem_rec = model(spike_image)

        values, indices = spk_rec.max(dim=0)

        first_spk_rec = torch.where(values == 0, 255, indices)

        spk_image = first_spk_rec

        recon_img = decoder.predict(spk_image.unsqueeze(0).detach().numpy())
        recon_img = recon_img.reshape(28,28)

        orig_images.append(org_image.squeeze().numpy())
        recon_images.append(recon_img)

    mses = []
    ssims = []

    for orig, recon in zip(orig_images, recon_images):
        mses.append(mean_squared_error(orig, recon))
        ssims.append(structural_similarity(orig, recon, data_range=1.0))

    avg_ssim = np.mean(ssims)
    avg_mse = np.mean(mses)

    plot_reconstructions(orig_images, recon_images,
                         save_path, num_output, epoch)

    return avg_ssim, avg_mse


def plot_reconstructions(orig_images, recon_images,
                         save_path, num_output, epoch,
                         max_pairs=100, pairs_per_row=10):
    """
    Plots up to max_pairs original/reconstructed image pairs side by side.

    Args:
      orig_images    : list or array of 2D arrays (orig)
      recon_images   : list or array of 2D arrays (recon)
      save_path      : directory to save the figure
      num_output     : number of neurons (for filename)
      epoch          : current epoch (for filename)
      max_pairs      : how many pairs to plot (default 100)
      pairs_per_row  : how many pairs in each grid row
    """
    n = min(max_pairs, len(orig_images))
    rows = math.ceil(n / pairs_per_row)
    cols = pairs_per_row * 2  # two columns per pair

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 1.5, rows * 1.5),
                             squeeze=False)

    for i in range(n):
        r = i // pairs_per_row
        c0 = (i % pairs_per_row) * 2

        ax_o = axes[r][c0]
        ax_r = axes[r][c0 + 1]

        ax_o.imshow(orig_images[i], cmap="gray", vmin=0.0, vmax=1.0)
        ax_o.axis("off")
        if c0 == 0: 
            ax_o.set_ylabel("Orig", fontsize=6)

        ax_r.imshow(recon_images[i], cmap="gray", vmin=0.0, vmax=1.0)
        ax_r.axis("off")
        if c0 == 0:
            ax_r.set_ylabel("Recon", fontsize=6)

    # turn off any leftover axes
    for j in range(n, rows * pairs_per_row):
        r = j // pairs_per_row
        c0 = (j % pairs_per_row) * 2
        axes[r][c0].axis("off")
        axes[r][c0 + 1].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_path}/{num_output}_Neurons_Network_Recon--{epoch}_Epoch.png", dpi=200)
    plt.close()
