
import torch 
import pandas as pd
from torchvision import datasets, transforms
import torch.optim as optim

from hyperparameters.hyperparameters_backprop import *
from model.model_backprop import *

from train.train_model_backprop import *
from train.evaluate_decoder_backprop import *
from train.train_decoder_backprop import *

# Datasets

transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
            dog_transform])

train_dataset = datasets.FashionMNIST('assets/FashionMNIST', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST('assets/FashionMNIST', train=False, transform=transform, download=True)

train_set = torch.utils.data.Subset(train_dataset, range(num_samples))
test_set = torch.utils.data.Subset(test_dataset, range(num_samples))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True) 
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

# Helpers

def print_batch_accuracy(data, targets, train=False):
    output, _, _, _= model(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")


# ── compute full‐set accuracies ──
def compute_accuracy(loader):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for x, y in loader:
            x = x[:, 0:2, :, :]
            x = x.view(x.size(0), -1)
            out, _, _, _ = model(x)
            _, preds = out.sum(dim=0).max(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return 100 * correct / total


####################### Surrogate Backpropagation (50, 100, 200, 400) ######################

metrics_dict = {
    "50": [[],[]],
    "100": [[],[]],
    "200": [[],[]],
    "400": [[],[]],
}


for net_size in num_hidden:

    save_path = f"data/{net_size}_neuron_backprop"
    
    model = Net(num_input=num_input,
                num_hidden=net_size,
                num_output=num_output,
                beta=beta,
                threshold=threshold,
                reset_mechanism=reset_mechanism,
                num_steps=num_steps)
    
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    loss_hist = []
    test_loss_hist = []
    counter = 0

    for epoch in range(num_epochs):

        train_start = len(loss_hist)
        test_start  = len(test_loss_hist)

        iter_counter = 0
        train_batch = iter(train_loader)

        # Minibatch training loop
        for data, targets in train_batch:
            data = data
            data = data[:, 0:2, :, :]
            targets = targets

            # forward pass
            model.train()
            spk_rec, mem_rec, _, _ = model(data.view(batch_size, -1))

            # initialize the loss & sum over time
            loss_val = torch.zeros((1))
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            with torch.no_grad():
                model.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data
                test_data = test_data[:, 0:2, :, :]
                test_targets = test_targets

                # Test set forward pass
                test_spk, test_mem, _, _ = model(test_data.view(batch_size, -1))

                # Test set loss
                test_loss = torch.zeros((1))
                for step in range(num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())

                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    train_printer()
                counter += 1
                iter_counter +=1

        viz_model(model=model, # We need to make sure we're pulling the right layer
                  epoch=epoch,
                  save_path=save_path)

        # ── end of epoch: compute averages ──
        epoch_train_losses = loss_hist[train_start : len(loss_hist)]
        epoch_test_losses  = test_loss_hist[test_start : len(test_loss_hist)]
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        avg_test_loss  = sum(epoch_test_losses)  / len(epoch_test_losses)

        train_acc = compute_accuracy(train_loader)
        test_acc  = compute_accuracy(test_loader)

        # ── print epoch summary ──
        print(f"=== Net hidden={net_size} | Epoch {epoch+1}/{num_epochs} ===")
        print(f" Avg Train Loss: {avg_train_loss:.4f}    Train Acc: {train_acc:.2f}%")
        print(f" Avg Test  Loss: {avg_test_loss:.4f}    Test  Acc: {test_acc:.2f}%")
        print()

        decoder = train_decoder_backprop(backprop_model = model,
                               num_steps = num_steps, 
                               epoch = epoch, 
                               save_path = save_path)

        avg_ssim, avg_mse = evaluate_decoder_backprop(backprop_model = model,
                                decoder = decoder,
                                num_steps = num_steps,
                                epoch = epoch,
                                save_path = save_path)
    
        key = str(net_size)
        metrics_dict[key][0].append(avg_ssim)
        metrics_dict[key][1].append(avg_mse)

rows = []
for net_size, (ssim_list, mse_list) in metrics_dict.items():
    for epoch_idx, (ssim, mse) in enumerate(zip(ssim_list, mse_list)):
        rows.append({
            "net_size": int(net_size),
            "epoch_idx": epoch_idx,   # idx within your saved epochs
            "avg_ssim": ssim,
            "avg_mse": mse
        })

df = pd.DataFrame(rows)
df.to_csv("data/metrics.csv", index=False)
