import torch

def spikegen(image, num_steps):

    spike_rec = torch.zeros([num_steps, image.shape[0]])

    norm_latency = latency(image, num_steps)

    for index, pixel_latency in enumerate(norm_latency):
        spike_rec[pixel_latency, index] = 1

    return spike_rec

def latency(image, num_steps):

    latency = 1.0 / image
    
    finite_vals = latency[torch.isfinite(latency)]
    min_val = torch.min(finite_vals)
    max_val = torch.max(finite_vals)
    
    norm_latency = (latency - min_val) / (max_val - min_val) * (num_steps - 1)
    
    norm_latency[~torch.isfinite(norm_latency)] = num_steps - 1

    norm_latency = norm_latency.int()
    
    return norm_latency