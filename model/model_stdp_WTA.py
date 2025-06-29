import torch
import torch.nn as nn
import snntorch as snn

class Net(nn.Module):
    def __init__(
            self, 
            num_input: int,
            num_output: int,
            beta: float,
            threshold: float,
            reset_mechanism: str,
            seed: int | None = None
        ):
        super().__init__()
        '''
        Initialize a single-layer SNN.

        Parameters:
            num_input: Number of input features
            num_output: Number of output neurons 
            beta: Membrane potential decay rate
            threshold: Neuron firing threshold
            reset_mechanism: Neuron reset mechanism ("subtract", "zero", "none")
            seed: Random seed for reproducibility | optional
        '''

        self.fc1 = nn.Linear(in_features=num_input, out_features=num_output, bias=False)
        self.lif = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)

        self.num_output = num_output

        if seed is not None:
            torch.manual_seed(seed)

        torch.nn.init.uniform_(self.fc1.weight, a=0.0, b=1.0)

    def forward(self, x):
        '''
        Forward pass through SNN.

        Parameters:
            x: Input image with the shape [timesteps x input_features]

            For example, given a [channel x height x width] image. Each image 
            must be flattened to [channel x input_features], where input_features
            is the number of pixels, or height x width. Then Use spikegen.latency 
            to convert the image into the shape [timesteps x channel x input_features].
            Remove the channel dimension to have a [timesteps x input_features] formatted
            time series which will serve as your input into the model.
        '''

        timesteps = x.shape[0]

        mem = self.lif.init_leaky()
        spk_rec = x.new_zeros((timesteps, self.num_output))
        mem_rec = x.new_zeros((timesteps, self.num_output))

        for step in range(timesteps):
            cur = self.fc1(x[step])
            spk, mem = self.lif(cur, mem)

            spike_idx = torch.nonzero(spk).flatten()

            if spike_idx.numel() > 0:

                winner = spike_idx[torch.randint(0, spike_idx.numel(), (1,)).item()]

                new_spk = torch.zeros_like(spk)
                new_spk[winner] = 1.0

                spk_rec[step] = new_spk
                mem_rec[step] = mem

                break

            spk_rec[step] = spk
            mem_rec[step] = mem

        return spk_rec, mem_rec