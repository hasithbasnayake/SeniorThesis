import torch
import torch.nn as nn
import snntorch as snn

class Net(nn.Module):
    def __init__(
            self, 
            num_input: int,
            num_hidden: int,
            num_output: int,
            beta: float,
            threshold: float,
            reset_mechanism: str,
            num_steps: int,
            seed: int | None = None
        ):
        super().__init__()
        '''
        Initialize a multi-layer SNN to be trained using surrogate backpropagation.
        The network has a single hidden layer and an output layer

        Parameters:
            num_input: Number of input features
            num_output: Number of output neurons 
            beta: Membrane potential decay rate
            threshold: Neuron firing threshold
            reset_mechanism: Neuron reset mechanism ("subtract", "zero", "none")
            seed: Random seed for reproducibility | optional
        '''

        self.fc1 = nn.Linear(in_features=num_input, out_features=num_hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
        
        self.fc2 = nn.Linear(in_features=num_hidden, out_features=num_output)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)

        self.num_steps = num_steps

        # Uniform weight initialization breaks the surrogate gradient 
    
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

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        num_steps = self.num_steps

        spk1_rec = []
        mem1_rec = []

        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec), torch.stack(spk1_rec), torch.stack(mem1_rec)
