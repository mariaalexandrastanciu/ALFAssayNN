# Created by alexandra at 28/07/2023
from torch import nn
import math
import torch.nn.functional as F
import torch as t
import utils as u


class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim_amino_acid, input_dim_chain, output_dim, hidden_layers, dev="cpu",
                 name="FragmentationPatterns", batch_size=1):
        super(NeuralNetworkModel, self).__init__()
        self.dev = dev
        self.name = name
        self.batch_size = batch_size
        self.input_dim = input_dim_amino_acid
        self.input_dim_chain = input_dim_chain
        self.output_dim = output_dim
        self.hidden_layer = hidden_layers

        # initialize alphabet to random values between -pi and pi
        # u = torch.distributions.Uniform(-3.14, 3.14)
        # self.alphabet = nn.Parameter(u.rsample(torch.Size([80, 8])))

        self.forward_fragment_counts = nn.Sequential(nn.Linear(2, 39),  nn.Tanh(),
                                                     nn.Linear(39, 5),  nn.Tanh(),
                                                     # nn.Linear(10, 10), nn.Tanh(),
                                                     # nn.Linear(100, 10), nn.Tanh(),
                                               # nn.Linear(150, 150), nn.BatchNorm1d(6), nn.Dropout(0.1), nn.ReLU(),
                                               # nn.Linear(150, 150), nn.BatchNorm1d(6), nn.Dropout(0.1), nn.ReLU(),
                                               # nn.Linear(150, 150), nn.BatchNorm1d(6), nn.Dropout(0.1), nn.ReLU(),
                                               #       nn.Linear(10, 10),  nn.Tanh(),
                                                     nn.Linear(5, 1)  ).to(dev)

        # self.forward_chain = nn.Sequential(nn.Linear(self.input_dim_chain*10, 140), nn.LayerNorm(140),  nn.Tanh(),
        #                                    nn.Linear(140, 140), nn.LayerNorm(140),  nn.Tanh(),
        #                                    # nn.Linear(360, 360), nn.Dropout(0.6), nn.ReLU(),
        #                                    # nn.Linear(60, 60), nn.Dropout(0.1), nn.ReLU(),
        #                                    # nn.Linear(60, 60), nn.Dropout(0.1), nn.ReLU(),
        #                                    # nn.Linear(60, 60), nn.Dropout(0.1), nn.ReLU(),
        #                                    nn.Linear(140, self.input_dim_chain*self.output_dim*2),
        #                                    nn.LayerNorm(self.input_dim_chain*self.output_dim*2),  nn.Tanh()
        #                                    ).to(dev)

        # self.constraint = nn.Sequential(nn.Linear(10, 1), nn.LayerNorm(1))

    def forward(self, fragment_length_counts):

        # tensor_sequence = u.get_hashed_sequence(input_sequence=sequence).transpose(0, 1)
        # tensor_sequence = sequence.transpose(0, 1)
        # embedded_sequence = self.embeds(tensor_sequence)

        #
        # eps = 1e-6
        first_step = self.forward_fragment_counts(fragment_length_counts) #.view(self.output_dim) + eps #
        # label_per_arm = u.sigmoid(first_step).view(self.output_dim)
        label_per_arm = t.sigmoid(first_step).squeeze() #.view(self.output_dim)

        # concatenated_seq = intermmediate_step.view(self.batch_size, self.input_dim_chain*10)
        # 
        # angles = self.forward_chain(concatenated_seq).view(self.batch_size, self.input_dim_chain, self.output_dim*2)

        #print("angles:", angles)

        # label_per_arm = F.softmax(self.constraint(first_step), dim=1).view(self.output_dim) + eps

        # label_per_arm = F.softmax(first_step, dim=1).view(self.output_dim)
        #print("constraint_angles:", constraint_angles)
        # sine = torch.matmul(constraint_angles, torch.sin(self.alphabet))
        # cosine = torch.matmul(constraint_angles, torch.cos(self.alphabet))
        # predicted_angles = torch.atan2(sine, cosine)

        # return predicted_angles.view(self.input_dim_chain, self.batch_size, self.output_dim)
        return label_per_arm


    def save_grad(self, name):  # stuff you need to plot. leave it there
        def hook(grad): self.grads[name] = grad

        return hook

