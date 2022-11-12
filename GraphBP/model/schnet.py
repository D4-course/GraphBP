"""
Code adapted from
https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/schnet.html#SchNet
"""

import torch
from torch.nn import Embedding, ModuleList, Sequential, Linear
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, MessagePassing
from math import pi as PI


class GaussianSmearing(torch.nn.Module):
    """
    Neural network class for Gaussian Smearing
    """
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        """
        Forward propagation function for Gaussian Smearing
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    """
    Neural network class for Shifted Softplus
    """
    def __init__(self):    
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        """
        Forward propagation function for Shifted Softplus
        """
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    """
    Class for CF Conv
    """
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        """
        Function to reset the parameters of CF Conv
        """
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        """
        Function for forward propagation of CF Conv
        """
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        """
        Message function for CF Conv
        """
        return x_j * W


class InteractionBlock(torch.nn.Module):
    """
    Neural Network Class for Interaction Block
    """
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(
            hidden_channels,
            hidden_channels,
            num_filters,
            self.mlp,
            cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Function to reset parameters for Interaction Block
        """
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        """
        Forward propagation for Interaction Block
        """
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNet(torch.nn.Module):
    """
    Neural Network class for SchNet
    """
    def __init__(
            self,
            num_node_types,
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0):
        super(SchNet, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff

        self.embedding = Embedding(num_node_types, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

#         self.lin1 = Linear(hidden_channels, hidden_channels // 2)
#         self.act = ShiftedSoftplus()
#         self.lin2 = Linear(hidden_channels // 2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Function to reset SchNet parameters
        """
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
#         torch.nn.init.xavier_uniform_(self.lin1.weight)
#         self.lin1.bias.data.fill_(0)
#         torch.nn.init.xavier_uniform_(self.lin2.weight)
#         self.lin2.bias.data.fill_(0)

    def forward(self, z, pos, batch):
        """
        Forward propagation function for SchNet
        """
        h = self.embedding(z)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

#         h = self.lin1(h)
#         h = self.act(h)
#         h = self.lin2(h)

        return h
