import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import Library.Utility as utility


class MLPChannels(nn.Module):
    def __init__(self, n_channels, bn):
        super().__init__()
        self.layers = []
        self.n_channels = n_channels
        for i in range(len(n_channels) - 1):
            self.layers.append(nn.Linear(n_channels[i], n_channels[i + 1]))
            if i != len(n_channels) - 2:
                if bn:
                    self.layers.append(nn.BatchNorm1d(n_channels[i + 1]))
                self.layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class MLP(nn.Module):
    def __init__(self, n_layers, n_channel_in, n_channel_out, n_phase_channel, bn=False, last_activation=False):
        super().__init__()
        self.layers = []
        self.need_add_one = last_activation
        for i in range(n_layers):
            n_out = n_channel_out if i == n_layers - 1 else n_channel_in
            self.layers.append(nn.Linear(n_channel_in, n_out))
            if i != n_layers - 1:
                if bn:
                    self.layers.append(nn.BatchNorm1d(n_phase_channel))
                self.layers.append(nn.LeakyReLU(negative_slope=0.2))
            if last_activation:
                self.layers.append(nn.ELU())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        if self.need_add_one:
            x = x + 1
        return x


class NormalizedMLP(nn.Module):
    def __init__(self, std_in, mean_in, std_out, mean_out, n_layers=5, activation='ELU', shape_change='first'):
        super().__init__()
        std_in = torch.from_numpy(std_in).float()
        mean_in = torch.from_numpy(mean_in).float()
        std_out = torch.from_numpy(std_out).float()
        mean_out = torch.from_numpy(mean_out).float()

        if shape_change == 'first':
            n_channels = [std_in.shape[-1]] + [std_out.shape[-1]] * n_layers
        elif shape_change == 'last':
            n_channels = [std_in.shape[-1]] * n_layers + [std_out.shape[-1]]
        else:
            raise Exception("Unknown shape change type")
        if activation == 'ELU':
            activation_layer = nn.ELU()
        elif activation == 'LeakyReLU':
            activation_layer = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise Exception('Unknown activation function')

        self.layers = []
        for i in range(n_layers):
            self.layers.append(nn.Linear(n_channels[i], n_channels[i + 1]))
            if i != n_layers - 1:
                self.layers.append(activation_layer)

        self.layers = nn.Sequential(*self.layers)
        self.std_in = nn.Parameter(std_in, requires_grad=False)
        self.mean_in = nn.Parameter(mean_in, requires_grad=False)
        self.std_out = nn.Parameter(std_out, requires_grad=False)
        self.mean_out = nn.Parameter(mean_out, requires_grad=False)

    def forward(self, x):
        if not self.training:
            x = (x - self.mean_in) / self.std_in
        x = self.layers(x)
        if not self.training:
            x = x * self.std_out + self.mean_out
        return x


class NormalizedCNN(nn.Module):
    def __init__(self, std_in, mean_in, std_out, mean_out, n_layers=5, kernel_size=3, activation='ELU',
                 use_down_up=False):
        super().__init__()
        std_in = torch.from_numpy(std_in).float()
        mean_in = torch.from_numpy(mean_in).float()
        std_out = torch.from_numpy(std_out).float()
        mean_out = torch.from_numpy(mean_out).float()

        n_channels = [std_in.shape[-2]] + [std_out.shape[-2]] * n_layers
        if activation == 'ELU':
            activation_layer = nn.ELU()
        elif activation == 'LeakyReLU':
            activation_layer = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise Exception('Unknown activation function')

        self.layers = []
        padding = (kernel_size - 1) // 2
        for i in range(n_layers):
            if use_down_up:
                if 2 * i >= n_layers:
                    stride = 1
                    self.layers.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=False))
                else:
                    stride = 2
            else:
                stride = 1
            self.layers.append(nn.Conv1d(n_channels[i], n_channels[i + 1], kernel_size,
                                         padding=padding, padding_mode='reflect', stride=stride))
            if i != n_layers - 1:
                self.layers.append(activation_layer)

        self.layers = nn.Sequential(*self.layers)
        self.std_in = nn.Parameter(std_in, requires_grad=False)
        self.mean_in = nn.Parameter(mean_in, requires_grad=False)
        self.std_out = nn.Parameter(std_out, requires_grad=False)
        self.mean_out = nn.Parameter(mean_out, requires_grad=False)

    def forward(self, x):
        if not self.training:
            x = (x - self.mean_in) / self.std_in
        x = self.layers(x)
        if not self.training:
            x = x * self.std_out + self.mean_out
        return x


def save_manifold(network, motion_data, filename, args):
    E = np.arange(len(motion_data))
    batch_size = args.batch_size * 16
    loop = tqdm(range(0, len(motion_data), batch_size))
    from models import VQ

    manifolds = []
    manifolds_ori = []
    states = []
    states_ori = []
    phases = []
    indices = []
    frequencies = []
    for i in loop:
        eval_indices = E[i:i + batch_size]
        eval_batch = motion_data.load_batches(eval_indices)[..., :motion_data.frames_per_window]
        eval_batch = motion_data.get_feature_by_names(eval_batch, args.needed_channel_names)
        eval_batch = utility.ToDevice(eval_batch)
        output = network(eval_batch)
        params = output[3]
        info = output[4]
        p = utility.Item(params[0])
        f = utility.Item(params[1])
        state = utility.Item(info[2])
        state_ori = utility.Item(info[4])
        index = utility.Item(info[3])
        manifold = VQ.get_phase_manifold(state, 2 * np.pi * p.unsqueeze(-1))[0].squeeze(-1)
        manifold_ori = VQ.get_phase_manifold(state_ori, 2 * np.pi * p.unsqueeze(-1))[0].squeeze(-1)

        states.append(state)
        states_ori.append(state_ori)
        phases.append(p)
        frequencies.append(f)
        indices.append(index)
        manifolds.append(manifold)
        manifolds_ori.append(manifold_ori)
        if args.debug:
            break
    states = torch.cat(states, dim=0)
    states_ori = torch.cat(states_ori, dim=0)
    phases = torch.cat(phases, dim=0)
    indices = torch.cat(indices, dim=0)
    manifolds = torch.cat(manifolds, dim=0)
    manifolds_ori = torch.cat(manifolds_ori, dim=0)
    frequencies = torch.cat(frequencies, dim=0)

    np.savez(filename, manifold=manifolds.numpy(), state=states.numpy(), phase=phases.numpy(), index=indices.numpy(),
             state_ori=states_ori.numpy(), manifold_ori=manifolds_ori.numpy(), frequency=frequencies.numpy())
    return {'manifold': manifolds, 'state': states, 'phase': phases, 'index': indices, 'state_ori': states_ori,
            'manifold_ori': manifolds_ori, 'frequency': frequencies}
