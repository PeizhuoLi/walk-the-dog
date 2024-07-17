import torch
import numpy as np
import Library.Utility as utility
import os.path as osp
from models.phase_decoder import export_named_onnx


class PhaseDecoderTrainer:
    def __init__(self, n_input_channels, network, lambda_args, phase_model_name, output_channel_names,
                 motion_data_name, output_feature_dims, lr, is_cnn=False):
        self.n_input_channels = n_input_channels

        network = utility.ToDevice(network)
        network.train()
        self.network = network

        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.criteria = {'rec': torch.nn.MSELoss()}
        self.lambda_args = lambda_args
        self.phase_model_name = phase_model_name
        self.export_name = f'{phase_model_name}_{motion_data_name}'
        self.output_channel_names = output_channel_names
        self.output_feature_dims = output_feature_dims
        self.loss_total = torch.tensor(0.)
        self.is_cnn = is_cnn

    def forward(self, input_f, gt_f):
        # todo: Do we need normalization on the phase?
        self.losses = {}
        input_f = utility.ToDevice(input_f)
        pred_f = self.network(input_f)

        # Compute loss and train
        self.losses['rec'] = self.criteria['rec'](pred_f, gt_f)
        self.loss_total = sum([self.losses[k] * getattr(self.lambda_args, f'lambda_{k}') for k in self.losses])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def backward(self):
        self.loss_total.backward()

    def step(self):
        self.optimizer.step()

    def record_loss(self, loss_recorder, prefix=''):
        for k in self.losses:
            loss_recorder.add_scalar(f'{prefix}loss_{k}', self.losses[k].item())
        loss_recorder.add_scalar(f'{prefix}loss_total', self.loss_total.item())

        loss_descript = ' '.join([f'{k}: {v.item():.4f}' for k, v in self.losses.items()])
        loss_descript = f'total: {self.loss_total.item():.4f} ' + loss_descript
        return loss_descript

    def export_model(self, save, epoch, prefix=''):
        if epoch is not None:
            torch.save(self.network.state_dict(), osp.join(save, f'{prefix}{epoch:04d}_Channels.pt'))
        suffix = '_cnn' if self.is_cnn else ''
        middle = f'{epoch:04d}' if epoch is not None else 'final'
        save_filename = f'{self.export_name}_{middle}{suffix}.onnx'
        if self.is_cnn:
            export_named_onnx(self.network, (1, self.n_input_channels, 50),
                              osp.join(save, save_filename),
                              self.output_channel_names, self.output_feature_dims,
                              dynamic_axes={'input': {2: 'temporal_axis'}}, is_cnn=True)
        else:
            export_named_onnx(self.network, (1, self.n_input_channels),
                              osp.join(save, save_filename),
                              self.output_channel_names, self.output_feature_dims, is_cnn=False)
