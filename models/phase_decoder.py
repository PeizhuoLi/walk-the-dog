import torch
import numpy as np
from modules import NormalizedMLP, NormalizedCNN
import torch.nn as nn


def create_model_from_args(args, dataset):
    sep_dim = dataset.feature_dims[0]

    std_in = dataset.data_std[:sep_dim]
    mean_in = dataset.data_mean[:sep_dim]
    std_out = dataset.data_std[sep_dim:]
    mean_out = dataset.data_mean[sep_dim:]

    model = NormalizedMLP(std_in, mean_in, std_out, mean_out, n_layers=args.n_layers, activation=args.activation)
    return model


def create_cnn_model_from_args(args, dataset):
    sep_dim = dataset.feature_dims[0]

    std_in = dataset.data_std[:sep_dim][..., None]
    mean_in = dataset.data_mean[:sep_dim][..., None]
    std_out = dataset.data_std[sep_dim:][..., None]
    mean_out = dataset.data_mean[sep_dim:][..., None]

    model = NormalizedCNN(std_in, mean_in, std_out, mean_out, n_layers=args.n_layers, kernel_size=args.kernel_size,
                          activation=args.activation, use_down_up=args.use_down_up)
    return model


def create_model_from_args2(args, dataset):
    std_out = dataset.get_feature_by_names(dataset.data_std[:, None], args.needed_channel_names_phase_decoder)[:, 0]
    mean_out = dataset.get_feature_by_names(dataset.data_mean[:, None], args.needed_channel_names_phase_decoder)[:, 0]
    std_in = np.ones(args.n_latent_channel)
    mean_in = np.zeros(args.n_latent_channel)
    model = NormalizedMLP(std_in, mean_in, std_out, mean_out, n_layers=args.n_layers_phase_decoder, activation=args.activation_phase_decoder)
    return model


class NamedOutputModel(nn.Module):
    def __init__(self, model, feature_dims):
        super().__init__()
        self.model = model
        self.feature_dims = feature_dims

    def forward(self, x):
        output = self.model.forward(x)
        assert output.shape[-1] == sum(self.feature_dims)
        outputs = []
        for d in self.feature_dims:
            outputs.append(output[..., :d])
            output = output[..., d:]
        return outputs


class NamedCNNModel(nn.Module):
    def __init__(self, model, feature_dims):
        super().__init__()
        self.model = model
        self.feature_dims = feature_dims

    def forward(self, x):
        output = self.model.forward(x)
        assert output.shape[-2] == sum(self.feature_dims)
        outputs = []
        for d in self.feature_dims:
            outputs.append(output[..., :d, :])
            output = output[..., d:, :]
        return outputs


def export_named_onnx(model, input_shape, filename, output_names, feature_dims, dynamic_axes=None, is_cnn=False):
    model.eval()
    dummy_input = torch.randn(*input_shape, device=list(model.parameters())[0].device)
    if is_cnn:
        named_model = NamedCNNModel(model, feature_dims)
    else:
        named_model = NamedOutputModel(model, feature_dims)
    torch.onnx.export(named_model, dummy_input, filename, verbose=False, input_names=['input'],
                      output_names=output_names, dynamic_axes=dynamic_axes)
