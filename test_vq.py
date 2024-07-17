import os
import os.path as osp

from tqdm import tqdm

import Library.Utility as utility
from models import VQ as model
from models import phase_decoder as phase_decoder_model

import pickle

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from option import TrainVQOptionParser, TestOptionParser

from dataset import create_dataset_from_args
from modules import save_manifold
from utils.criteria_test import get_usage, get_dataset_usage, get_combinatorial_usage, get_combinatorial_dataset_usage
from sklearn.decomposition import PCA


class PositionLoss:
    def __init__(self, feature_name, feature_dims, std, mean):
        self.idx = feature_name.index('Positions')
        self.feature_dims = feature_dims
        self.std = torch.from_numpy(std).cuda()
        self.mean = torch.from_numpy(mean).cuda()

    def get(self, v):
        v = v * self.std + self.mean
        for i in range(self.idx):
            v = v[..., self.feature_dims[i]:]
        return v[..., :self.feature_dims[self.idx]].reshape(-1, 3)

    def __call__(self, x, y):
        x = self.get(x)
        y = self.get(y)
        diff = ((x - y) ** 2).sum(-1).mean()
        return diff


def write_vq(filename, embeddings, usages):
    n_steps = embeddings.shape[0] if embeddings.ndim == 3 else 1
    usages = (usages > 0).astype(np.int32)
    usages = sum([(2 ** i) * usages[i] for i in range(usages.shape[0])])
    pca = PCA(n_components=2)
    embeddings_flat = embeddings.reshape(-1, embeddings.shape[-1])
    pca.fit(embeddings_flat)
    pca_mean = pca.mean_
    pca_mat = pca.components_.T
    embeddings2d = pca.transform(embeddings_flat)
    embeddings2d = embeddings2d.reshape(n_steps, -1, 2)

    pca_mean = pca_mean[None].repeat(n_steps, 0)
    pca_mat = pca_mat[None].repeat(n_steps, 0)

    np.savez(filename, embedding2d=embeddings2d, usage=usages, embeddings=embeddings,
             pca_mean=pca_mean, pca_mat=pca_mat)


def accumulate_usage(network, VQ, motion_data, args):
    E = np.arange(len(motion_data))
    batch_size = args.batch_size * 16
    loop = tqdm(range(0, len(motion_data), batch_size))
    usage = np.zeros(VQ.num_embed, dtype=np.int32)

    for i in loop:
        eval_indices = E[i:i + batch_size]
        eval_batch = motion_data.load_batches(eval_indices)[..., :args.frames]
        eval_batch = utility.ToDevice(eval_batch)
        output = network(eval_batch)
        vq_info = output[4]
        index = vq_info[3].detach().cpu().numpy()
        np.add.at(usage, index, 1)

    return usage


def clean_vq_state_dict(state_dict):
    for key in list(state_dict.keys()):
        if not (key.startswith("embedding") or key.startswith("vqs.")):
            state_dict.pop(key)
    return state_dict


def main():
    option_parser = TrainVQOptionParser()
    test_option_parser = TestOptionParser()
    test_args = test_option_parser.parse_args()
    n_sample = 500

    to_save = {}

    if osp.exists(osp.join(test_args.save, "args.pkl")):
        args = pickle.load(open(osp.join(test_args.save, "args.pkl"), "rb"))
        args = option_parser.deserialize(args)
    else:
        with open(osp.join(test_args.save, "args.txt"), "r") as f:
            args = option_parser.text_deserialize(f.read().split())
            args = option_parser.post_process(args)

    if test_args.plot_cnt > 0:
        if osp.exists(test_args.plot_save):
            os.system(f'rm -rf {test_args.plot_save}')
        summary_writer = SummaryWriter(test_args.plot_save)

    Load = args.load
    Save = test_args.save

    motion_datas = create_dataset_from_args(args)
    #Build network model
    networks, VQ = model.create_model_from_args(args, motion_datas)

    #Load model
    ref_files = [f for f in os.listdir(Save) if f.endswith("Channels_VQ.pt")]
    ref_files.sort(key=lambda x: int(x.split('_')[0]))
    largest_epoch = ref_files[-1].split('_')[0]

    if args.train_phase_decoder:
        phase_decoders = []
        for data in motion_datas:
            phase_decoders.append(phase_decoder_model.create_model_from_args2(args, data))
            state_file_name = f'{data.name}_{int(largest_epoch) - 1:04d}_Channels.pt'
            state_dict = torch.load(osp.join(Save, state_file_name), map_location='cpu')
            phase_decoders[-1].load_state_dict(state_dict)
    else:
        phase_decoders = []

    for i in range(len(networks)):
        network = networks[i]
        target_file = f'{largest_epoch}_{i}_{args.phase_channels}Channels.pt'
        state_dict = torch.load(osp.join(Save, target_file), map_location='cpu')
        network.load_state_dict(state_dict)

    VQ_target_file = f'{largest_epoch}_{args.phase_channels}Channels_VQ.pt'
    state_dict = torch.load(osp.join(Save, VQ_target_file), map_location='cpu')
    state_dict = clean_vq_state_dict(state_dict)
    VQ.load_state_dict(state_dict, strict=False)

    for i in range(len(networks)):
        networks[i] = utility.ToDevice(networks[i])
        networks[i].eval()

    for i in range(len(phase_decoders)):
        phase_decoders[i] = utility.ToDevice(phase_decoders[i])
        phase_decoders[i].train() # So it won't perform an extra normalization

    VQ = utility.ToDevice(VQ)
    VQ.eval()

    result_dict = []
    for i, network in enumerate(networks):
        # filename = osp.join(Save, f'Parameters_{i}_final.txt')
        filename_npy = osp.join(Save, f'Manifolds_{i}_final.npz')
        # save_parameters(network, motion_datas[i], filename, args)
        result_dict.append(save_manifold(network, motion_datas[i], filename_npy, args))

        loss_function = torch.nn.MSELoss()
        rec_loss_function = PositionLoss(motion_datas[i].channel_names, motion_datas[i].feature_dims,
                                     motion_datas[i].data_std, motion_datas[i].data_mean)
        motion_data = motion_datas[i]
        data_loader = DataLoader(motion_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

        phase_decoder = phase_decoders[i] if len(phase_decoders) > 0 else None

        # Short sequence evaluation
        iterator = iter(data_loader)
        loop = tqdm(range(n_sample))
        losses = []
        losses_decoder = []
        for _ in loop:
            # Run model prediction
            train_batch = next(iterator)
            pae_input = motion_data.get_feature_by_names(train_batch, args.needed_channel_names)
            pae_input = utility.ToDevice(pae_input)
            yPred, latent, signal, params, vq_info = network(pae_input)

            if phase_decoder is not None:
                if args.decoder_before_quantization:
                    input = params[5]
                else:
                    input = params[4]
                input = input.permute(0, 2, 1)
                input = input.reshape(-1, input.shape[-1])
                output = phase_decoder(input)

                gt = motion_data.get_feature_by_names(train_batch, args.needed_channel_names_phase_decoder)
                gt = gt.permute(0, 2, 1)
                gt = gt.reshape(-1, gt.shape[-1])
                gt = utility.ToDevice(gt)
                loss_decoder = rec_loss_function(output, gt)
                losses_decoder.append(loss_decoder.item())

            # Compute loss
            loss = loss_function(yPred, pae_input)
            losses.append(loss.detach().cpu().item())

        losses = np.array(losses)
        if phase_decoder is not None:
            losses_decoder = np.array(losses_decoder)
            to_save[f'loss_decoder_mean_{i}'] = losses_decoder.mean()
        to_save[f'loss_mean_{i}'] = losses.mean()

    usages = [get_usage(d['index'], args.num_embed_vq) for d in result_dict]
    usages = np.array(usages)
    write_vq(osp.join(test_args.save, 'VQ.npz'), utility.ItemNumpy(VQ.get_weight()), usages)

    c_usages = [get_combinatorial_usage(d['index'], args.num_embed_vq) for d in result_dict]
    c_usages = np.stack(c_usages)
    used_by_both = (c_usages > 0).prod(axis=0)
    to_save['Embed usage'] = (c_usages.sum(axis=0).reshape(-1) > 0).astype(np.float32).mean()
    to_save['Overlap percentage embedding'] = used_by_both.astype(np.float32).mean()

    for i in range(len(motion_datas)):
        c_usage = get_combinatorial_dataset_usage(result_dict[i]['index'], used_by_both)
        to_save[f'Overlap percentage {motion_datas[i].name}'] = c_usage

    print(to_save)
    with open(osp.join(test_args.save, 'summary_data.pickle'), 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
