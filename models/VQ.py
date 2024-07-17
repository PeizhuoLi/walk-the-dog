import torch
import torch.nn as nn
import numpy as np
import Library.Utility as utility
from functools import partial
from modules import MLP, MLPChannels
import torch.nn.functional as F


def create_model_from_args(args, motion_data):
    kernel_size = args.kernel_size if hasattr(args, 'kernel_size') else None
    networks = []
    for data in motion_data:
        n_input_channels = data.get_n_channel_by_names(args.needed_channel_names)
        intermediate_channels = args.intermediate_channels if hasattr(args, 'intermediate_channels') else n_input_channels // 3
        network = VQAE(
            n_input_channels=n_input_channels,
            n_latent_channels=args.n_latent_channel,
            window=args.window,
            n_layers=args.n_layers,
            kernel_size=kernel_size,
            intermediate_channels=intermediate_channels,
            n_layers_fft=args.n_layers_fft,
            time_range=data.frames_per_window,
            n_layers_state=args.n_layers_state,
            v_quantizer=None,
            use_vq=args.use_vq,
            n_timing_phases=args.phase_channels
        )
        networks.append(network)
    VQ = ResidualVectorQuantizer(args.use_vq, args.num_embed_vq, network.num_embed, args.beta, distance=args.vq_distance,
                                 anchor='probrandom', first_batch=False, contras_loss=args.use_contrastive_loss, n_dataset=len(motion_data),
                                 multiple_updater=args.multiple_updater)
    for network in networks:
        network.v_quantizer = VQ.forward
    return networks, VQ


class VectorQuantizer(nn.Module):
    def __init__(self, num_embed, embed_dim, beta, distance='cos',
                 anchor='probrandom', first_batch=False, contras_loss=False, n_dataset=1,
                 multiple_updater=False):
        """
            Taken from https://github.com/lyndonzheng/CVQ-VAE
            This class implements a feature buffer that stores previously encoded features

            This buffer enables us to initialize the codebook using a history of generated features
            rather than the ones produced by the latest encoders
        """
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = distance
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False
        self.multiple_updater = multiple_updater

        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.usage = np.zeros((n_dataset, self.num_embed), dtype=np.int32)
        self.calling_from = -1
        if self.multiple_updater:
            updater = [self.UpdateModule(self.num_embed, self.decay, anchor) for _ in range(n_dataset)]
        else:
            updater = [self.UpdateModule(self.num_embed, self.decay, anchor)] * n_dataset
        self.updater = nn.ModuleList(updater)

    def get_weight(self):
        return self.embedding.weight

    class UpdateModule(nn.Module):
        def __init__(self, num_embed, decay, anchor):
            super().__init__()
            self.buffer = {}
            self.num_embed = num_embed
            self.decay = decay
            self.register_buffer("embed_prob", torch.zeros(self.num_embed))
            self.clear_buffer()
            self.anchor = anchor

        def clear_buffer(self):
            self.buffer['encodings'] = []
            self.buffer['d'] = []
            self.buffer['z_flattened'] = []

        def update_prob(self, prob):
            self.embed_prob.mul_(self.decay).add_(prob, alpha=1 - self.decay)

        def get_alpha(self):
            return torch.exp(-(self.embed_prob * self.num_embed * 10) / (1 - self.decay) - 1e-3).unsqueeze(1)

        def unpack_buffer(self, clear=True):
            encodings = torch.concat(self.buffer['encodings'], axis=0)
            d = torch.concat(self.buffer['d'], axis=0)
            z_flattened = torch.concat(self.buffer['z_flattened'], axis=0)
            if clear:
                self.clear_buffer()
            return encodings, d, z_flattened

        def update_buffer(self, encodings, d, z_flattened):
            self.buffer['encodings'].append(encodings)
            self.buffer['d'].append(d)
            self.buffer['z_flattened'].append(z_flattened)

        def get_update(self, embeddings):
            encodings, d, z_flattened = self.unpack_buffer(True)
            avg_probs = torch.mean(encodings, dim=0)
            self.update_prob(avg_probs)
            random_feat = self.sample_feat(d, z_flattened)
            decay = self.get_alpha()
            update = (1 - decay) * embeddings + decay * random_feat
            return update

        def sample_feat(self, d, z_flattened):
            if self.anchor == 'closest':
                sort_distance, indices = d.sort(dim=0)
                random_feat = z_flattened.detach()[indices[-1, :]]
            # feature pool based random sampling
            elif self.anchor == 'random':
                random_feat = self.pool.query(z_flattened.detach())
            # probability based random sampling
            elif self.anchor == 'probrandom':
                norm_distance = F.softmax(d.t(), dim=1)
                prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                random_feat = z_flattened.detach()[prob]
            return random_feat

    def set_caller(self, idx):
        self.calling_from = idx

    def clear_buffer(self):
        for updater in self.updater:
            updater.clear_buffer()

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        # z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_shape = z.shape
        z_flattened = z.view(-1, self.embed_dim)

        # clculate the distance
        if self.distance == 'l2':
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.matmul(z_flattened.detach(), self.embedding.weight.t())
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.matmul(normed_z_flattened, normed_codebook.t())

        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:, -1]

        # quantise and unflatten
        z_q = self.embedding.weight[encoding_indices]
        # reshape back to match original input shape
        z_q = z_q.reshape(z_shape)

        if self.training:
            # compute loss for embedding
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
            # preserve gradients
            z_q = z + (z_q - z).detach()

            encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
            encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            min_encodings = encodings
        else:
            loss = torch.tensor(0., device=z.device)
            perplexity = torch.zeros(1, device=z.device)
            min_encodings = torch.zeros(1, device=z.device)

        # update the running usage
        if self.training and self.calling_from >= 0:
            np.add.at(self.usage[self.calling_from], encoding_indices.detach().cpu().numpy(), 1)
            self.updater[self.calling_from].update_buffer(encodings, d, z_flattened)

        # contrastive loss
        if self.training and self.contras_loss:
            sort_distance, indices = d.sort(dim=0)
            dis_pos = sort_distance[-max(1, int(sort_distance.size(0) / self.num_embed)):, :].mean(dim=0,
                                                                                                   keepdim=True)
            dis_neg = sort_distance[:int(sort_distance.size(0) * 1 / 2), :]
            dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
            contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
            loss = loss + contra_loss

        return z_q, loss, (perplexity, min_encodings, encoding_indices)

    def reinitialize(self):
        # online clustered reinitialisation for unoptimized points
        if self.training:
            if self.multiple_updater:
                updates = [self.updater[i].get_update(self.embedding.weight) for i in range(len(self.updater))]
                self.embedding.weight.data = torch.stack(updates, dim=0).mean(dim=0)
            else:
                updater = self.updater[0]
                self.embedding.weight.data = updater.get_update(self.embedding.weight)


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_steps, *args, **kwargs):
        super().__init__()
        self.num_steps = num_steps
        vqs = []
        for _ in range(num_steps):
            vqs.append(VectorQuantizer(*args, **kwargs))
        self.vqs = nn.ModuleList(vqs)

    def forward(self, z):
        if torch.isnan(z).any():
            zero = torch.tensor(0., device=z.device)
            return z, zero, [[zero, zero, torch.zeros((z.shape[0]), device=z.device, dtype=torch.int64)] for _ in range(self.num_steps)]
        z_quant = 0.
        loss = 0.
        infos = []
        for i, vq in enumerate(self.vqs):
            z_q, l, info = vq(z)
            z_quant = z_quant + z_q
            z = z - z_q
            loss = loss + l
            infos.append(info)

        return z_quant, loss, infos

    def reinitialize(self):
        for vq in self.vqs:
            vq.reinitialize()

    def set_caller(self, idx):
        for vq in self.vqs:
            vq.set_caller(idx)

    def clear_buffer(self):
        for vq in self.vqs:
            vq.clear_buffer()

    def clear_usage(self):
        for vq in self.vqs:
            vq.usage[:] = 0.

    def get_weight(self):
        res = []
        for vq in self.vqs:
            res.append(vq.get_weight())
        return torch.stack(res, dim=0)


class FeaturePool:
    """
    Taken from https://github.com/lyndonzheng/CVQ-VAE
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features


@utility.numpy_wrapper
def get_phase_manifold(state, angles):
    """
    :param state: (batch_size, n_channel_latent)
    :param angles: (batch_size, n_channel_phase, time_range)
    :return:
    """
    state = state.reshape((state.shape[0], angles.shape[1], -1, 2))
    y0 = torch.cos(angles)
    y1 = torch.sin(angles)
    y = torch.stack((y0, y1), dim=-2)
    signal = y
    y = state @ y
    y = y.reshape(y.shape[0], -1, y.shape[-1])
    return y, signal


class VQAE(nn.Module):
    def __init__(self, n_input_channels, n_latent_channels, window, n_layers, kernel_size,
                 intermediate_channels, n_layers_fft, time_range, n_layers_state, v_quantizer,
                 use_vq=True, n_timing_phases=1):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.n_latent_channels = n_latent_channels
        self.window = window
        self.v_quantizer = v_quantizer
        self.time_range = time_range
        self.use_vq = use_vq
        self.n_timing_phases = n_timing_phases

        self.num_embed = 2 * n_latent_channels

        self.tpi = nn.Parameter(torch.tensor(2 * np.pi, dtype=torch.float32), requires_grad=False)
        self.args = nn.Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range,
                                                              dtype=np.float32)), requires_grad=False)

        n_channels = [n_input_channels] + [intermediate_channels] * (n_layers - 1) + [n_latent_channels]
        normalizer = partial(utility.LN_v3, keep_std=True)
        self.convs = []
        for i in range(n_layers):
            self.convs.append(nn.Conv1d(n_channels[i], n_channels[i+1], kernel_size, padding='same'))
            self.convs.append(normalizer(time_range))
            self.convs.append(nn.ELU())
        self.convs = nn.Sequential(*self.convs)

        self.phase_conv = nn.Sequential(
            nn.Conv1d(n_latent_channels, n_timing_phases, kernel_size, padding='same'))
        in_length = time_range // 2 + 1
        self.freq_fc = MLP(n_layers_fft, in_length, 1, 1, bn=False, last_activation=True)

        n_channels_state_mlp = [n_latent_channels] + [self.num_embed] * n_layers_state
        self.state_fc = MLPChannels(n_channels_state_mlp, bn=False)

        self.deconvs = []
        n_channels = n_channels[::-1]
        for i in range(n_layers):
            self.deconvs.append(nn.Conv1d(n_channels[i], n_channels[i+1], kernel_size, padding='same'))
            if i != n_layers - 1:
                self.deconvs.append(normalizer(time_range))
                self.deconvs.append(nn.ELU())
        self.deconvs = nn.Sequential(*self.deconvs)

    def fft_with_nn(self, func, dim):
        amp = torch.std(func, dim=dim) * np.sqrt(2)
        amp = torch.ones_like(amp)
        offset = torch.mean(func, dim=dim)

        rfft = torch.fft.rfft(func, dim=dim) / self.time_range * 2
        rfft = rfft.abs() ** 2
        func = rfft

        freq = self.freq_fc(func).squeeze(-1)

        return freq, amp, offset

    def analytical_phase(self, latent, f, b):
        b = b.unsqueeze(-1)
        f = f.unsqueeze(-1)

        y = latent - b
        sx = torch.sum(y * torch.cos(self.tpi * f * self.args), dim=2)
        sy = torch.sum(y * torch.sin(self.tpi * f * self.args), dim=2)
        p = -torch.atan2(sy, sx) / self.tpi
        return p

    def pae(self, latent):
        latent1d = self.phase_conv(latent)
        f, a, b = self.fft_with_nn(latent1d, dim=2)
        p = self.analytical_phase(latent1d, f, b)
        return f, a, b, p

    def forward(self, x):
        # Encoding
        latent = self.convs(x)
        f, a, b, p = self.pae(latent)

        state_input = latent.mean(axis=-1)
        state = self.state_fc(state_input)
        state_ori = state
        if self.use_vq:
            state, loss_embed, info = self.v_quantizer(state)
        else:
            loss_embed = 0.
            info = [None] * 3
            index_placeholder = torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
            index_placeholder[:] = -1
            info[2] = index_placeholder

        # Decoding
        angles = self.tpi * (f.unsqueeze(-1) * self.args + p.unsqueeze(-1))

        y, signal = get_phase_manifold(state, angles)
        manifold = y
        manifold_ori, _ = get_phase_manifold(state_ori, angles)

        y = self.deconvs(y)
        encoding_indices = torch.stack([i[2] for i in info], dim=-1)
        perplexities = [i[0] for i in info]
        return y, latent, signal, (p, f, a, b, manifold, manifold_ori), (loss_embed, perplexities, state, encoding_indices, state_ori)
