import numpy as np
import torch
import Library.Utility as utility
from torch.utils.data import Dataset
import os.path as osp


def create_dataset_from_args(args):
    paths = args.load.split(',')
    motion_data = []
    for path in paths:
        motion_data.append(FeatureCombinedData(path, args.window, args.normalize, args.test_sequence_ratio,
                                               args.std_cap, args.extra_frames,
                                               needed_channel_names='all'))
    return motion_data


def get_shape(Load):
    try:
        return utility.LoadTxtAsInt(Load + "/DataShape.txt")
    except:
        _, d1, d0 = get_combined_shape(Load)
        return np.array([d0, sum(d1)])


def check_path(path):
    if 'Datasets' not in path:
        path = osp.join('Datasets', path)
    return path


def get_combined_shape(prefix):
    filename = osp.join(prefix, 'Description.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()
    channel_names = lines[0].strip().split(',')
    channel_dims = [int(x) for x in lines[1].strip().split(',')]
    n_frames = int(lines[2].strip())
    return channel_names, channel_dims, n_frames


def get_fps(prefix):
    filename = osp.join(prefix, 'Description.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()
    if len(lines) < 6:
        return 60
    return int(lines[5].strip())


def load_single_dataset_bin(path, normalize, needed_feature_names, std_cap):
    channel_names, channel_dims, n_frames = get_combined_shape(path)
    if needed_feature_names == None or needed_feature_names == 'all':
        needed_feature_names = channel_names
    shape = (n_frames, sum(channel_dims))
    data = path + "/Data.bin"
    data = utility.ReadBinary(data, shape[0], shape[1])

    # Reorder the data channel according to the needed_feature_names
    named_data = {}
    for i, name in enumerate(channel_names):
        named_data[name] = data[:, :channel_dims[i]]
        data = data[:, channel_dims[i]:]
    assert data.shape[-1] == 0

    channel_dims = []
    data = []
    for name in needed_feature_names:
        data.append(named_data[name])
        channel_dims.append(named_data[name].shape[-1])
    data = np.concatenate(data, axis=-1)

    data_std = data.std(axis=0)
    set_std_cap(data_std, std_cap)
    data_mean = data.mean(axis=0)
    if not normalize:
        data_std[:] = 1.0
        data_mean[:] = 0.0
    data = (data - data_mean) / data_std

    return data, data_mean, data_std, channel_dims, needed_feature_names


def get_with_gather(Data, gather_window, sequence):
    gather = gather_window
    pivot = sequence[0]
    _min = sequence[1]
    _max = sequence[2]

    gather = np.clip(gather + pivot, _min, _max)

    data = Data[gather]
    data = torch.from_numpy(data).float()

    data = data.permute(1, 0)

    return data


def get_with_gather_numpy(Data, gather_window, sequence):
    gather = gather_window
    pivot = sequence[0]
    _min = sequence[1]
    _max = sequence[2]

    gather = np.clip(gather + pivot, _min, _max)

    data = Data[gather].astype(np.float32)
    return data


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.single_frame = False

    def set_single_frame(self, val):
        self.single_frame = val

    def prepare_sequence(self, frames, Load, extra_frames, test_sequence_ratio):
        Shape = get_shape(Load)
        Sequences = utility.LoadSequences(Load + "/Sequences.txt", True, Shape[0])

        sample_count = Shape[0]
        feature_dim = Shape[1]
        gather_padding = (int((frames - 1) / 2))
        gather_window = np.arange(frames + extra_frames) - gather_padding
        gather_window_test = np.arange(frames) - gather_padding

        print("Generating Data Sequences")
        data_sequences = []
        test_sequences = []

        for i in range(Sequences[-1]):
            utility.PrintProgress(i, Sequences[-1])
            indices = np.where(Sequences == (i + 1))[0]
            for j in range(indices.shape[0]):
                slice = [indices[j], indices[0], indices[-1]]
                data_sequences.append(slice)
                if np.random.uniform(0, 1) < test_sequence_ratio and (
                        indices[0] + gather_padding) <= indices[j] <= (indices[-1] - gather_padding):
                    test_sequences.append(j)

        print("Data Sequences:", len(data_sequences))
        print("Test Sequences:", len(test_sequences))
        data_sequences = np.array(data_sequences)

        self.Sequences = Sequences
        self.data_sequences = data_sequences
        self.test_sequences = test_sequences
        self.sample_count = len(data_sequences)
        self.gather_window = gather_window
        self.gather_window_test = gather_window_test
        self.window_size = len(gather_window)
        self.window_size_test = len(gather_window_test)
        self.feature_dim = feature_dim

        self.Data = None
        self.data_mean = 0.
        self.data_std = 1.

    def get_window_starting_frame_index(self, item):
        gather = self.gather_window
        sequence = self.data_sequences[item]
        pivot = sequence[0]
        _min = sequence[1]
        _max = sequence[2]

        gather = np.clip(gather + pivot, _min, _max)
        return gather[0]

    def __getitem__(self, item):
        gather = self.gather_window
        sequence = self.data_sequences[item]
        pivot = sequence[0]
        _min = sequence[1]
        _max = sequence[2]

        gather = np.clip(gather + pivot, _min, _max)

        data = self.Data[gather]
        data = torch.from_numpy(data).float()

        data = data.permute(1, 0)

        return data

    # def sample_test_sequence(self):
    #     return self[np.random.choice(self.test_sequences)].unsqueeze(0)

    def get_window_bound(self, item):
        sequence = self.data_sequences[item]
        _min = sequence[1]
        _max = sequence[2]
        return _min, _max

    def sample_continuous_test_window(self):
        indices = self.gather_window_test + np.random.choice(self.test_sequences)
        if self.single_frame:
            return torch.from_numpy(self.Data[indices].astype(np.float32))
        return self.load_batches(indices)[..., :self.window_size_test]

    def load_batches(self, indices):
        res = []
        for i in indices:
            res.append(self[i])
        res = torch.stack(res, dim=0)
        return res

    def __len__(self):
        if self.single_frame:
            return self.Data.shape[0]
        return self.sample_count

    def sample_long_sequence(self, length):
        seq = self.Data[:length]
        seq = torch.from_numpy(seq).permute(1, 0)
        return seq


def get_dataset_name_from_path(path: str):
    path = path.strip().lower()
    if 'human' in path:
        if 'loco' in path:
            return 'human_loco'
        return 'human'
    if 'dog' in path:
        return 'dog'
    if 'mocha' in path:
        return path[path.index('mocha'):]
    return 'unknown'


class FeatureCombinedData(BaseDataset):
    def __init__(self, path, window, normalize, test_sequence_ratio, std_cap, extra_frames=0,
                 needed_channel_names=None):
        super().__init__()
        self.feature_dim = None
        path = check_path(path)
        self.fps = get_fps(path)
        frames = int(window * self.fps) + 1
        self.frames_per_window = frames
        self.prepare_sequence(frames, path, extra_frames, test_sequence_ratio)
        self.load_dataset(path, normalize, std_cap,
                          needed_channel_names)
        self.name = get_dataset_name_from_path(path)

    def load_dataset(self, path, normalize, std_cap, needed_channel_names=[]):
        data, data_mean, data_std, channel_dims, needed_channel_names = load_single_dataset_bin(path, normalize,
                                                                          needed_channel_names, std_cap)

        self.Data = data

        self.data_mean = data_mean
        self.data_std = data_std
        self.feature_dims = channel_dims
        self.channel_names = needed_channel_names

        self.indices = {}

        st = 0
        for i in range(len(needed_channel_names)):
            self.indices[needed_channel_names[i]] = slice(st, st + channel_dims[i])
            st += channel_dims[i]

        self.name_mask = None

    @utility.numpy_wrapper
    def get_feature_by_names(self, all, names):
        res = []
        for name in names:
            res.append(all[..., self.indices[name], :])
        return torch.cat(res, dim=-2)

    def get_feature_dim_by_names(self, names):
        res = []
        for name in names:
            res.append(self.feature_dims[self.channel_names.index(name)])
        return res

    def get_n_channel_by_names(self, names):
        res = 0
        for name in names:
            idx = self.channel_names.index(name)
            res += self.feature_dims[idx]
        return res

    def __getitem__(self, item):
        val = super().__getitem__(item)
        if self.name_mask is not None:
            val = self.get_feature_by_names(val, self.name_mask)
        return val


def set_std_cap(data_std, cap):
    print(f"Set {(data_std < cap).sum()} entries cap to", cap)
    print("The entries are", np.where(data_std < cap)[0])
    data_std[data_std < cap] = cap


class SequenceAndManifold(BaseDataset):
    def __init__(self, path, window, test_sequence_ratio, path4manifold, needed_channel_names, normalize, use_manifold_ori,
                 std_cap, extra_frames=0, frames=None, needed_manifold_names=['manifold', ], normalize_manifold=True,
                 requires_full_sequence=False, additional_manifold_names=[]):
        super().__init__()
        path = check_path(path)

        data, _, _, channel_dims, needed_channel_names = load_single_dataset_bin(path, normalize=False,
                                                                                 needed_feature_names=needed_channel_names,
                                                                                 std_cap=0)

        manifold = np.load(path4manifold)
        manifold_features = []
        manifold_dims = []

        additional_manifold_features = []
        additional_manifold_dims = []
        if path4manifold.endswith('.npz'):
            for name in needed_manifold_names:
                manifold_features.append(manifold[name])
                manifold_dims.append(manifold[name].shape[-1])

            for name in additional_manifold_names:
                additional_manifold_features.append(manifold[name])
                additional_manifold_dims.append(manifold[name].shape[-1])

        manifold = np.concatenate(manifold_features, axis=-1)
        data = np.concatenate((manifold, data), axis=-1)
        self.n_channel_manifold = manifold.shape[-1]

        self.additional_manifold_names = additional_manifold_names
        self.additional_manifold_features = additional_manifold_features
        self.additional_manifold_dims = additional_manifold_dims

        data_std = data.std(axis=0)
        data_mean = data.mean(axis=0)
        if not normalize:
            data_std[:] = 1.0
            data_mean[:] = 0.0
        if not normalize_manifold:
            manifold_dim = manifold.shape[-1]
            data_std[:manifold_dim] = 1.0
            data_mean[:manifold_dim] = 0.0

        set_std_cap(data_std, std_cap)
        data = (data - data_mean) / data_std

        self.fps = get_fps(path)
        if frames is None:
            frames = int(window * self.fps) + 1
        self.frames_per_window = frames
        self.prepare_sequence(frames, path, extra_frames, test_sequence_ratio)

        if requires_full_sequence:
            self.full_sequence = utility.LoadFullSequence(path + "/Sequences.txt", True, data.shape[0])
            self.restore_full_sequence_mapping()
        self.name = get_dataset_name_from_path(path)
        if use_manifold_ori:
            self.name += '_ori'
        self.Data = data
        self.data_mean = data_mean
        self.data_std = data_std
        self.feature_dims = manifold_dims + channel_dims
        self.channel_names = needed_manifold_names + needed_channel_names
        self.fps = get_fps(path)

    def get_manifold_feature(self, name):
        feature_idx = self.additional_manifold_names.index(name)
        return self.additional_manifold_features[feature_idx]

    def get_motion_feature(self, name):
        feature_idx = self.channel_names.index(name)
        all_features = self.Data
        for i in range(feature_idx):
            all_features = all_features[..., self.feature_dims[i]:]
        return all_features[..., :self.feature_dims[feature_idx]]

    def get_motion_window(self, name, item):
        data = self.get_motion_feature(name)
        return get_with_gather_numpy(data, self.gather_window, self.data_sequences[item])

    def get_one_cycle(self):
        if self.name.startswith('dog') or self.name.startswith('human'):
            return 1
        elif self.name.startswith('mocha'):
            return 2
        else:
            raise Exception("Unknown dataset")

    def get_num_states(self):
        return self.get_manifold_feature('index').max() + 1

    def get_manifold_window(self, item, extra_frames, name, extra_frames_rear_only=False):
        feature_idx = self.additional_manifold_names.index(name)
        data = self.additional_manifold_features[feature_idx]
        gather = self.gather_window
        if extra_frames > 0:
            gather0 = np.arange(-extra_frames, 0, dtype=np.int64) + gather[0] if not extra_frames_rear_only else np.zeros((0,), dtype=np.int64)
            gather1 = np.arange(0, extra_frames, dtype=np.int64) + gather[-1] + 1
            gather = np.concatenate([gather0, gather, gather1])
        return get_with_gather_numpy(data, gather, self.data_sequences[item])

    def get_phase(self, item, extra_frames=0):
        return self.get_manifold_window(item, extra_frames, 'phase')

    def restore_full_sequence_mapping(self):
        """
        This function exists because the id for motion is modified in order to remove breaking frames
        by cutting the motion into multiple sequences.
        """
        self.full_sequence_mapping = {}
        current_count = 1
        for i in range(1, max(self.Sequences) + 1):
            self.full_sequence_mapping[i] = current_count
            indices = np.where(self.Sequences == i)[0]
            start = indices[0]
            if start == 0 or \
                (self.full_sequence[start][-1] != self.full_sequence[start-1][-1] or  \
                        self.full_sequence[start][2] != self.full_sequence[start-1][2]):
                current_count += 1
            else:
                # print('Something is wrong')
                pass
