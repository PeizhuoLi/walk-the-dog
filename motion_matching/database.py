import torch
from dataset import SequenceAndManifold
from motion_matching.feature_utils import create_resampled_mm_features_single_window, get_pose_from_frame_idx
from motion_matching.kernel import find_nn_separate
import numpy as np
import os
import os.path as osp
from tqdm import tqdm


database_save_prefix = './results/motion_matching/database/'


def filter_incomplete_windows(motion_data):
    all_indices = np.arange(len(motion_data))
    window_low = all_indices + motion_data.gather_window[0]
    window_up = all_indices + motion_data.gather_window[-1]
    if motion_data.name.startswith('human'):
        overflow_low = window_low < motion_data.data_sequences[:, 1] + 1
    else:
        overflow_low = window_low < motion_data.data_sequences[:, 1]
    overflow_up = window_up > motion_data.data_sequences[:, 2]
    overflow = np.logical_or(overflow_low, overflow_up)
    return overflow


def try_get_processed_windows(name, motion_data, frequency_cap, window_size, one_cycle, create=False):
    filename = osp.join(database_save_prefix, name) + '.npz'
    if os.path.exists(filename):
        res = np.load(filename)
        res = {key: res[key] for key in res}
        return res, True
    elif create:
        os.makedirs(database_save_prefix, exist_ok=True)
        res = create_resampled_mm_features(motion_data, frequency_cap, window_size,
                                           one_cycle)
        np.savez(filename, **res)
        return res, True
    else:
        return None, False


def create_resampled_mm_features(motion_data, frequency_cap, window_size, one_cycle):
    manifold_windows = []
    start_poses = []
    exact_lengths = []
    phases = []
    frequencies = []

    for i in tqdm(range(len(motion_data))):
        one_window = create_resampled_mm_features_single_window(motion_data, i, frequency_cap,
                                                                window_size, one_cycle)

        manifold_windows.append(one_window[0])
        start_poses.append(one_window[1])
        exact_lengths.append(one_window[2])
        phases.append(one_window[3])
        frequencies.append(one_window[4])

    manifold_windows = np.stack(manifold_windows)
    start_poses = np.stack(start_poses)
    exact_lengths = np.stack(exact_lengths)
    phases = np.stack(phases)
    frequencies = np.array(frequencies)

    res = {"manifold_windows": manifold_windows, "start_poses": start_poses, "exact_lengths": exact_lengths,
           "phases": phases, 'frequencies': frequencies}
    return res


class MotionMatchingDatabase:
    def __init__(self, motion_data: SequenceAndManifold, window_size, use_frequency_align=False,
                 frequency_cap=2, weight_pose=0.1, weight_frequency=1, name_suffix=None):
        if not use_frequency_align:
            frequency_cap = 1
        self.frequency_cap = frequency_cap
        self.weight_frequency = weight_frequency
        self.window2frame_bias = motion_data.gather_window[0]
        self.motion_data = motion_data
        self.motion_data_name = motion_data.name
        self.window_size = window_size
        self.one_cycle = motion_data.get_one_cycle()
        self.name_suffix = name_suffix

        name = self.get_name_with_parameters()

        mm_features, _ = try_get_processed_windows(name, motion_data, frequency_cap,
                                                   window_size, self.one_cycle, create=True)

        self.weight_pose = weight_pose
        self.mm_features = mm_features
        self.frequency_cap = frequency_cap

        self.controls = torch.from_numpy(self.mm_features['manifold_windows']).cuda()
        self.start_poses = torch.from_numpy(self.mm_features['start_poses']).cuda()
        self.frequencies = torch.from_numpy(self.mm_features['frequencies']).cuda()
        self.phases = self.mm_features['phases']
        self.controls[filter_incomplete_windows(motion_data)] = 10000000
        self.exact_lengths = self.mm_features['exact_lengths']

    def get_name_with_parameters(self):
        names = ['motion_data_name', 'window_size', 'frequency_cap']
        finale_names = "_".join([str(getattr(self, name)) for name in names])
        if self.one_cycle != 1:
            finale_names += f'_one_cycle={self.one_cycle}'
        if self.name_suffix is not None:
            finale_names += f'_{self.name_suffix}'
        return finale_names

    def match_with_target(self, control, start_pose, input_frequency=None):
        control = torch.from_numpy(control).unsqueeze(0).cuda()
        start_pose = torch.from_numpy(start_pose).unsqueeze(0).cuda() if start_pose is not None else None
        input_frequency = torch.from_numpy(np.array([input_frequency])).cuda() if input_frequency is not None else None
        nn, losses, nearest_frequency = find_nn_separate(control, start_pose,
                                      self.controls, self.start_poses, self.weight_pose,
                                      input_frequency, self.frequencies, self.weight_frequency)
        return nn.item(), self.exact_lengths[nn], losses, nearest_frequency

    def sample_from_matched_window(self, window_idx, num_frames):
        low, up = self.motion_data.get_window_bound(window_idx)
        start_frame_idx = window_idx + self.window2frame_bias

        sample = np.linspace(start_frame_idx, start_frame_idx + self.exact_lengths[window_idx],
                             num_frames, endpoint=False)
        sample = np.clip(sample, low, up)
        return sample

    def get_pose_from_frame_idx(self, frame_idx):
        return get_pose_from_frame_idx(self.motion_data, frame_idx)