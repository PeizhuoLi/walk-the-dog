import torch
import numpy as np
from motion_matching.DTW import find_resample_length, resample_with_index
from models.VQ import get_phase_manifold



def split_by_channel(window, channel_dims):
    res = []
    for d in channel_dims:
        res.append(window[..., :d, :])
        window = window[..., d:, :]
    return res


def get_pose(window, feature_dims):
    if window.ndim == 1:
        window = window.unsqueeze(-1)
    res = split_by_channel(window, feature_dims)
    return res[2][..., 0]


def get_pose_from_frame_idx(motion_data, pivot):
    data = motion_data.Data[pivot][..., None]
    res = split_by_channel(data, motion_data.feature_dims)
    return res[2].squeeze(-1)


def get_control(window, feature_dims):
    if window.ndim == 1:
        window = window.unsqueeze(-1)
    res = split_by_channel(window, feature_dims)
    return res[0].reshape(-1)


def compose_features(window, name_channels, dim_channels):
    """
    Args:
        window: (sum(dim_channels), window_size)
    Returns:
    """
    final_names = ['manifold_per_frame', 'start_frame']
    final_features = []
    final_dims = []

    # Manifold per frame
    control = get_control(window, dim_channels)
    final_features.append(control.squeeze(0))
    final_dims.append(control.shape[-1])

    # Start frame
    pose = get_pose(window, dim_channels)
    final_features.append(pose.squeeze(0))
    final_dims.append(pose.shape[-1])

    final_features = torch.cat(final_features, dim=-1)

    return final_features, final_names, final_dims


def create_resampled_mm_features_single_window(motion_data, idx, frequency_cap, window_size, one_cycle):
    extra_frames = window_size * (frequency_cap - 1) + 1 # Because the ceil might use the next one∂

    valid_phases = motion_data.get_manifold_window(idx, extra_frames, 'phase', True)
    valid_states = motion_data.get_manifold_window(idx, extra_frames, 'state', True)
    frequency_window = motion_data.get_manifold_window(idx, extra_frames, 'frequency', True)
    start_pose = get_pose(motion_data[idx], motion_data.feature_dims)

    resample_length, t = find_resample_length(frequency_window, frequency_cap, window_size, one_cycle)
    frequency = window_size / resample_length
    resampled_phase = resample_with_index(valid_phases, t, 'linear')
    resampled_state = resample_with_index(valid_states, t, 'nearest')

    resampled_manifold, _ = get_phase_manifold(resampled_state, 2 * np.pi * resampled_phase.reshape(-1, 1, 1))
    resampled_manifold = resampled_manifold.reshape(-1)
    return resampled_manifold, start_pose, resample_length, resampled_phase, frequency


def create_mock_manifold(motion_data, state_idx, length, harmonic_number):
    mock_phase = np.arange(length, dtype=np.float32) / length
    data_state_indices = motion_data.get_manifold_feature('index')
    idx = np.argmax(data_state_indices == state_idx)
    state = motion_data.get_manifold_feature('state')[idx]
    state = state.reshape(1, -1)
    manifold, _ = get_phase_manifold(state, 2 * np.pi * mock_phase.reshape(-1, 1, 1), harmonic_number=harmonic_number)
    manifold = manifold.reshape(-1)
    return manifold, None, None, mock_phase, None