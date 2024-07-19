import numpy as np
import torch
from motion_matching.feature_utils import create_resampled_mm_features_single_window, create_mock_manifold


def L2(x, y):
    return torch.sum((x - y) ** 2, dim=-1) ** 0.5


def scale_error(x, y):
    return torch.maximum(x / y, y / x)


def find_nn_separate(control, start_pose, control_dataset, start_pose_dataset, weight_pose, frequency, frequency_dataset, weight_frequency):
    distance_feature = L2(control, control_dataset)
    distance_pose = L2(start_pose, start_pose_dataset) if start_pose is not None else torch.zeros_like(distance_feature)
    distance_frequency = scale_error(frequency, frequency_dataset) if frequency is not None else torch.zeros_like(distance_feature)
    scale_limit = 1.5
    distance_frequency[distance_frequency > scale_limit] = 10000
    distance = distance_feature + weight_pose * distance_pose + weight_frequency * distance_frequency
    nn = torch.argmin(distance)
    return nn, [distance_feature[nn].item(), distance_pose[nn].item(), distance_frequency[nn].item()], frequency_dataset[nn].item()


def motion_matching_kernel(source_motion_data, start_window_idx, end_window_idx, mm_database, source_one_cycle, initial_pose=None):
    window_size = mm_database.window_size
    frequency_cap = mm_database.frequency_cap

    replay_matching = np.zeros((0,), dtype=np.float32)
    transit = np.zeros((end_window_idx - start_window_idx + window_size * 2,), dtype=bool)
    current_starting_frame = start_window_idx

    while current_starting_frame < end_window_idx:
        feature_new_window = create_resampled_mm_features_single_window(source_motion_data, current_starting_frame,
                                                                        frequency_cap, window_size, source_one_cycle)
        new_control = feature_new_window[0]
        delta_frames = feature_new_window[2]
        new_frequency = feature_new_window[4]

        if len(replay_matching) > 0:
            new_start_pose = mm_database.get_pose_from_frame_idx(np.round(replay_matching[-1]).astype(np.int32).item() + 1)
        elif initial_pose is not None:
            new_start_pose = initial_pose
        else:
            new_start_pose = None

        matched_window_idx, length_in_matched, losses, _ = mm_database.match_with_target(new_control, new_start_pose, new_frequency)

        print(delta_frames, losses)
        replay = mm_database.sample_from_matched_window(matched_window_idx, delta_frames)
        replay_matching = np.append(replay_matching, replay)

        transit[replay_matching.shape[0]] = 1
        current_starting_frame += delta_frames

    return replay_matching, transit


def motion_retrival_kernel(source_state_idx, source_frequency, mm_database, cycles=1, initial_pose=None):
    window_size = mm_database.window_size

    replay_matching = np.zeros((0,), dtype=np.float32)
    transit = np.zeros(20000, dtype=bool)
    current_starting_frame = 0

    for i in range(cycles):
        feature_new_window = create_mock_manifold(mm_database.motion_data, source_state_idx, window_size)

        new_control = feature_new_window[0]
        new_frequency = source_frequency

        if len(replay_matching) > 0:
            new_start_pose = mm_database.get_pose_from_frame_idx(np.round(replay_matching[-1]).astype(np.int32).item() + 1)
        elif initial_pose is not None:
            new_start_pose = initial_pose
        else:
            new_start_pose = None

        matched_window_idx, length_in_matched, losses, nearest_frequency = mm_database.match_with_target(new_control, new_start_pose, new_frequency)

        delta_frames = length_in_matched

        # print(delta_frames, losses, nearest_frequency)
        replay = mm_database.sample_from_matched_window(matched_window_idx, delta_frames)
        replay_matching = np.append(replay_matching, replay)

        transit[replay_matching.shape[0]] = 1
        current_starting_frame += delta_frames

    return replay_matching, transit, nearest_frequency