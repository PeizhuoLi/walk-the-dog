import os
import os.path as osp

import numpy as np


def convert_from_unbiased_index(motion_data, idx):
    sequence = motion_data.full_sequence[idx]
    sequence_id = motion_data.full_sequence_mapping[int(sequence[0])]
    return (sequence_id - 1) // 2, int(sequence[1]) - 1, sequence[2] == 'Mirrored'


def export_npz(filename, replay_match, target_dataset, transit, version=1, source_start_idx=None):
    to_save = {}
    to_save['Version'] = np.array([version], dtype=np.int32)

    segment_number = np.cumsum(transit.astype(np.int32))
    to_save['SegmentNumber'] = segment_number

    # For debugging
    to_save['original_replay_match'] = replay_match
    if source_start_idx is not None:
        to_save['source_start_idx'] = np.array([source_start_idx], dtype=np.int32)

    replay_match_int = np.floor(replay_match).astype(np.int32)
    seq_num = []
    frame_num = []
    mirrored = []

    for i in range(len(replay_match)):
        seq, frame, mirror = convert_from_unbiased_index(target_dataset, replay_match_int[i])
        seq_num.append(seq)
        frame_num.append(frame)
        mirrored.append(mirror)

    to_save['SequenceNumber'] = np.array(seq_num)
    to_save['FrameNumber'] = np.array(frame_num)
    to_save['Mirrored'] = np.array(mirrored)

    if version == 2:
        to_save['FrameNumber'] = to_save['FrameNumber'] + replay_match - replay_match_int
        to_save['FrameNumber'] = to_save['FrameNumber'].astype(np.float32)

    os.makedirs(osp.dirname(filename), exist_ok=True)
    np.savez(filename, **to_save)