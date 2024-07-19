from tqdm import tqdm
import torch
from motion_matching.feature_utils import compose_features
import numpy as np
import os
import os.path as osp
from option import MotionMatchingOption, TrainVQOptionParser
from dataset import SequenceAndManifold


def find_start_and_end(motion_data, target_id):
    mask = np.where(motion_data.Sequences == target_id * 2 - 1)[0]
    start = mask[0] - motion_data.gather_window[0]
    end = mask[-1] - motion_data.gather_window[0] + 1
    print(f'Matching on {motion_data.full_sequence[start][3]}')
    return start, end


def prepare_args_and_data(args, create):
    option_parser = MotionMatchingOption()
    base_train_option_parser = TrainVQOptionParser()

    class_identifier = int(args.path4manifold.split('/')[-1].split('_')[1])
    args_train_name = '/'.join(args.path4manifold.split('/')[:-1] + ['args.txt'])
    with open(args_train_name, "r") as f:
        args_base = base_train_option_parser.text_deserialize(f.read().split())
        args_base = base_train_option_parser.post_process(args_base)
    args.load = args_base.load.split(',')[class_identifier]
    phase_model_name = args_base.save[-7:].replace('/', '-')

    if create:
        os.makedirs(args.save, exist_ok=True)
        with open(osp.join(args.save, "args.txt"), "w") as file:
            file.write(option_parser.text_serialize(args))

    args = option_parser.post_process(args)

    motion_data = SequenceAndManifold(args.load, None, 0, args.path4manifold,
                                      args.needed_channel_names, args.normalize, False,
                                      frames=args.num_frames_per_window, std_cap=args.std_cap, requires_full_sequence=True,
                                      normalize_manifold=False, additional_manifold_names=['phase', 'frequency', 'state', 'manifold', 'index'])

    return args, motion_data, phase_model_name


def prepare_dataset_only(preset_name, args):
    if preset_name is not None:
        args.path4manifold = './pre-trained'

        if preset_name == 'human2dog':
            args.input_idx = 1
            args.output_idx = 0
        elif preset_name == 'dog2human':
            args.input_idx = 0
            args.output_idx = 1

    source_path = osp.join(args.path4manifold, f'Manifolds_{args.input_idx}_final.npz')
    target_path = osp.join(args.path4manifold, f'Manifolds_{args.output_idx}_final.npz')

    args.path4manifold = source_path
    _, motion_source, _ = prepare_args_and_data(args, False)
    args.path4manifold = target_path
    _, motion_target, _ = prepare_args_and_data(args, False)

    return motion_source, motion_target, f'{args.input_idx}_{args.output_idx}'
