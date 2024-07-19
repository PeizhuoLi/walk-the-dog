import argparse
import sys
import os


class BaseOptionParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--use_tqdm', type=int, default=1)

    @staticmethod
    def checker(args):
        return args

    @staticmethod
    def text_serialize(args):
        d = vars(args)
        res = ''
        for k, v in d.items():
            res += f'--{k}={v} '
        return res

    @staticmethod
    def serialize(args):
        d = vars(args)
        return d

    def deserialize(self, d):
        args = self.parse_args('')
        args.__dict__.update(d)
        args = self.checker(args)
        return args

    def text_deserialize(self, d):
        args = self.parse_args(d)
        return args

    def parse_args(self, args_str=None):
        return self.checker(self.parser.parse_args(args_str))

    def get_parser(self):
        return self.parser

    def save(self, filename, args_str=None):
        if args_str is None:
            args_str = ' '.join(sys.argv[1:])
        path = '/'.join(filename.split('/')[:-1])
        os.makedirs(path, exist_ok=True)
        with open(filename, 'w') as file:
            file.write(args_str)

    def load(self, filename):
        with open(filename, 'r') as file:
            args_str = file.readline()
        return self.parse_args(args_str.split())


class TrainVQOptionParser(BaseOptionParser):
    def __init__(self):
        super().__init__()

        self.parser.add_argument('--window', type=float, default=1.0)
        self.parser.add_argument('--fps', type=int, default=60)

        self.parser.add_argument('--input_channels', type=int, default=0)
        self.parser.add_argument('--phase_channels', type=int, default=1)
        self.parser.add_argument('--kernel_size', type=int, default=23)
        self.parser.add_argument('--n_layers', type=int, default=2)

        self.parser.add_argument('--epochs', type=int, default=10)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--learning_rate', type=float, default=1e-4)
        self.parser.add_argument('--weight_decay', type=float, default=1e-4)
        self.parser.add_argument('--restart_period', type=int, default=10)
        self.parser.add_argument('--restart_mult', type=int, default=2)

        self.parser.add_argument('--plotting_interval', type=int, default=500)
        self.parser.add_argument('--pca_sequence_count', type=int, default=100)
        self.parser.add_argument('--test_sequence_ratio', type=float, default=0.01)

        self.parser.add_argument('--load', type=str, default="Dataset")
        self.parser.add_argument('--save', type=str, default="./results/test")

        self.parser.add_argument('--normalize', type=int, default=1)
        self.parser.add_argument('--std_cap', type=float, default=0.)

        self.parser.add_argument('--device', type=str, default='')  # Make massive helper happy

        self.parser.add_argument('--extra_frames', type=int, default=0)
        self.parser.add_argument('--lambda_rec', type=float, default=1)

        self.parser.add_argument('--n_layers_fft', type=int, default=7)

        self.parser.add_argument('--n_layers_state', type=int, default=5)
        self.parser.add_argument('--needed_channel_names', type=str, default='Velocities')
        self.parser.add_argument('--beta', type=float, default=0.25)
        self.parser.add_argument('--n_latent_channel', type=int, default=10)

        self.parser.add_argument('--num_embed_vq', type=int, default=32)
        self.parser.add_argument('--use_vq', type=int, default=1)
        self.parser.add_argument('--lambda_vq', type=float, default=1.)
        self.parser.add_argument('--vq_distance', type=str, default='l2')
        self.parser.add_argument('--multiple_updater', type=int, default=1)
        self.parser.add_argument('--use_contrastive_loss', type=int, default=0)

        self.parser.add_argument('--train_phase_decoder', type=int, default=1)
        self.parser.add_argument('--n_layers_phase_decoder', type=int, default=8)
        self.parser.add_argument('--activation_phase_decoder', type=str, default='LeakyReLU')
        self.parser.add_argument('--needed_channel_names_phase_decoder', type=str, default='Velocities,Positions,Rotations')
        self.parser.add_argument('--lr_phase_decoder', type=float, default=1e-3)
        self.parser.add_argument('--decoder_before_quantization', type=int, default=0)

        self.parser.add_argument('--debug', type=int, default=0)

    @staticmethod
    def checker(args):
        return args

    @staticmethod
    def post_process(args):
        args.needed_channel_names = args.needed_channel_names.strip().split(',')
        args.needed_channel_names_phase_decoder = args.needed_channel_names_phase_decoder.strip().split(',')
        return args


class TestOptionParser(BaseOptionParser):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--save', type=str, default='./test')
        self.parser.add_argument('--plot_save', type=str, default='./results/plots')
        self.parser.add_argument('--plot_cnt', type=int, default=5)


class PhaseDecoderParser(BaseOptionParser):
    def __init__(self):
        super().__init__()

        self.parser.add_argument('--lambda_rec', type=float, default=1.)
        self.parser.add_argument('--load', type=str, default="Dataset")
        self.parser.add_argument('--save', type=str, default="./results/test")
        self.parser.add_argument('--n_layers', type=int, default=5)
        self.parser.add_argument('--batch_size', type=int, default=128)
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--weight_decay', type=float, default=1e-4)
        self.parser.add_argument('--epochs', type=int, default=100)
        self.parser.add_argument('--normalize', type=int, default=1)
        self.parser.add_argument('--device', type=str, default='')
        self.parser.add_argument('--activation', type=str, default='ELU')
        self.parser.add_argument('--path4manifold', type=str, default='')
        self.parser.add_argument('--needed_channel_names', type=str, default='')
        self.parser.add_argument('--use_manifold_ori', type=int, default=0)

    @staticmethod
    def post_process(args):
        args.needed_channel_names = args.needed_channel_names.strip().split(',')
        return args


class MotionMatchingOption(BaseOptionParser):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--num_frames_per_window', type=int, default=60)
        self.parser.add_argument('--path4manifold', type=str, default='')
        self.parser.add_argument('--needed_channel_names', type=str, default='Velocities,Positions,Rotations')
        self.parser.add_argument('--normalize', type=int, default=1)
        self.parser.add_argument('--std_cap', type=float, default=1e-3)
        self.parser.add_argument('--weight_pose', type=float, default=0.1)
        self.parser.add_argument('--preset_name', type=str, default='human2dog')
        self.parser.add_argument('--target_id', type=int, default=0)
        self.parser.add_argument('--use_frequency_align', type=int, default=1)
        self.parser.add_argument('--export_npz_version', type=int, default=2)
        self.parser.add_argument('--output_filename', type=str, default='replay_sequence.npz')
        self.parser.add_argument('--save_prefix', type=str, default='./results/motion_matching')
        self.parser.add_argument('--input_idx', type=int, default=None)
        self.parser.add_argument('--output_idx', type=int, default=None)

    @staticmethod
    def post_process(args):
        if isinstance(args.needed_channel_names, str):
            args.needed_channel_names = args.needed_channel_names.strip().split(',')
        return args
