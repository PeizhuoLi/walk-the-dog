import os
import os.path as osp

from option import MotionMatchingOption
from motion_matching.export_npz import export_npz
from motion_matching.feature_utils import get_pose_from_frame_idx
from motion_matching.dataset_utils import find_start_and_end, prepare_dataset_only
from motion_matching.database import MotionMatchingDatabase
from motion_matching.kernel import motion_matching_kernel


def main():
    option_parser = MotionMatchingOption()
    args = option_parser.parse_args()
    os.makedirs(args.save_prefix, exist_ok=True)

    input_dataset, target_dataset, name = prepare_dataset_only(args.preset_name, args)

    mm_database = MotionMatchingDatabase(target_dataset, args.num_frames_per_window,
                                         use_frequency_align=args.use_frequency_align,
                                         weight_pose=args.weight_pose,
                                         name_suffix=name)

    start_idx, end_idx = find_start_and_end(input_dataset, args.target_id)    # Index with dataset indexing convention
    source_one_cycle = input_dataset.get_one_cycle()

    if args.preset_name == 'human2dog':
        initial_pose = mm_database.get_pose_from_frame_idx(980)
        # Since all the static poses are mapped to the same manifold, it makes more sense to set
        # the initial pose for the human to dog motion matching
    elif args.preset_name == 'dog2human':
        initial_pose = None
    else:
        initial_pose = get_pose_from_frame_idx(input_dataset, start_idx + input_dataset.gather_window[0])

    replay_match, transit = motion_matching_kernel(input_dataset, start_idx, end_idx, mm_database, source_one_cycle, initial_pose)

    export_npz(osp.join(args.save_prefix, args.output_filename), replay_match,
               target_dataset, transit, args.export_npz_version, start_idx)


if __name__ == '__main__':
    main()