import numpy as np


def resample_with_index(v, t, interpolation='linear'):
    """
    Args:
        v: shape = (n_frame, n_dim)
        t: shape = (n_point). t = 0 corresponds to v[0].
        interpolation:
    Returns:
    """

    if interpolation == 'nearest':
        index = np.round(t).astype(np.int32)
        return v[index]

    elif interpolation == 'linear':
        lower_f = np.floor(t)
        lower = lower_f.astype(np.int32)
        upper = lower + 1
        l = (t - lower_f)[..., None]
        return v[lower] * (1 - l) + v[upper] * l

    else:
        raise NotImplementedError("Unknown interpolation method: " + interpolation)


def find_resample_length(f, scale_cap, target_length, one_cycle):
    f = np.clip(f, 1 / scale_cap, scale_cap).astype(np.float32)
    fsum = np.cumsum(f) / target_length # * 2    # * 2 is for Mocha because it's 120fps
    if fsum[-1] < one_cycle:
        final_length = f.shape[0] + 1
    else:
        final_length = np.argmax(fsum > one_cycle) + 1
    return final_length, np.arange(target_length, dtype=np.float32) / target_length * final_length
