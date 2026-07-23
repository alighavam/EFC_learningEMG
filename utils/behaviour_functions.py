import numpy as np

def movload(fname):
    # loads .mov files given the path of the file. The .mov
    # files have a specific custom format hence the need for a custom function.
    # Column count is inferred from the first data line (not hardcoded).
    A = []
    fid = open(fname, 'rt')
    if fid == -1:
        raise Exception('Could not open ' + fname)

    trial = 0
    n_cols = None
    for line in fid:
        if line[0] == 'T':
            # print('Trial: ', line.split()[1])
            a = int(line.split()[1])
            trial += 1
            if a != trial:
                print('Trials out of sequence')
                trial = a
            A.append(np.empty((0, n_cols if n_cols is not None else 0)))
        else:
            lineData = line.strip().split('\t')
            if n_cols is None:
                n_cols = len(lineData)
                # Retroactively set width on the current trial's empty buffer
                A[trial - 1] = np.empty((0, n_cols))
            elif len(lineData) != n_cols:
                raise ValueError(
                    f"Inconsistent column count in {fname}: expected {n_cols}, "
                    f"got {len(lineData)} (trial {trial})"
                )
            a = np.array([float(x) for x in lineData], ndmin=2)
            A[trial - 1] = np.vstack((A[trial - 1], a))

    fid.close()
    return A


# .mov column layout (Diedrichsen-lab force-box format):
#   col 0: block / run index
#   col 1: trial state (state-machine)
#   col 2: absolute clock (ms; can glitch — do not use for timing)
#   col 3: state-local clock (ms; resets across some state boundaries)
#   last 5 cols: digit forces (digit1..digit5)
#
# Force box samples at ~500 Hz (median col-2 dt ≈ 2 ms). EMG trigger pulses
# match states 4–6 (execution), not waiting (1) or planning (3).
MOV_STATE_COL = 1
MOV_FORCE_COLS = 5
FS_FORCE = 500.0  # Hz
EMG_TRIGGER_STATE = 4  # first state covered by the EMG trigger pulse


def parse_mov_trial(mov_trial, fs_force=FS_FORCE):
    '''
        Extract state, continuous trial time (s), and forces from one .mov trial.

        trial_time is sample-index / fs_force from trial onset. Prefer this over
        the absolute clock (col 2 can glitch) or the state-local clock (col 3
        resets across state boundaries).
    '''
    mov_trial = np.asarray(mov_trial, dtype=float)
    state = mov_trial[:, MOV_STATE_COL].astype(int)
    trial_time = np.arange(len(mov_trial), dtype=float) / float(fs_force)
    forces = mov_trial[:, -MOV_FORCE_COLS:]
    return state, trial_time, forces


def align_state_to_times(state, t_src, t_dst):
    '''
        Zero-order-hold map of a discrete state vector from t_src onto t_dst.

        Both time vectors should be in the same units and share a common origin
        (e.g. seconds from trial onset).
    '''
    state = np.asarray(state)
    t_src = np.asarray(t_src, dtype=float)
    t_dst = np.asarray(t_dst, dtype=float)
    if len(state) == 0 or len(t_src) == 0:
        raise ValueError("state and t_src must be non-empty.")
    if len(state) != len(t_src):
        raise ValueError("state and t_src must have the same length.")

    idx = np.searchsorted(t_src, t_dst, side='right') - 1
    idx = np.clip(idx, 0, len(state) - 1)
    return state[idx]


def align_state_to_emg(state, trial_time, t_emg, trigger_state=EMG_TRIGGER_STATE):
    '''
        Map behaviour state onto the EMG trial clock.

        The EMG trigger spans execution states (default: from state 4 onward),
        so behaviour time is re-zeroed at the first sample of trigger_state
        before zero-order-hold resampling onto t_emg.
    '''
    state = np.asarray(state)
    trial_time = np.asarray(trial_time, dtype=float)
    t_emg = np.asarray(t_emg, dtype=float)

    hits = np.where(state == trigger_state)[0]
    if len(hits) == 0:
        raise ValueError(
            f"No samples with state=={trigger_state}; cannot align to EMG trigger."
        )
    t0 = trial_time[hits[0]]
    return align_state_to_times(state, trial_time - t0, t_emg)
