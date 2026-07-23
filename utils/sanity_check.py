'''
Sanity-check helpers for preprocessed EMG + behaviour pickles.

EMG and forces are timelocked to WAIT_PLAN (state 3) — the DIO trigger rise.
'''
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils import behaviour_functions

FS_EMG = 2148.1481


def list_subjects(analysis_path):
    '''Subjects that have both s{sn}_emg.pkl and s{sn}_behav.pkl.'''
    subjects = []
    for f in sorted(glob.glob(os.path.join(analysis_path, 's*_emg.pkl'))):
        sn = int(os.path.basename(f)[1:].split('_')[0])
        if os.path.exists(os.path.join(analysis_path, f's{sn}_behav.pkl')):
            subjects.append(sn)
    return subjects


def load_subject(analysis_path, sn):
    '''Load EMG and behaviour dataframes for one subject.'''
    emg_path = os.path.join(analysis_path, f's{sn}_emg.pkl')
    behav_path = os.path.join(analysis_path, f's{sn}_behav.pkl')
    if not os.path.exists(emg_path) or not os.path.exists(behav_path):
        raise FileNotFoundError(f'Missing pickles for sn={sn} in {analysis_path}')
    return pd.read_pickle(emg_path), pd.read_pickle(behav_path)


def _match_behav(df_behav, day, BN, TN):
    match = df_behav[
        (df_behav['day'] == day) & (df_behav['BN'] == BN) & (df_behav['TN'] == TN)
    ]
    if match.empty:
        return None
    return match.iloc[0]


def pick_random_trial(analysis_path, sn=None, random_state=None):
    '''
        Randomly select one EMG trial (and matching behaviour row).

        Returns a dict with sn, day, BN, TN, chordID, row_emg, row_behav.
    '''
    subjects = list_subjects(analysis_path)
    if not subjects:
        raise FileNotFoundError(f'No paired pickles in {analysis_path}')
    rng = np.random.default_rng(random_state)
    if sn is None:
        sn = int(rng.choice(subjects))
    else:
        sn = int(sn)
        if sn not in subjects:
            raise FileNotFoundError(f'No paired pickles for sn={sn}')

    df_emg, df_behav = load_subject(analysis_path, sn)
    row_emg = df_emg.sample(1, random_state=random_state).iloc[0]
    day, BN, TN = int(row_emg['day']), int(row_emg['BN']), int(row_emg['TN'])
    row_behav = _match_behav(df_behav, day, BN, TN)
    if row_behav is None:
        raise ValueError(f'No behaviour match for sn={sn} day={day} BN={BN} TN={TN}')

    return {
        'sn': sn,
        'day': day,
        'BN': BN,
        'TN': TN,
        'chordID': int(row_emg['chordID']),
        'row_emg': row_emg,
        'row_behav': row_behav,
    }


def _scale_forces_for_overlay(emg, forces):
    '''Scale/shift forces so they sit clearly above EMG on a shared y-axis.'''
    emg_peak = np.nanpercentile(np.abs(emg), 99)
    if emg_peak <= 0:
        emg_peak = 1.0
    force_peak = np.nanpercentile(np.abs(forces), 99)
    if force_peak <= 0:
        force_peak = 1.0
    force_display_amp = emg_peak * 1.2
    force_offset = emg_peak * 1.8
    scale = force_display_amp / force_peak
    return emg_peak, force_peak, force_display_amp, force_offset, scale


def _draw_state_lines(ax, times, labels):
    for t, lab in zip(times, labels):
        ax.axvline(t, color='0.55', lw=1.0, alpha=0.45, zorder=0)
        ax.text(
            t, 0.98, f'state {lab}',
            transform=ax.get_xaxis_transform(),
            ha='left', va='top', fontsize=9, color='0.35', rotation=90,
        )


def _add_split_legends(ax, emg_channels, force_channels, emg_cmap, force_cmap):
    emg_handles = [
        Line2D([0], [0], color=emg_cmap[i], lw=1.5, label=ch)
        for i, ch in enumerate(emg_channels)
    ]
    force_handles = [
        Line2D([0], [0], color=force_cmap[i], lw=2.0, ls='--', label=ch)
        for i, ch in enumerate(force_channels)
    ]
    leg1 = ax.legend(
        handles=emg_handles, title='EMG', loc='upper left',
        fontsize=8, framealpha=0.9, ncol=2,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=force_handles, title='Force (scaled)', loc='upper right',
        fontsize=8, framealpha=0.9,
    )


def plot_trial_emg_force(row_emg, row_behav, sn=None, fs_emg=FS_EMG, ax=None):
    '''
        Overlay one trial's EMG and forces, timelocked to state 3.

        Returns (fig, ax).
    '''
    emg = np.asarray(row_emg['emg'], dtype=float)
    emg_channels = list(row_emg['channels'])
    trial_state_emg = np.asarray(row_emg['trial_state'])
    t_emg = np.arange(emg.shape[0], dtype=float) / fs_emg

    forces = np.asarray(row_behav['forces'], dtype=float)
    force_channels = list(row_behav['channels'])
    t_force = behaviour_functions.emg_aligned_force_time(
        row_behav['trial_time'], row_behav['trial_state']
    )

    mask_f = (t_force >= -0.01) & (t_force <= t_emg[-1] + 0.01)
    t_force_plot = t_force[mask_f]
    forces_plot = forces[mask_f]

    emg_peak, force_peak, force_display_amp, force_offset, scale = (
        _scale_forces_for_overlay(emg, forces_plot)
    )
    forces_scaled = forces_plot * scale + force_offset

    emg_cmap = plt.cm.tab10(np.linspace(0, 1, len(emg_channels), endpoint=False))
    force_cmap = plt.cm.Dark2(np.linspace(0, 1, len(force_channels), endpoint=False))

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7), dpi=120)
    else:
        fig = ax.figure

    for i, ch in enumerate(emg_channels):
        ax.plot(t_emg, emg[:, i], color=emg_cmap[i], lw=1.1, alpha=0.9, label=ch)
    for i, ch in enumerate(force_channels):
        ax.plot(
            t_force_plot, forces_scaled[:, i],
            color=force_cmap[i], lw=1.8, ls='--', alpha=0.95, label=ch,
        )

    chg = np.where(np.diff(trial_state_emg) != 0)[0] + 1
    _draw_state_lines(ax, t_emg[chg], trial_state_emg[chg].astype(int))

    ax.axhline(force_offset, color='0.7', lw=0.8, ls=':', alpha=0.6)
    ax.set_xlim(0, t_emg[-1])
    ymin = min(-0.05 * emg_peak, float(np.nanmin(emg)) * 1.05)
    ymax = force_offset + force_display_amp * 1.35
    ax.set_ylim(ymin, ymax)

    day = int(row_emg['day'])
    BN = int(row_emg['BN'])
    TN = int(row_emg['TN'])
    chordID = int(row_emg['chordID'])
    sn_str = f'sn={sn}  ' if sn is not None else ''
    ax.set_xlabel('Time from EMG trigger / WAIT_PLAN (s)', fontsize=12)
    ax.set_ylabel('EMG (mV)  |  forces (scaled + offset)', fontsize=12)
    ax.set_title(
        f'{sn_str}day={day}  BN={BN}  TN={TN}  chordID={chordID}',
        fontsize=13, fontweight='bold',
    )
    ax.grid(True, alpha=0.25)
    _add_split_legends(ax, emg_channels, force_channels, emg_cmap, force_cmap)
    fig.tight_layout()
    return fig, ax


def plot_random_trial(analysis_path, sn=None, fs_emg=FS_EMG, random_state=None):
    '''Pick a random trial and plot EMG + force overlay.'''
    trial = pick_random_trial(analysis_path, sn=sn, random_state=random_state)
    print(
        f"Selected sn={trial['sn']}  day={trial['day']}  BN={trial['BN']}  "
        f"TN={trial['TN']}  chordID={trial['chordID']}"
    )
    fig, ax = plot_trial_emg_force(
        trial['row_emg'], trial['row_behav'], sn=trial['sn'], fs_emg=fs_emg
    )
    plt.show()
    return trial, fig, ax


def average_chord_trials(
    analysis_path, chordID, day, BN=None, sn=None, fs_emg=FS_EMG
):
    '''
        Average EMG and forces for a chord selection, timelocked to state 3.

        BN / sn may be None (= all available). Trials are truncated to the
        shortest length. Forces are resampled onto each trial's EMG clock
        before averaging.

        Returns a dict with means, SEMs, time, channels, metadata.
    '''
    subjects = list_subjects(analysis_path)
    if not subjects:
        raise FileNotFoundError(f'No paired pickles in {analysis_path}')

    sn_list = subjects if sn is None else [int(sn)]
    missing = [s for s in sn_list if s not in subjects]
    if missing:
        raise FileNotFoundError(f'Missing pickles for subjects: {missing}')

    emg_stack, force_stack, state_stack, trial_meta = [], [], [], []
    emg_channels = force_channels = None
    n_skipped = 0

    for s in sn_list:
        df_emg, df_behav = load_subject(analysis_path, s)
        mask = (df_emg['day'] == day) & (df_emg['chordID'] == chordID)
        if BN is not None:
            mask = mask & (df_emg['BN'] == BN)

        for _, row_e in df_emg.loc[mask].iterrows():
            row_b = _match_behav(df_behav, row_e['day'], row_e['BN'], row_e['TN'])
            if row_b is None:
                n_skipped += 1
                continue

            emg_tr = np.asarray(row_e['emg'], dtype=float)
            state_tr = np.asarray(row_e['trial_state'])
            if emg_channels is None:
                emg_channels = list(row_e['channels'])
                force_channels = list(row_b['channels'])

            t_emg_tr = np.arange(emg_tr.shape[0], dtype=float) / fs_emg
            try:
                t_force = behaviour_functions.emg_aligned_force_time(
                    row_b['trial_time'], row_b['trial_state']
                )
            except ValueError:
                n_skipped += 1
                continue

            forces_tr = np.asarray(row_b['forces'], dtype=float)
            forces_on_emg = np.column_stack([
                np.interp(
                    t_emg_tr, t_force, forces_tr[:, c],
                    left=np.nan, right=np.nan,
                )
                for c in range(forces_tr.shape[1])
            ])

            emg_stack.append(emg_tr)
            force_stack.append(forces_on_emg)
            state_stack.append(state_tr)
            trial_meta.append((s, int(row_e['BN']), int(row_e['TN'])))

    n_trials = len(emg_stack)
    if n_trials == 0:
        raise ValueError(
            f'No trials found for chordID={chordID}, day={day}, BN={BN}, sn={sn}'
        )

    T = min(a.shape[0] for a in emg_stack)
    emg_arr = np.stack([a[:T] for a in emg_stack], axis=0)
    force_arr = np.stack([a[:T] for a in force_stack], axis=0)
    state_arr = np.stack([a[:T] for a in state_stack], axis=0)
    t = np.arange(T, dtype=float) / fs_emg

    trans_times = {}
    for st in state_arr:
        chg = np.where(np.diff(st) != 0)[0] + 1
        for idx in chg:
            trans_times.setdefault(int(st[idx]), []).append(t[idx])
    mean_trans = {s: float(np.mean(ts)) for s, ts in sorted(trans_times.items())}

    result = {
        'chordID': int(chordID),
        'day': int(day),
        'BN': BN,
        'sn': sn,
        'n_trials': n_trials,
        'n_skipped': n_skipped,
        'trial_meta': trial_meta,
        't': t,
        'emg_mean': np.nanmean(emg_arr, axis=0),
        'emg_sem': np.nanstd(emg_arr, axis=0, ddof=1) / np.sqrt(n_trials),
        'force_mean': np.nanmean(force_arr, axis=0),
        'force_sem': np.nanstd(force_arr, axis=0, ddof=1) / np.sqrt(n_trials),
        'emg_channels': emg_channels,
        'force_channels': force_channels,
        'mean_trans': mean_trans,
    }
    print(
        f"Averaged {n_trials} trials "
        f"(chordID={chordID}, day={day}, BN={BN}, sn={sn}); skipped={n_skipped}"
    )
    print(f"Truncated length T={T} samples ({t[-1]:.3f} s)")
    print(f"Mean state transitions (s): {mean_trans}")
    if sn is None:
        print(f"Subjects included: {sorted({m[0] for m in trial_meta})}")
    else:
        print(f"Blocks included: {sorted({m[1] for m in trial_meta})}")
    return result


def plot_mean_emg_force(avg, ax=None):
    '''
        Plot mean ± SEM EMG and force overlay from average_chord_trials().

        Returns (fig, ax).
    '''
    t = avg['t']
    emg_mean = avg['emg_mean']
    emg_sem = avg['emg_sem']
    force_mean = avg['force_mean']
    force_sem = avg['force_sem']
    emg_channels = avg['emg_channels']
    force_channels = avg['force_channels']

    emg_peak, force_peak, force_display_amp, force_offset, scale = (
        _scale_forces_for_overlay(emg_mean, force_mean)
    )
    force_mean_plot = force_mean * scale + force_offset
    force_sem_plot = force_sem * scale

    emg_cmap = plt.cm.tab10(np.linspace(0, 1, len(emg_channels), endpoint=False))
    force_cmap = plt.cm.Dark2(np.linspace(0, 1, len(force_channels), endpoint=False))

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7), dpi=120)
    else:
        fig = ax.figure

    for i, ch in enumerate(emg_channels):
        ax.plot(t, emg_mean[:, i], color=emg_cmap[i], lw=1.4, alpha=0.95, label=ch)
        ax.fill_between(
            t, emg_mean[:, i] - emg_sem[:, i], emg_mean[:, i] + emg_sem[:, i],
            color=emg_cmap[i], alpha=0.15, linewidth=0,
        )
    for i, ch in enumerate(force_channels):
        ax.plot(
            t, force_mean_plot[:, i],
            color=force_cmap[i], lw=2.0, ls='--', alpha=0.95, label=ch,
        )
        ax.fill_between(
            t,
            force_mean_plot[:, i] - force_sem_plot[:, i],
            force_mean_plot[:, i] + force_sem_plot[:, i],
            color=force_cmap[i], alpha=0.12, linewidth=0,
        )

    _draw_state_lines(ax, list(avg['mean_trans'].values()), list(avg['mean_trans'].keys()))

    ax.axhline(force_offset, color='0.7', lw=0.8, ls=':', alpha=0.6)
    ax.set_xlim(0, t[-1])
    ymin = min(-0.05 * emg_peak, float(np.nanmin(emg_mean)) * 1.05)
    ymax = force_offset + force_display_amp * 1.35
    ax.set_ylim(ymin, ymax)

    bn_str = 'all' if avg['BN'] is None else str(avg['BN'])
    sn_str = 'all' if avg['sn'] is None else str(avg['sn'])
    ax.set_xlabel('Time from EMG trigger / WAIT_PLAN (s)', fontsize=12)
    ax.set_ylabel('mean EMG (mV)  |  mean force (scaled + offset)', fontsize=12)
    ax.set_title(
        f"Mean ± SEM  |  chordID={avg['chordID']}  day={avg['day']}  "
        f"BN={bn_str}  sn={sn_str}  (n={avg['n_trials']})",
        fontsize=13, fontweight='bold',
    )
    ax.grid(True, alpha=0.25)
    _add_split_legends(ax, emg_channels, force_channels, emg_cmap, force_cmap)
    fig.tight_layout()
    return fig, ax


def plot_chord_average(
    analysis_path, chordID, day, BN=None, sn=None, fs_emg=FS_EMG
):
    '''Average matching trials and plot mean ± SEM EMG + force overlay.'''
    avg = average_chord_trials(
        analysis_path, chordID=chordID, day=day, BN=BN, sn=sn, fs_emg=fs_emg
    )
    fig, ax = plot_mean_emg_force(avg)
    plt.show()
    return avg, fig, ax
