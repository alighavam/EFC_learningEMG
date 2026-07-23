import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


def preprocess_emg_run(fpath, channels, fs_emg=2148.1481, fs_trigger=2222.2222,
                       riseThresh=0.6, fallThresh=0.4, min_width_ms=20,
                       bp_low=20, bp_high=500, bp_order=4,
                       lp_cutoff=30, lp_order=4, debug=0):
    '''
        Description: Load one EMG CSV, detect trigger edges, segment each trial
        onto the shared EMG clock, and preprocess (bandpass → demean → rectify →
        low-pass). Does not build a dataframe — returns a list of trial dicts so
        behaviour fields can be added before forming the subject dataframe.

        <inputs>
        fpath: Path to the EMG CSV for one run.

        channels: Ordered list of EMG channel names (without ' (mV)' suffix),
        e.g. ['flx_D1', ..., 'ext_D5'].

        fs_emg: EMG sampling rate (Hz). Default 2148.1481.

        fs_trigger: Trigger (Analog 1) sampling rate (Hz). Default 2222.2222.

        riseThresh, fallThresh, min_width_ms: Passed to find_trigger_rise_edge.

        bp_low, bp_high, bp_order: Bandpass filter params.

        lp_cutoff, lp_order: Envelope low-pass filter params.

        debug: Passed to trigger detection / filters. Default 0.

        <outputs>
        trials: List of dicts, one per trial, each with:
            'trial' (1-based int),
            'emg' ((N, C) preprocessed matrix),
            't_start', 't_end' (trigger-clock times in seconds),
            'channels' (copy of the channels list).
    '''
    # -------------------- load CSV --------------------
    header = pd.read_csv(fpath, skiprows=5, nrows=1, header=None)
    header = header.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    data = pd.read_csv(fpath, skiprows=8, header=None, usecols=list(range(28)), low_memory=False)
    data = data.iloc[:, :28]
    data.columns = header.iloc[0, :28]
    data = data.reset_index(drop=True)
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce')

    # -------------------- trigger edges --------------------
    trig_time = data['Analog 1 Time Series (s)'].values  # seconds
    trigger = data['Analog 1 (V)'].values
    n_trig = int(np.sum(~np.isnan(trig_time)))
    trig_time = trig_time[:n_trig]
    trigger = trigger[:n_trig]

    riseIdx, fallIdx = find_trigger_rise_edge(
        trigger, fs_trigger,
        riseThresh=riseThresh, fallThresh=fallThresh,
        min_width_ms=min_width_ms, debug=debug,
    )

    # -------------------- EMG time + data (shared clock across channels) --------------------
    emg_time = data['flx_D1 Time Series (s)'].values
    n_emg = int(np.sum(~np.isnan(emg_time)))
    emg_time = emg_time[:n_emg]

    emg_full = np.column_stack([
        data[f'{ch} (mV)'].values[:n_emg] for ch in channels
    ])

    # -------------------- segment + preprocess each trial --------------------
    trials = []
    for trial, (r_idx, f_idx) in enumerate(zip(riseIdx, fallIdx), start=1):
        t_start = trig_time[r_idx]
        t_end = trig_time[f_idx]

        i_start = np.searchsorted(emg_time, t_start) - 1
        i_end = np.searchsorted(emg_time, t_end) - 1

        trial_emg = emg_full[i_start:i_end + 1, :]  # (N, C)

        trial_emg = bandpass_filter_emg(
            trial_emg, fs_emg, low=bp_low, high=bp_high, order=bp_order, debug=debug
        )
        trial_emg = trial_emg - np.mean(trial_emg, axis=0)
        trial_emg = np.abs(trial_emg)
        trial_emg = low_pass_filter(
            trial_emg, fs_emg, cutoff=lp_cutoff, order=lp_order, debug=debug
        )

        trials.append({
            'trial': trial,
            'emg': trial_emg,
            't_start': float(t_start),
            't_end': float(t_end),
            'channels': list(channels),
        })

    return trials

def _trigger_is_idle_high(trig, fs, start_idx=0, baseline_ms=200):
    '''
        Returns True if the trigger channel is idle-high at the start of the recording
        (electrode wired backwards: pulses go high→0 instead of 0→high).

        Compares the median of an early baseline window to the robust [1, 99]
        percentile range of the usable signal. Baseline near the high end ⇒ inverted.
    '''
    x = np.asarray(trig[start_idx:], dtype=float)
    if len(x) < 2:
        return False

    lo, hi = np.percentile(x, [1, 99])
    scale = hi - lo
    if scale < 1e-12:
        return False

    n_base = max(1, int(round(baseline_ms * 1e-3 * fs)))
    n_base = min(n_base, len(x))
    baseline = np.median(x[:n_base])
    return (baseline - lo) / scale > 0.5


def find_trigger_rise_edge(trig, fs, riseThresh=0.6, fallThresh=0.4, min_width_ms=20,
                           start_idx=0, debug=0):
    '''
        Description: Detects triggers from the trigger channel of the data using
        hysteresis thresholding (Schmitt trigger) with a min pulse-width gate.

        If the electrode is wired backwards, the idle level is high and pulses go
        high→0. That case is detected from the early baseline and the raw trigger
        is negated before normalization and edge detection.

        <inputs>
        trig: emg channel that records triggers. For my case it should always be the first channel.

        fs: sampling rate of the data. Should be accessible from the .csv file of the emg.

        riseThresh: High threshold (0-1 after robust normalization). Crossing above this
        while LOW starts a pulse. Default 0.6.

        fallThresh: Low threshold (0-1). Crossing below this while HIGH ends a pulse.
        Must be < riseThresh. Default 0.4. The fall edge denotes the end of a trial.

        min_width_ms: Minimum accepted pulse width in milliseconds. Shorter pulses are rejected.

        start_idx: Optional sample index. Signal before this index is ignored for detection
        (useful when the start of a recording is corrupted). Returned rise/fall indices are
        still global indices into the original trig array. Default 0 (use full signal).

        debug: debug mode. Plots the detected triggers for eye inspection. Also prints
        sanity checks. Run with debug=1 the first time you detect triggers in a block.

        <outputs>
        riseIdx: Global indices where rising edges (trial starts) were detected.

        fallIdx: Global indices where falling edges (trial ends) were detected. Same length as riseIdx.
    '''
    if fallThresh >= riseThresh:
        raise ValueError("fallThresh must be < riseThresh for hysteresis.")

    trig = np.asarray(trig, dtype=float)
    start_idx = int(start_idx)
    if start_idx < 0 or start_idx >= len(trig):
        raise ValueError("start_idx must be in [0, len(trig)).")

    # Backwards electrode: idle is high; negate raw signal so pulses are low→high.
    inverted = _trigger_is_idle_high(trig, fs, start_idx=start_idx)
    if inverted:
        trig = -trig

    # Detect only on the usable suffix; normalize on this segment so early corruption
    # does not warp thresholds.
    x = trig[start_idx:]

    # Robust scale to ~[0, 1] (resistant to single-sample spikes)
    lo, hi = np.percentile(x, [1, 99])
    scale = hi - lo
    if scale < 1e-12:
        raise ValueError("Trigger channel has near-zero amplitude; cannot detect edges.")
    x = np.clip((x - lo) / scale, 0.0, 1.0)

    # Light median filter (~1 ms) to suppress sample-to-sample chatter
    k = max(1, int(round(0.001 * fs)))
    if k > 1:
        if k % 2 == 0:
            k += 1
        x = signal.medfilt(x, kernel_size=k)

    min_w = max(1, int(round(min_width_ms * 1e-3 * fs)))
    rise_list, fall_list = [], []
    state = 0  # 0 = low, 1 = high
    t_rise = None

    for i in range(1, len(x)):
        if state == 0 and x[i - 1] < riseThresh <= x[i]:
            state = 1
            t_rise = i
        elif state == 1 and x[i - 1] > fallThresh >= x[i]:
            if t_rise is not None and (i - t_rise) >= min_w:
                rise_list.append(t_rise)
                fall_list.append(i)
            state = 0
            t_rise = None

    # Convert segment-local indices back to global indices in the original trig array
    riseIdx = np.asarray(rise_list, dtype=int) + start_idx
    fallIdx = np.asarray(fall_list, dtype=int) + start_idx

    if debug:
        print("\n\n======== Trigger Detection Results: ======== \n")
        print("Polarity: {}".format(
            "INVERTED (raw negated; idle was high)" if inverted else "normal (idle low)"))
        print("Num Rise Trigger = {:d}".format(len(riseIdx)))
        print("Num Fall Triggers = {:d}".format(len(fallIdx)))
        print("Two numbers should be equal to the number of trials.\n")
        print("riseThresh={:.2f}, fallThresh={:.2f}, min_width_ms={:g}, start_idx={:d}".format(
            riseThresh, fallThresh, min_width_ms, start_idx))

        if len(fallIdx) == len(riseIdx) and len(riseIdx) > 0:
            widths_ms = (fallIdx - riseIdx) / fs * 1e3
            diffRiseFall = fallIdx - riseIdx
            numNegative = int(np.sum(diffRiseFall <= 0))
            print("\nNumber of non-positive fall-rise edges = {:d}".format(numNegative))
            print("This value should be 0.")
            print("Pulse width ms: min={:.1f}, median={:.1f}, max={:.1f}\n".format(
                widths_ms.min(), np.median(widths_ms), widths_ms.max()))

        # Plot analyzed segment with global sample indices on the x-axis
        global_n = np.arange(start_idx, start_idx + len(x))
        plt.figure(figsize=(22, 4))
        plt.plot(global_n, x, label='trig (normalized)')
        plt.axhline(riseThresh, color='r', linestyle='--', linewidth=0.8, label='riseThresh')
        plt.axhline(fallThresh, color='g', linestyle='--', linewidth=0.8, label='fallThresh')
        if start_idx > 0:
            plt.axvline(start_idx, color='k', linestyle=':', linewidth=0.8, label='start_idx')
        if len(riseIdx):
            plt.plot(riseIdx, x[riseIdx - start_idx], 'ro', label='Rising Edges', markersize=3)
        if len(fallIdx):
            plt.plot(fallIdx, x[fallIdx - start_idx], 'go', label='Falling Edges', markersize=3)
        plt.xlabel('sample index (global)')
        plt.legend(loc='upper right')
        plt.show()

    return riseIdx, fallIdx


def plot_raw_trigger(trig, fs=None, title='Raw trigger'):
    '''
        Description: Plots the raw (unprocessed) trigger channel for visual debugging.
        Unlike find_trigger_rise_edge(debug=1), this does not normalize or median-filter.

        <inputs>
        trig: Raw trigger channel array (e.g. Analog 1).

        fs: Optional sampling rate. If provided, x-axis is time in seconds;
        otherwise sample index is used.

        title: Plot title. Default 'Raw trigger'.
    '''
    trig = np.asarray(trig, dtype=float)
    n = np.arange(len(trig))

    plt.figure(figsize=(22, 4))
    if fs is not None:
        plt.plot(n / fs, trig, label='raw trigger', linewidth=0.8)
        plt.xlabel('time (s)')
    else:
        plt.plot(n, trig, label='raw trigger', linewidth=0.8)
        plt.xlabel('sample index')
    plt.ylabel('Volts')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def downsample_emg(emg, fs, target_fs=1000, debug=0):
    
    # resampled emg:
    emg_resampled = []

    # designing lowpass filter with cutoff frequency of target_fs/2:
    sos = signal.butter(2, int(target_fs/2), btype='lowpass', fs=fs, output='sos')

    # iterating through signals and downsampling with zero-phase anti-aliasing filter:
    for i in range(len(emg)):
        # selecting the emg signal of trial i:
        emg_trial = emg[i]

        # number of samples in the resampled signal:
        target_len = int(np.floor(len(emg_trial[:,0])*target_fs/fs))

        # making an empty array to contain the resampled signal:
        emg_trial_resampled = np.empty((target_len, np.shape(emg_trial)[1]))

        # iterating through emg channels:
        for ch in range(np.shape(emg_trial)[1]):
            # selecting the signal:
            sig = emg_trial[:,ch]

            # zero-phase low pass filtering the signal to avoid aliasing:
            sig = signal.sosfiltfilt(sos, sig)

            # downsampling the signal:
            sig_resampled = signal.resample(sig, target_len)

            # appending the resampled signal to the resampled trial:
            emg_trial_resampled[:,ch] = sig_resampled

            # plotting resampled against original signal:
            if debug and i==0 and ch==0:
                # time vector for plotting:
                t_orig = np.linspace(0,len(sig)/fs,len(sig))
                t_resampled = np.linspace(0,len(sig_resampled)/target_fs,len(sig_resampled))
                print("Original Signal Length = {:d}".format(len(sig)))
                print("Resampled Signal Length = {:d}".format(len(sig_resampled)))

                # plotting:
                plt.figure()
                plt.plot(t_orig, sig, label='Original Signal')
                plt.plot(t_resampled, sig_resampled, label='Resampled Signal')
                plt.legend()
                plt.show()
        
        # appending the resampled trial to the resampled emg:
        emg_resampled.append(emg_trial_resampled)

    return emg_resampled, target_fs

def bandpass_filter_emg(emg, fs, low=20, high=500, order=2, debug=0):
    '''
        Description: Zero-phase Butterworth bandpass filter for a single-trial
        EMG matrix (samples x channels), matching the per-trial workflow in
        preprocessing.ipynb.

        <inputs>
        emg: (N, C) array — samples x channels for one trial.

        fs: Sampling rate in Hz.

        low: High-pass cutoff in Hz. Default 20.

        high: Low-pass cutoff in Hz. Default 500. Clamped just below Nyquist
        if it would otherwise be invalid for fs.

        order: Butterworth filter order. Default 2.

        debug: If 1, plot original vs filtered for channel 0.

        <outputs>
        emg_filtered: (N, C) bandpass-filtered array (same shape as emg).
    '''
    emg = np.asarray(emg, dtype=float)
    if emg.ndim != 2:
        raise ValueError("emg must be a 2D array of shape (samples, channels).")

    nyq = 0.5 * fs
    high_eff = min(float(high), nyq * 0.999)
    if low <= 0 or high_eff <= low:
        raise ValueError(
            f"Invalid bandpass [{low}, {high}] Hz for fs={fs} (Nyquist={nyq:.3f})."
        )

    # sos form avoids numerical errors from high-order transfer-function filters
    sos = signal.butter(order, [low, high_eff], btype='bandpass', fs=fs, output='sos')
    emg_filtered = signal.sosfiltfilt(sos, emg, axis=0)

    if debug:
        sig = emg[:, 0]
        sig_filtered = emg_filtered[:, 0]
        t = np.linspace(0, len(sig) / fs, len(sig), endpoint=False)
        plt.figure()
        plt.plot(t, sig, label='Original Signal')
        plt.plot(t, sig_filtered, label='Filtered Signal')
        plt.xlabel('time (s)')
        plt.legend()
        plt.show()

    return emg_filtered


def low_pass_filter(emg, fs, cutoff=30, order=4, debug=0):
    '''
        Description: Zero-phase Butterworth low-pass filter for a single-trial
        EMG matrix (samples x channels).

        <inputs>
        emg: (N, C) array — samples x channels for one trial.

        fs: Sampling rate in Hz.

        cutoff: Low-pass cutoff in Hz. Default 30. Clamped just below Nyquist
        if it would otherwise be invalid for fs.

        order: Butterworth filter order. Default 4.

        debug: If 1, plot original vs filtered for channel 0.

        <outputs>
        emg_filtered: (N, C) low-pass-filtered array (same shape as emg).
    '''
    emg = np.asarray(emg, dtype=float)
    if emg.ndim != 2:
        raise ValueError("emg must be a 2D array of shape (samples, channels).")

    nyq = 0.5 * fs
    cutoff_eff = min(float(cutoff), nyq * 0.999)
    if cutoff_eff <= 0:
        raise ValueError(
            f"Invalid low-pass cutoff={cutoff} Hz for fs={fs} (Nyquist={nyq:.3f})."
        )

    sos = signal.butter(order, cutoff_eff, btype='lowpass', fs=fs, output='sos')
    emg_filtered = signal.sosfiltfilt(sos, emg, axis=0)

    if debug:
        sig = emg[:, 0]
        sig_filtered = emg_filtered[:, 0]
        t = np.linspace(0, len(sig) / fs, len(sig), endpoint=False)
        plt.figure()
        plt.plot(t, sig, label='Original Signal')
        plt.plot(t, sig_filtered, label='Low-pass Filtered')
        plt.xlabel('time (s)')
        plt.legend()
        plt.show()

    return emg_filtered


def rectify_emg(emg, debug=0):
    # rectified emg:
    emg_rectified = []

    # iterating through signals and rectifying:
    for i in range(len(emg)):
        # selecting the emg signal of trial i:
        emg_trial = emg[i]

        # making an empty array to contain the rectified signal:
        emg_trial_rectified = np.empty((np.shape(emg_trial)[0], np.shape(emg_trial)[1]))

        # iterating through emg channels:
        for ch in range(np.shape(emg_trial)[1]):
            # selecting the signal:
            sig = emg_trial[:,ch]

            # rectifying the signal:
            sig_rectified = np.absolute(sig)

            # appending the rectified signal to the emg_trial_rectified:
            emg_trial_rectified[:,ch] = sig_rectified

            # plotting rectified against original signal:
            if debug and i==0 and ch==0:
                # time vector for plotting:
                t = np.linspace(0,len(sig),len(sig))

                # plotting:
                plt.figure()
                plt.plot(t, sig, label='Original Signal')
                plt.plot(t, sig_rectified, label='Rectified Signal')
                plt.legend()
                plt.show()
        
        # appending the rectified trial:
        emg_rectified.append(emg_trial_rectified)

    return emg_rectified
