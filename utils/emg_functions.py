import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

def find_trigger_rise_edge(trig, fs, riseThresh=0.6, fallThresh=0.4, min_width_ms=20,
                           start_idx=0, debug=0):
    '''
        Description: Detects triggers from the trigger channel of the data using
        hysteresis thresholding (Schmitt trigger) with a min pulse-width gate.

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

def filter_emg(emg, fs, low=20, high=500, order=2, debug=0):
    # filtered emg:
    emg_filtered = []

    # designing bandpass filter:
    sos = signal.butter(order, [low, high-1], btype='bandpass', fs=fs, output='sos')  # using 'sos' to avoid numerical errors

    # iterating through signals and filtering:
    for i in range(len(emg)):
        # selecting the emg signal of trial i:
        emg_trial = emg[i]

        # making an empty array to contain the filtered signal:
        emg_trial_filtered = np.empty((np.shape(emg_trial)[0], np.shape(emg_trial)[1]))

        # iterating through emg channels:
        for ch in range(np.shape(emg_trial)[1]):
            # selecting the signal:
            sig = emg_trial[:,ch]

            # filtering the signal:
            sig_filtered = signal.sosfiltfilt(sos, sig)

            # appending the filtered signal to the emg_trial_filtered:
            emg_trial_filtered[:,ch] = sig_filtered

            # plotting filtered against original signal:
            if debug and i==0 and ch==0:
                # time vector for plotting:
                t = np.linspace(0,len(sig)/fs,len(sig))

                # plotting:
                plt.figure()
                plt.plot(t, sig, label='Original Signal')
                plt.plot(t, sig_filtered, label='Filtered Signal')
                plt.legend()
                plt.show()
        
        # appending the resampled trial to the resampled emg:
        emg_filtered.append(emg_trial_filtered)

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
