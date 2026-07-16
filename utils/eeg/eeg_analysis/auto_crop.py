import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt

from .crop import (FS, SKIP_SECONDS, EEG_COLS, cut_spans, split_spans_at,
                   trigger_rows, trigger_minutes)


def bandpass(x, lowcut=0.5, highcut=40.0, order=4):
    """Same band as EEG_plot_tk, applied down the sample axis."""
    b, a = butter(order, [lowcut / (FS / 2), highcut / (FS / 2)], btype='band')
    return filtfilt(b, a, x, axis=0)


def robust_threshold(x, k):
    """median + k * sigma, with sigma from the MAD so artifacts don't inflate it.

    Self-calibrating, so it survives a change of amplifier, gain or subject.
    """
    median = np.median(x)
    sigma = 1.4826 * np.median(np.abs(x - median))
    return median + k * sigma


def detect_noise(arr, amp_threshold, grad_k, window):
    """Flag noisy samples by amplitude OR gradient.

    Detection runs on filtered data: the raw channels carry DC offsets of several
    thousand uV, which would swamp any amplitude threshold.

    Two criteria, because they fail on opposite artifacts. The amplitude envelope
    averages |x| over a window, so it finds sustained noise but dilutes a sharp
    electrode pop -- a one-sample step barely moves a 1 s mean. The gradient does
    the reverse. Neither alone catches both.
    """
    eeg = bandpass(arr[FS * SKIP_SECONDS:, EEG_COLS] / 1e6) * 1e6

    amp = uniform_filter1d(np.abs(eeg), size=int(window * FS), axis=0).max(axis=1)

    # deliberately not smoothed: averaging a one-sample step over a window would
    # dilute it away, which is the very failure the gradient exists to catch.
    grad = np.abs(np.diff(eeg, axis=0, prepend=eeg[:1])).max(axis=1)
    grad_threshold = robust_threshold(grad, grad_k)

    print(f"amplitude: median {np.median(amp):7.1f}  threshold {amp_threshold:7.1f} uV"
          f"   -> {100 * (amp > amp_threshold).mean():5.2f}% of samples")
    print(f"gradient : median {np.median(grad):7.1f}  threshold {grad_threshold:7.1f} uV/sample"
          f" -> {100 * (grad > grad_threshold).mean():5.2f}% of samples")

    return (amp > amp_threshold) | (grad > grad_threshold), amp, grad


def find_spans(bad, margin, min_gap):
    """Turn a boolean mask into padded, merged spans of displayed samples."""
    if not bad.any():
        return []

    edges = np.diff(bad.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if bad[0]:
        starts = np.r_[0, starts]
    if bad[-1]:
        ends = np.r_[ends, len(bad)]

    pad, gap = int(margin * FS), int(min_gap * FS)
    merged = []
    for start, end in zip(starts, ends):
        start, end = max(start - pad, 0), min(end + pad, len(bad))
        if merged and start - merged[-1][1] < gap:
            merged[-1][1] = end  # close enough that a separate seam isn't worth it
        else:
            merged.append([start, end])
    return [tuple(span) for span in merged]


def auto_crop_csv(path, output_path=None, amp_threshold=200.0, grad_k=20.0, window=1.0,
                  margin=0.5, min_gap=1.0, max_removed_fraction=0.2):
    """Write a noise-cropped copy of `path` and report what happened.

    Returns (spans_in_raw_rows, eeg_trigger_minutes). The output file is always
    written, even when nothing is flagged -- the caller feeds it to the EEG
    pipeline and cannot be left without a file.

    `max_removed_fraction` caps how much of the recording the crop may delete.
    amp_threshold is a fixed 200 uV (only grad_k self-calibrates via the MAD), so
    a session with a bad electrode or a different gain can flag ~100% of samples.
    Cutting that would leave the EEG pipeline with too few rows to epoch, which is
    a worse outcome than analysing noisy data: before cropping existed the same
    file still produced a report. Past the cap we give up and pass the file
    through unchanged.

    Note: trigger minutes are floored on the *cropped* timeline, so removing noise
    ahead of a trigger can pull its minute back by one (e.g. 15.06 -> 14.76 -> 14).
    That is intended -- EEG phases follow the cropped timeline -- but it means EEG
    phase boundaries can sit up to ~1 min off the raw-timeline event.
    """
    path = Path(path)
    if output_path is None:
        output_path = path.parent / f"updated_{path.name}"
    output_path = Path(output_path)

    arr = pd.read_csv(path, sep='\t', header=None).to_numpy(dtype=float)
    bad, amp, grad = detect_noise(arr, amp_threshold, grad_k, window)

    spans = find_spans(bad, margin, min_gap)

    removed_fraction = sum(end - start for start, end in spans) / len(bad) if spans else 0.0
    if removed_fraction > max_removed_fraction:
        print(f"\nwould remove {100 * removed_fraction:.1f}% of the recording, over the "
              f"{100 * max_removed_fraction:.0f}% cap -- the thresholds do not fit this file "
              f"(bad electrode? different gain?).")
        print("giving up on cropping; writing the file through unchanged")
        spans = []

    if not spans:
        print("\nnothing flagged; writing the file through unchanged")
        cropped = arr
        raw_spans = []
    else:
        print(f"\n{len(spans)} noisy span(s):")
        for start, end in spans:
            print(f"  {start / FS:8.2f} - {end / FS:8.2f} s  ({(end - start) / FS:5.2f} s, "
                  f"peak amp {amp[start:end].max():7.1f} uV, peak grad {grad[start:end].max():7.1f})")

        removed = sum(end - start for start, end in spans)
        print(f"\nremoving {removed / FS:.1f} s ({100 * removed_fraction:.1f}% of the recording), "
              f"leaving {len(spans)} seam(s)")

        offset = FS * SKIP_SECONDS  # detection starts at raw row 625
        raw_spans = [(start + offset, end + offset) for start, end in spans]

        # A trigger inside a noisy span would be deleted along with it, losing a
        # phase boundary. Split the span around it instead.
        protected = split_spans_at(raw_spans, trigger_rows(arr))
        if len(protected) != len(raw_spans):
            print(f"  trigger protection: {len(raw_spans)} span(s) -> {len(protected)} "
                  f"piece(s); trigger samples inside noisy spans are kept")
        raw_spans = protected

        cropped = cut_spans(arr, raw_spans) if raw_spans else arr
        print(f"{len(arr)} -> {len(cropped)} rows "
              f"({(len(arr) - len(cropped)) / FS:.1f} s removed)")

    pd.DataFrame(cropped).to_csv(output_path, sep='\t', header=False, index=False,
                                 float_format='%.6f')
    print(f"Saved to: {output_path}")

    # Recomputed from the cropped array's own trigger channel, so it matches what
    # ECGFeatureExtractor would have produced for this file.
    eeg_trigger = trigger_minutes(trigger_rows(cropped))
    print(f"trigger (cropped timeline): {eeg_trigger}")

    return raw_spans, eeg_trigger
