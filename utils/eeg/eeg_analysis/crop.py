import numpy as np
import pandas as pd
from pathlib import Path

FS = 125
SKIP_SECONDS = 5  # EEG_plot_tk.eeg_data plots from raw_data[..., 125*5:]

# columns as indexed by EEG_plot_tk.eeg_data
EEG_COLS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
TRIGGER_COL = 31

# ECGFeatureExtractor.extract() does `filtered_trigger //= 7500` to get minutes.
# 7500 = FS * 60 samples per minute; keep the two in step.
SAMPLES_PER_MINUTE = FS * 60


def to_row(second):
    """A second on the plot's time axis -> a row index in the CSV."""
    return FS * (SKIP_SECONDS + second)


def count_rows(path):
    """Number of data rows in the CSV, without parsing it into floats.

    main.py needs the original length to rebuild the server-side trigger on the
    uncropped timeline; re-reading a 130 MB file through pandas just for a row
    count is not worth it.
    """
    rows = 0
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            rows += chunk.count(b'\n')
        f.seek(0, 2)
        size = f.tell()
        if size:
            f.seek(size - 1)
            if f.read(1) != b'\n':  # last line has no trailing newline
                rows += 1
    return rows


def trigger_rows(arr):
    """Raw row indices carrying a trigger, by the same rule CleanUpECG uses."""
    return np.where(arr[:, TRIGGER_COL] > 0)[0]


def trigger_minutes(rows):
    """Trigger rows -> minutes from the first trigger, rounded to the nearest.

    ECGFeatureExtractor.extract() floor-divides instead. On an uncropped file the
    two agree, because the operator fires each trigger just after a whole minute:
    the fractional parts measured across the sample recordings run 0.01-0.21, so
    floor and round both land on the same minute.

    Cropping breaks that premise. Deleting noise shifts every later trigger
    earlier by an arbitrary sub-minute amount, so the fractional part is no longer
    small -- and floor then drops a boundary by almost a full minute. Measured on
    data/2026-06-08-1531.csv, where auto_crop removes only 22.6 s (0.6% of the
    recording): trigger 1 lands at 14.764 min, floor gives 14, and main_analysis
    slices the phase at 14*60 = 840 s while the event is really at 885.8 s. That
    is a 46 s error -- 1.5 epochs of stimulation scored as baseline. Rounding
    gives 15 and cuts the error to 14 s, in line with the 0.6-4.5 s error the
    uncropped file already carries.

    The two conventions serve different timelines and are never compared: this
    one feeds main_analysis over the cropped temp.csv, ECG's feeds HRV over the
    original upload. Only the list lengths have to match, and they do -- see
    split_spans_at, which keeps every trigger sample alive across a cut.
    """
    rows = list(rows)
    if not rows:
        return []
    # round-half-up; Python's round() is banker's rounding, which would send
    # x.5 to the nearer even minute instead of consistently upward.
    return [int((row - rows[0] + SAMPLES_PER_MINUTE // 2) // SAMPLES_PER_MINUTE)
            for row in rows]


def split_spans_at(spans, protect_rows):
    """Carve protected rows out of the spans so those samples survive the cut.

    A trigger sitting inside a noisy span would otherwise be deleted, silently
    collapsing a phase boundary. Splitting the span around the row costs one
    extra seam and keeps the boundary.
    """
    protect = sorted(set(int(row) for row in protect_rows))
    pieces = []
    for start, end in sorted(spans):
        cursor = start
        for row in protect:
            if not start <= row < end:
                continue
            if cursor < row:
                pieces.append((cursor, row))
            cursor = row + 1  # the protected row itself is kept
        if cursor < end:
            pieces.append((cursor, end))

    # cut_spans needs a row before the span to match the DC offset against, so a
    # piece starting at row 0 is unusable.
    return [(start, end) for start, end in pieces if 1 <= start < end]


def cut_spans(arr, spans):
    """Delete row spans, keeping every EEG channel continuous across each seam.

    A plain deletion splices two samples whose DC offsets differ by thousands of
    uV. filtfilt is zero-phase, so that step rings for seconds on *both* sides of
    the seam. Shifting the tail of each channel by a constant closes the step, and
    the 0.5 Hz high-pass in EEG_plot_tk discards the constant anyway.
    """
    arr = arr.copy()
    for start, end in sorted(spans, reverse=True):
        if start < 1:
            raise ValueError(f"span {(start, end)} starts at row 0; nothing to match against")
        if not start < end <= len(arr):
            raise ValueError(f"span {(start, end)} is outside the file (rows 0..{len(arr)})")

        dropped = arr[start:end, TRIGGER_COL]
        if np.any(dropped != 0):
            print(f"  warning: dropping {int((dropped != 0).sum())} trigger sample(s)")

        if end < len(arr):  # nothing to rejoin when the span runs to the last row
            arr[end:, EEG_COLS] += arr[start - 1, EEG_COLS] - arr[end, EEG_COLS]
        arr = np.delete(arr, np.s_[start:end], axis=0)
    return arr


def crop_csv(path, spans_in_seconds, output_path=None):
    """Manual crop: delete hand-picked second ranges from a CSV.

    Public API for operator use, not part of the automatic pipeline -- main.py and
    gui.py go through auto_crop.auto_crop_csv instead. Seconds are on the plot's
    time axis (see to_row), i.e. 0 s is raw row 625.
    """
    path = Path(path)
    if output_path is None:
        output_path = path.parent / f"updated_{path.name}"
    output_path = Path(output_path)

    arr = pd.read_csv(path, sep='\t', header=None).to_numpy(dtype=float)

    spans = []
    for start_second, end_second in spans_in_seconds:
        start, end = to_row(start_second), to_row(end_second)
        print(f"cutting {start_second}-{end_second} s -> rows {start}:{end} ({end - start} samples)")
        spans.append((start, end))

    cropped = cut_spans(arr, spans)
    print(f"{len(arr)} -> {len(cropped)} rows ({(len(arr) - len(cropped)) / FS:.1f} s removed)")

    pd.DataFrame(cropped).to_csv(output_path, sep='\t', header=False, index=False,
                                 float_format='%.6f')

    print(f"Saved to: {output_path}")
