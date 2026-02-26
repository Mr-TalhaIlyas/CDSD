import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from typing import Sequence, Tuple, Union, Optional, Dict, List

# ----------------------------
# Color utilities (Nature-ish, colorblind-friendly)
# ----------------------------
def tol_muted_palette() -> List[str]:
    # Paul Tol's "muted" qualitative palette (high-quality, colorblind safe)
    return [
        "#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
        "#DDCC77", "#CC6677", "#882255", "#AA4499", "#DDDDDD"
    ]

def okabe_ito_palette() -> List[str]:
    # Okabe-Ito palette (also colorblind safe)
    return ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

def make_color_map(
    class_names: Sequence[str],
    background: str = "background",
    palette: Optional[Sequence[str]] = None
) -> Dict[str, str]:
    """
    Map each class name to a color hex string.
    Background is assigned a light neutral gray.
    For many classes, falls back to matplotlib 'tab20' for extras.
    """
    class_names = list(class_names)

    if palette is None:
        # Start with very publication-friendly palettes
        base = (
            tol_muted_palette()[:-1] +
            ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"] +
            okabe_ito_palette()[:-1]
        )
        palette = base
    palette = list(palette)

    bg_color = "#E6E6E6"  # light gray background segment
    color_map = {}

    # Force background first (if present) so it gets the neutral color
    names_order = class_names.copy()
    if background in names_order:
        names_order.remove(background)
        names_order = [background] + names_order

    if len(names_order) <= len(palette):
        colors = palette[:len(names_order)]
    else:
        # Extend with tab20 if there are many classes
        cmap = plt.get_cmap("tab20")
        extra_needed = len(names_order) - len(palette)
        extra_colors = [mpl.colors.to_hex(cmap(i % cmap.N)) for i in range(extra_needed)]
        colors = palette + extra_colors

    for name, col in zip(names_order, colors):
        if name == background:
            color_map[name] = bg_color
        else:
            color_map[name] = col
    return color_map


# ----------------------------
# Segmentation helpers
# ----------------------------
def frame_labels_to_segments(labels: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Run-length encode per-frame labels into segments [start, end)."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    segments = []
    start = 0
    prev = int(labels[0])
    for i in range(1, len(labels)):
        if int(labels[i]) != prev:
            segments.append((start, i, prev))
            start = i
            prev = int(labels[i])
    segments.append((start, len(labels), prev))
    return segments

def segments_to_frame_labels(
    segments: Sequence[Tuple[int, int, int]],
    length: int,
    background_id: int = 0
) -> np.ndarray:
    """Create per-frame labels from segments [start, end)."""
    out = np.full(length, background_id, dtype=int)
    for s, e, lab in segments:
        s = max(0, min(int(s), length))
        e = max(0, min(int(e), length))
        if e > s:
            out[s:e] = int(lab)
    return out

def merge_short_segments(
    segments: List[Tuple[int, int, int]],
    min_len: int,
    background_id: Optional[int] = None
) -> List[Tuple[int, int, int]]:
    """
    Merge segments shorter than min_len into neighbors (reduces flicker).
    background_id (optional): tries not to swallow foreground into background unless unavoidable.
    """
    if min_len <= 1 or len(segments) <= 1:
        return segments

    segs = segments.copy()
    changed = True
    while changed:
        changed = False
        new = []
        i = 0
        while i < len(segs):
            s, e, lab = segs[i]
            if (e - s) < min_len and len(segs) > 1:
                prev_i = i - 1 if i > 0 else None
                next_i = i + 1 if i + 1 < len(segs) else None

                target = None
                if prev_i is not None and next_i is not None:
                    # Prefer merging into non-background if possible
                    if background_id is not None:
                        if segs[prev_i][2] == background_id and segs[next_i][2] != background_id:
                            target = next_i
                        else:
                            target = prev_i
                    else:
                        target = prev_i
                elif prev_i is not None:
                    target = prev_i
                elif next_i is not None:
                    target = next_i

                if target == prev_i:
                    ps, pe, pl = new[-1]
                    new[-1] = (ps, e, pl)
                    changed = True
                    i += 1
                    continue
                elif target == next_i:
                    ns, ne, nl = segs[next_i]
                    segs[next_i] = (s, ne, nl)
                    changed = True
                    i += 1
                    continue

            new.append(segs[i])
            i += 1

        # Merge adjacent same-label segments
        merged = []
        for s, e, lab in new:
            if merged and merged[-1][2] == lab:
                merged[-1] = (merged[-1][0], e, lab)
                changed = True
            else:
                merged.append((s, e, lab))
        segs = merged

    return segs


# ----------------------------
# Sliding-window -> per-frame aggregation
# ----------------------------
def aggregate_sliding_windows_to_frames(
    window_preds: Union[np.ndarray, Sequence[int]],
    window_size_s: float,
    stride_s: float,
    fps: float,
    num_frames: Optional[int] = None,
    window_start_times_s: Optional[Sequence[float]] = None,
    method: str = "vote",  # "vote" for hard labels, "prob" for (N,C) probabilities/logits
    background_id: int = 0,
    dtype_probs=np.float32,
    dtype_votes=np.int16,
) -> np.ndarray:
    """
    Convert sliding-window predictions into dense per-frame labels by overlap aggregation.

    method="vote": window_preds is (N,) class_id
    method="prob": window_preds is (N,C) probabilities/logits; we average over overlapping windows.

    This overlap aggregation is what gives you "maximum coverage" from 5s windows.
    """
    window_preds = np.asarray(window_preds)

    if window_start_times_s is None:
        starts_s = np.arange(len(window_preds), dtype=float) * float(stride_s)
    else:
        starts_s = np.asarray(window_start_times_s, dtype=float)
        if starts_s.shape[0] != len(window_preds):
            raise ValueError("window_start_times_s must have same length as window_preds")

    ends_s = starts_s + float(window_size_s)

    if num_frames is None:
        total_s = float(np.max(ends_s)) if len(ends_s) else 0.0
        num_frames = int(np.ceil(total_s * float(fps)))
    num_frames = int(num_frames)

    if method == "prob":
        if window_preds.ndim != 2:
            raise ValueError("For method='prob', window_preds must be (N, C)")
        N, C = window_preds.shape

        # Difference-array trick for efficiency: O(N*C + T*C) instead of O(N*T)
        diff = np.zeros((C, num_frames + 1), dtype=dtype_probs)
        diff_count = np.zeros(num_frames + 1, dtype=dtype_probs)

        for i in range(N):
            s_f = int(round(starts_s[i] * fps))
            e_f = int(round(ends_s[i] * fps))
            if e_f <= 0 or s_f >= num_frames:
                continue
            s_f = max(0, min(s_f, num_frames))
            e_f = max(0, min(e_f, num_frames))
            if e_f <= s_f:
                continue

            v = window_preds[i].astype(dtype_probs, copy=False)
            diff[:, s_f] += v
            diff[:, e_f] -= v
            diff_count[s_f] += 1.0
            diff_count[e_f] -= 1.0

        sum_probs = np.cumsum(diff, axis=1)[:, :num_frames]  # (C,T)
        count = np.cumsum(diff_count)[:num_frames]            # (T,)

        avg = sum_probs / np.maximum(count, 1e-6)
        frame_ids = np.argmax(avg, axis=0).astype(int)
        frame_ids[count <= 0] = background_id
        return frame_ids

    elif method == "vote":
        if window_preds.ndim != 1:
            raise ValueError("For method='vote', window_preds must be (N,)")

        N = window_preds.shape[0]
        C = int(np.max(window_preds)) + 1 if N else (background_id + 1)

        diff = np.zeros((C, num_frames + 1), dtype=dtype_votes)
        for i in range(N):
            k = int(window_preds[i])
            s_f = int(round(starts_s[i] * fps))
            e_f = int(round(ends_s[i] * fps))
            if e_f <= 0 or s_f >= num_frames:
                continue
            s_f = max(0, min(s_f, num_frames))
            e_f = max(0, min(e_f, num_frames))
            if e_f <= s_f:
                continue
            diff[k, s_f] += 1
            diff[k, e_f] -= 1

        votes = np.cumsum(diff, axis=1)[:, :num_frames]  # (C,T)
        frame_ids = np.argmax(votes, axis=0).astype(int)
        covered = votes.sum(axis=0) > 0
        frame_ids[~covered] = background_id
        return frame_ids

    else:
        raise ValueError("method must be 'vote' or 'prob'")


# ----------------------------
# Plotting (GT vs Pred bars)
# ----------------------------
def plot_action_segmentation_bars(
    gt_segments_s: Sequence[Tuple[float, float, str]],
    pred_segments_s: Sequence[Tuple[float, float, str]],
    class_names: Sequence[str],
    fps: float,
    background: str = "background",
    time_unit: str = "s",  # "s" or "frames"
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 2.4),
    show_legend: bool = True,
    legend_max_items: int = 20,
    legend_ncol: int = 6,
    show_gt_boundaries: bool = True,
    boundary_alpha: float = 0.25,
    savepath: Optional[str] = None,
    dpi: int = 300,
):
    """
    Clean "CVPR-style" GT vs Pred timeline plot with color-coded action segments.
    Input segments are in seconds: [(start_s, end_s, label_name), ...]
    """
    class_names = list(class_names)
    if background not in class_names:
        raise ValueError(f"background='{background}' must be in class_names")

    color_map = make_color_map(class_names, background=background)

    # Convert segments from seconds -> frames internally
    def to_frame_segments(segments_s):
        out = []
        max_end = 0
        for s_s, e_s, lab in segments_s:
            s_f = int(round(float(s_s) * fps))
            e_f = int(round(float(e_s) * fps))
            out.append((s_f, e_f, lab))
            max_end = max(max_end, e_f)
        return out, max_end

    gt_f, gt_T = to_frame_segments(gt_segments_s)
    pr_f, pr_T = to_frame_segments(pred_segments_s)
    T = max(gt_T, pr_T)

    def x(frames: int) -> float:
        return frames / fps if time_unit == "s" else frames

    x_max = x(T)

    style = {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with mpl.rc_context(style):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        bar_h = 0.35
        y_gt = 0.55
        y_pr = 0.10

        def draw_row(segs, y, row_label):
            for s_f, e_f, lab in segs:
                if e_f <= s_f:
                    continue
                ax.broken_barh(
                    [(x(s_f), x(e_f - s_f))],
                    (y, bar_h),
                    facecolors=color_map.get(lab, "#CCCCCC"),
                    edgecolors="none",
                )
            ax.text(-0.01 * x_max, y + bar_h / 2, row_label, va="center", ha="right")

        draw_row(gt_f, y_gt, "GT")
        draw_row(pr_f, y_pr, "Pred")

        # Optional: GT boundary guides (helps judge start/end alignment)
        if show_gt_boundaries:
            boundaries = sorted({s for s, _, lab in gt_f if lab != background} |
                                {e for _, e, lab in gt_f if lab != background})
            for b in boundaries:
                ax.axvline(x(b), linewidth=0.8, linestyle="--", alpha=boundary_alpha)

        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, x_max)
        ax.set_yticks([])
        ax.set_xlabel("Time (s)" if time_unit == "s" else "Frame")
        if title:
            ax.set_title(title, pad=6)

        ax.grid(axis="x", linewidth=0.6, alpha=0.25)

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_alpha(0.4)

        if show_legend:
            used = {lab for _, _, lab in (gt_f + pr_f)}
            used_ordered = [c for c in class_names if c in used and c != background]
            if background in used:
                used_ordered.append(background)

            if len(used_ordered) > legend_max_items:
                used_ordered = used_ordered[:legend_max_items]

            handles = [Patch(facecolor=color_map[c], edgecolor="none", label=c) for c in used_ordered]
            ax.legend(
                handles=handles,
                ncol=min(legend_ncol, max(1, len(handles))),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.35),
                frameon=False,
                columnspacing=1.0,
                handlelength=1.2,
                handletextpad=0.4,
            )

        if savepath:
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        return fig, ax


# ----------------------------
# High-level convenience wrapper (your main entry point)
# ----------------------------
def plot_gt_vs_pred_from_sliding_windows(
    gt_segments_s: Sequence[Tuple[float, float, str]],
    window_preds: Union[Sequence[Union[int, str]], np.ndarray],
    class_names: Sequence[str],
    window_size_s: float = 5.0,
    stride_s: float = 1.0,
    fps: float = 30.0,
    method: str = "vote",        # "vote" if you have hard labels, "prob" if you have (N,C) probs/logits
    background: str = "background",
    min_segment_s: float = 0.0,  # merge segments shorter than this (reduces flicker)
    time_unit: str = "s",
    title: Optional[str] = None,
    **plot_kwargs
):
    """
    GT: list of (start_s, end_s, label_name)
    Predictions: sliding window outputs over time.
      - method="vote": window_preds is list/array of class ids OR class names (length N)
      - method="prob": window_preds is array (N, C) probabilities/logits

    Produces: GT vs Pred segmentation bar plot.
    """
    class_names = list(class_names)
    class_to_id = {c: i for i, c in enumerate(class_names)}
    if background not in class_to_id:
        raise ValueError(f"background='{background}' must be in class_names")
    bg_id = class_to_id[background]

    total_s = max(e for _, e, _ in gt_segments_s) if gt_segments_s else 0.0
    num_frames = int(np.ceil(total_s * fps))

    if method == "vote":
        # convert window outputs to ids if they are strings
        if isinstance(window_preds, np.ndarray) and window_preds.ndim == 1 and np.issubdtype(window_preds.dtype, np.integer):
            pred_ids = window_preds.astype(int)
        else:
            pred_ids = np.array(
                [class_to_id[p] if isinstance(p, str) else int(p) for p in window_preds],
                dtype=int
            )
        frame_pred = aggregate_sliding_windows_to_frames(
            pred_ids, window_size_s, stride_s, fps,
            num_frames=num_frames, method="vote", background_id=bg_id
        )

    elif method == "prob":
        probs = np.asarray(window_preds, dtype=np.float32)
        frame_pred = aggregate_sliding_windows_to_frames(
            probs, window_size_s, stride_s, fps,
            num_frames=num_frames, method="prob", background_id=bg_id
        )
    else:
        raise ValueError("method must be 'vote' or 'prob'")

    # Optional: remove flicker by merging short segments
    if min_segment_s and min_segment_s > 0:
        segs = frame_labels_to_segments(frame_pred)
        segs = merge_short_segments(segs, min_len=int(round(min_segment_s * fps)), background_id=bg_id)
        frame_pred = segments_to_frame_labels(segs, length=num_frames, background_id=bg_id)

    # Convert dense per-frame pred to segments in seconds
    pred_segments_s = [
        (s / fps, e / fps, class_names[int(lab)])
        for s, e, lab in frame_labels_to_segments(frame_pred)
    ]

    fig, ax = plot_action_segmentation_bars(
        gt_segments_s=gt_segments_s,
        pred_segments_s=pred_segments_s,
        class_names=class_names,
        fps=fps,
        background=background,
        time_unit=time_unit,
        title=title,
        **plot_kwargs
    )
    return fig, ax


# ----------------------------
# Dummy demo (shows how it works)
# ----------------------------
if __name__ == "__main__":
    import math, random

    # Example classes (replace with your dataset classes)
    classes = ["background", "approach", "jump", "spin", "landing"]
    fps = 30.0

    # GT segments in seconds (replace with your GT)
    gt_segments_s = [
        (0, 5, "background"),
        (5, 15, "approach"),
        (15, 22, "jump"),
        (22, 35, "spin"),
        (35, 45, "approach"),
        (45, 50, "jump"),
        (50, 55, "landing"),
        (55, 60, "background"),
    ]
    total_s = 60.0

    # Sliding window outputs (5s window, 1s stride)
    window_size_s = 5.0
    stride_s = 1.0
    N = int(math.floor((total_s - window_size_s) / stride_s)) + 1
    starts = [i * stride_s for i in range(N)]

    # helper: GT label at time t
    def label_at_time(segments, t):
        for s, e, lab in segments:
            if s <= t < e:
                return lab
        return segments[-1][2]

    true_labels = [label_at_time(gt_segments_s, s + window_size_s / 2) for s in starts]

    # create a noisy model prediction per window (dummy)
    rng = random.Random(7)
    pred_labels = []
    for lab in true_labels:
        if rng.random() < 0.18:
            pred_labels.append(rng.choice([c for c in classes if c != lab]))
        else:
            pred_labels.append(lab)

    # Plot
    plot_gt_vs_pred_from_sliding_windows(
        gt_segments_s=gt_segments_s,
        window_preds=pred_labels,          # list of class names (or ids)
        class_names=classes,
        window_size_s=window_size_s,
        stride_s=stride_s,
        fps=fps,
        method="vote",
        min_segment_s=0.4,                 # merge flicker shorter than 0.4s
        title="GT vs sliding-window aggregated prediction (5s window, 1s stride)",
        figsize=(12, 2.6),
        show_gt_boundaries=True,           # helps judge start/end alignment
        legend_ncol=6
    )
    plt.show()
#%%