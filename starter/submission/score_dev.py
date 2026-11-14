#!/usr/bin/env python3
"""Score detector output against the development labels using the
rubric from page 4 of the official hackathon guide.

Rules implemented here (verbatim from the guide):
  * Temporal IoU >= 0.30 AND tractor-set IoU >= 0.50 for a match.
  * Same-class matches take priority; then maximise number of matches and
    total time overlap.
  * Onset credit: full within 5 min of true start, declines linearly to
    zero at 60 min; zero for miss or wrong class.
  * Final diagnostic: 25 * macro-F1 + 15 * mean onset credit (out of 40).
  * Recurrent multipath bursts <=3 min are exempt from the extra-detection
    penalty except duplicates of the scored representative window.

Usage::
    starter/.venv/bin/python starter/submission/score_dev.py \\
        --detections out/dev_detections.json \\
        --labels starter/data/labels/train_episodes.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

CLASSES = ("SPACE_WEATHER", "JAMMING", "SPOOFING", "EQUIPMENT_FAULT", "MULTIPATH")


def iso_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@dataclass
class Interval:
    start: datetime
    end: datetime
    cls: str
    tractors: set

    @property
    def dur(self) -> float:
        return (self.end - self.start).total_seconds()


def temporal_iou(a: Interval, b: Interval) -> float:
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    inter = max(0.0, (hi - lo).total_seconds())
    union = max((a.end - a.start).total_seconds(),
                (b.end - b.start).total_seconds(), 1e-9)
    union = (a.end - a.start).total_seconds() + (b.end - b.start).total_seconds() - inter
    if union <= 0:
        return 0.0
    return inter / union


def tractor_iou(a: Interval, b: Interval) -> float:
    inter = len(a.tractors & b.tractors)
    union = len(a.tractors | b.tractors)
    return inter / union if union else 0.0


def load_detections(path: Path) -> list[Interval]:
    j = json.loads(Path(path).read_text())
    dets = j.get("detections", j)  # accept submission.json shape or a list
    return [Interval(iso_to_dt(d["start"]), iso_to_dt(d["end"]),
                     d["class"], set(d["tractors"])) for d in dets]


def load_labels(path: Path) -> list[Interval]:
    j = json.loads(Path(path).read_text())
    eps = j["episodes"]
    return [Interval(iso_to_dt(e["start"]), iso_to_dt(e["end"]),
                     e["class"], set(e["tractors"])) for e in eps]


def match(detections: list[Interval],
          labels: list[Interval]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy matcher that honours the same-class priority rule and then
    maximises total time overlap among the remaining candidates."""
    # Build candidate list per policy
    same_cls_cands = []
    cross_cls_cands = []
    for di, d in enumerate(detections):
        for li, g in enumerate(labels):
            if temporal_iou(d, g) < 0.30 or tractor_iou(d, g) < 0.50:
                continue
            overlap_s = max(
                0.0,
                (min(d.end, g.end) - max(d.start, g.start)).total_seconds())
            if d.cls == g.cls:
                same_cls_cands.append((di, li, overlap_s))
            else:
                cross_cls_cands.append((di, li, overlap_s))
    matched_d = set()
    matched_l = set()
    pairs: list[tuple[int, int]] = []
    for pool in (same_cls_cands, cross_cls_cands):
        for di, li, _ in sorted(pool, key=lambda x: -x[2]):
            if di in matched_d or li in matched_l:
                continue
            matched_d.add(di)
            matched_l.add(li)
            pairs.append((di, li))
    unmatched_d = [di for di in range(len(detections)) if di not in matched_d]
    unmatched_l = [li for li in range(len(labels)) if li not in matched_l]
    return pairs, unmatched_d, unmatched_l


def onset_credit(det: Interval, truth: Interval) -> float:
    """Full credit within 5 min of truth.start; linear to 0 at 60 min."""
    diff = abs((det.start - truth.start).total_seconds()) / 60.0
    if diff <= 5:
        return 1.0
    if diff >= 60:
        return 0.0
    return (60 - diff) / 55.0


def score(detections: list[Interval],
          labels: list[Interval]) -> dict:
    pairs, unmatched_d, unmatched_l = match(detections, labels)

    # Multipath FP exemption: extra multipath bursts <=3 min aren't FP unless
    # they overlap the scored representative window of another multipath detect
    multipath_scored_windows = [
        (detections[di].start, detections[di].end)
        for di, li in pairs if detections[di].cls == "MULTIPATH"
    ]
    exempt = set()
    for di in unmatched_d:
        d = detections[di]
        if d.cls != "MULTIPATH":
            continue
        if d.dur > 180:  # >3 min: not exempt
            continue
        overlaps_scored = any(
            not (d.end <= ws or d.start >= we) for ws, we in multipath_scored_windows)
        if not overlaps_scored:
            exempt.add(di)

    # Per-class TP / FP / FN with same-class requirement
    tp = {c: 0 for c in CLASSES}
    fp = {c: 0 for c in CLASSES}
    fn = {c: 0 for c in CLASSES}
    pairs_scored: list[tuple[int, int, float]] = []
    for di, li in pairs:
        d = detections[di]
        g = labels[li]
        if d.cls == g.cls:
            tp[d.cls] += 1
            pairs_scored.append((di, li, onset_credit(d, g)))
        else:
            fp[d.cls] += 1  # wrong class = false positive for predicted class
            fn[g.cls] += 1  # and false negative for true class
            pairs_scored.append((di, li, 0.0))
    for di in unmatched_d:
        if di in exempt:
            continue
        fp[detections[di].cls] += 1
    for li in unmatched_l:
        fn[labels[li].cls] += 1

    # Macro-F1
    f1s = {}
    for c in CLASSES:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1s[c] = 2 * p * r / (p + r) if (p + r) else 0.0
    macro_f1 = sum(f1s.values()) / len(CLASSES)

    # Onset credit mean over truth episodes
    onsets = []
    for di, li, credit in pairs_scored:
        if detections[di].cls == labels[li].cls:
            onsets.append(credit)
    # Episodes that weren't matched count as 0 onset credit
    onsets.extend([0.0] * len(unmatched_l))
    mean_onset = sum(onsets) / len(onsets) if onsets else 0.0

    diagnostic = 25 * macro_f1 + 15 * mean_onset

    return {
        "macro_f1": macro_f1,
        "per_class_f1": f1s,
        "per_class_tp": tp,
        "per_class_fp": fp,
        "per_class_fn": fn,
        "mean_onset_credit": mean_onset,
        "diagnostic_score_out_of_40": diagnostic,
        "num_detections": len(detections),
        "num_labels": len(labels),
        "num_matched": len(pairs),
        "same_class_matches": sum(
            1 for di, li in pairs if detections[di].cls == labels[li].cls),
        "exempt_multipath": len(exempt),
        "matches": [
            {"det_idx": di, "label_idx": li,
             "det_class": detections[di].cls, "label_class": labels[li].cls,
             "det_start": detections[di].start.isoformat(),
             "label_start": labels[li].start.isoformat(),
             "onset_credit": onset_credit(detections[di], labels[li])
             if detections[di].cls == labels[li].cls else 0.0}
            for di, li in pairs],
        "unmatched_detections": [
            {"idx": di, "class": detections[di].cls,
             "start": detections[di].start.isoformat(),
             "end": detections[di].end.isoformat(),
             "tractors": sorted(detections[di].tractors)}
            for di in unmatched_d if di not in exempt],
        "unmatched_labels": [
            {"idx": li, "class": labels[li].cls,
             "start": labels[li].start.isoformat(),
             "end": labels[li].end.isoformat(),
             "tractors": sorted(labels[li].tractors)}
            for li in unmatched_l],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--labels",
                        default="starter/data/labels/train_episodes.json")
    parser.add_argument("--json", action="store_true",
                        help="print full report as JSON")
    args = parser.parse_args()
    dets = load_detections(Path(args.detections))
    labels = load_labels(Path(args.labels))
    report = score(dets, labels)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"detections: {report['num_detections']}, "
              f"labels: {report['num_labels']}, "
              f"same-class matches: {report['same_class_matches']}, "
              f"exempt multipath FPs: {report['exempt_multipath']}")
        print("per-class F1:")
        for c, v in report["per_class_f1"].items():
            tp = report["per_class_tp"][c]
            fp = report["per_class_fp"][c]
            fn = report["per_class_fn"][c]
            print(f"  {c:16s}  F1={v:.3f}  TP={tp}  FP={fp}  FN={fn}")
        print(f"macro-F1:              {report['macro_f1']:.3f}")
        print(f"mean onset credit:     {report['mean_onset_credit']:.3f}")
        print(f"diagnostic score / 40: {report['diagnostic_score_out_of_40']:.2f}")


if __name__ == "__main__":
    main()
