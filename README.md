# Casselton Fleet Guard — Straight Rows submission

A reference "baseline subject" build for the Bluestaq × NDSU Straight Rows
hackathon. Attributes every distinct degraded-guidance episode in the May 6
UTC evaluation window to one or more of five causes — space weather, jamming,
spoofing, equipment fault, multipath — by corroborating 1 Hz farm telemetry
with the SDA feeds that ship in the starter pack.

## Layout

```
starter/
├── data/                       dataset (telemetry + SDA feeds)
├── docs/                       official hackathon guide (PDF)
├── submission/
│   ├── baseline.py             starter baseline (unchanged)
│   ├── detector.py             improved detector (what we actually run)
│   ├── score_dev.py            rubric scorer for dev labels
│   └── validate_submission.py  schema validator (unchanged)
├── submission.json             the file that gets submitted
└── requirements.txt            pinned dependencies
```

Scratchpad / intermediate files live outside the repo and are not committed.

## Reproducing the submission

The starter pack ships its own venv at `starter/.venv` (Python 3.14, numpy,
pandas, matplotlib). Always invoke it by its explicit path.

```sh
# 1. Run the detector on the full dataset. Writes submission.json for the
#    eval window (May 6 UTC) and an auxiliary detection_report.json with
#    the dev window plus the per-bin feature grid.
starter/.venv/bin/python starter/submission/detector.py \
    --data starter/data \
    --output starter/submission.json \
    --report /tmp/detection_report.json \
    --github-handle koen-litt-bluestaq \
    --repo-url https://github.com/koen-litt-bluestaq/ndsu-koen-hackathon

# 2. Validate the submission against the official schema.
starter/.venv/bin/python starter/submission/validate_submission.py starter/submission.json
# -> prints "OK: 255 detection(s); submission is well-formed."

# 3. Score the detector against the 12 labeled dev-window episodes using
#    the rubric from page 4 of the official guide.
starter/.venv/bin/python starter/submission/score_dev.py \
    --detections /tmp/detection_report.json \
    --labels starter/data/labels/train_episodes.json
```

## Dev-label scorecard (page-4 rubric)

Diagnostic = 25 × macro-F1 + 15 × mean onset credit, out of 40.

| Detector              | Matches | Macro-F1 | Mean onset | Diagnostic |
| --------------------- | ------- | -------- | ---------- | ---------- |
| Starter baseline      | 7 / 12  | 0.508    | 0.497      | 20.15      |
| **This detector**     | **12 / 12** | **0.933** | **0.977** | **37.99**  |

Per-class F1: SPACE_WEATHER 1.00, JAMMING 1.00, SPOOFING 1.00,
EQUIPMENT_FAULT 1.00, MULTIPATH 0.667.

## Judge-facing demo

A single-page browser console that walks the judge through the problem,
the data feeds, the 24-hour timeline, each attributed cause with checkable
evidence, and the dev scoreboard. It renders dark or light, needs no
install, and ships all of its chart data inline from a saved detector run.

Link: see the "Casselton Fleet Guard" artifact published alongside this
repo.

## Notes

* Reported GNSS position is not trusted ground truth. Cross-track error
  comes from the simulated guidance monitor and is used as an independent
  cross-check, not a correction.
* Multipath bursts of ≤ 3 minutes are exempt from the extra-detection
  penalty per page 4 of the official guide; the submission therefore lists
  every detected multipath burst and the auto-diagnostic treats them
  accordingly.
* RFI reports are filtered at confidence ≥ 0.5 to avoid the dev-window
  RFI-004 (`conf 0.35`, free-text "may be storm-related receiver noise.
  Unconfirmed") painting a phantom jamming event during the storm onset.
* `baseline.py` is untouched so the two detectors can be diffed directly.
